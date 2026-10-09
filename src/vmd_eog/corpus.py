"""Source-isolated controlled mixtures and real-data proxies. Run on Kaggle."""
import json
from pathlib import Path
import shutil
import tarfile
import tempfile
import urllib.request
import numpy as np
from .contracts import canonical_hash
from .data import channel_metadata
from .io import atomic_json, read_jsonl, sha256_file, write_jsonl
from .posterior import boundary_valid
from .reference import lag_matrix
from .signal import preprocess, calibration_trials


def unique_parent(input_root,name):
    paths=list(Path(input_root).rglob(name))
    if len(paths)!=1: raise ValueError(f"Attach exactly one {name} parent artifact, found {len(paths)}")
    return paths[0]


def safe_download(url,path):
    request=urllib.request.Request(url,headers={"User-Agent":"VMD-EOG-Research/0.1"})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request,timeout=120) as response,Path(path).open("wb") as target:
                shutil.copyfileobj(response,target,length=8*1024*1024)
            if Path(path).stat().st_size<1024: raise ValueError("Empty source archive")
            return
        except Exception:
            if attempt==2: raise


def prepare_lemon(source,temp_root,config,profile):
    import mne
    folder=Path(temp_root)/source["participant"]; folder.mkdir()
    archive=folder/"raw.tar.gz"; safe_download(source["url"],archive)
    digest=sha256_file(archive)
    with tarfile.open(archive) as stream:
        # Python extraction filter rejects links, device files and escaping paths.
        stream.extractall(folder,filter="data")
    headers=sorted(folder.rglob("*.vhdr"))
    if len(headers)!=1: raise ValueError("Expected one raw BrainVision recording per participant")
    raw=mne.io.read_raw_brainvision(headers[0],preload=False,verbose="ERROR")
    eog=[name for name in raw.ch_names if "EOG" in name.upper()]
    if not eog: raise ValueError("Raw LEMON requires identifiable ocular reference")
    raw.set_channel_types({name:"eog" for name in eog},verbose="ERROR")
    picks=mne.pick_types(raw.info,eeg=True,eog=False,exclude="bads")
    names=[raw.ch_names[i] for i in picks]
    duration=raw.n_times/raw.info["sfreq"]
    cal_duration=min(120.,max(10.,.2*duration)); cut=int(np.ceil(cal_duration*raw.info["sfreq"]))
    if raw.n_times-cut<config["window"]*raw.info["sfreq"]/200: raise ValueError("Insufficient LEMON scoring data")
    allpicks=list(picks)+[raw.ch_names.index(n) for n in eog]
    calibration=preprocess(raw.get_data(picks=allpicks,start=0,stop=cut),raw.info["sfreq"])
    scoring=preprocess(raw.get_data(picks=allpicks,start=cut),raw.info["sfreq"])
    count=len(picks); candidates=[]
    # Fixed low-ocular eligibility, based solely on recipient EOG vs its own
    # calibration. It never depends on a denoiser's performance.
    baseline=np.sqrt(np.mean(calibration[count:]**2))
    for start in range(0,scoring.shape[-1]-config["window"]+1,config["hop"]):
        references=scoring[count:,start:start+config["window"]]
        burden=float(np.sqrt(np.mean(references**2)))
        if burden<=baseline:
            candidates.append({"start":start,"eeg":scoring[:count,start:start+config["window"]],"ocular_rms":burden})
    if len(candidates)<2: raise ValueError("No sufficient low-ocular recipient windows")
    # Spread retained windows chronologically; never call the unknown native
    # signal biologically artifact-free.
    limit=4 if profile=="pilot" else 12
    positions=np.linspace(0,len(candidates)-1,min(limit,len(candidates)),dtype=int)
    positions=list(dict.fromkeys(positions.tolist()))
    record={"participant":"lemon:"+source["participant"],"channel_names":names,
        "native_fs":raw.info["sfreq"],"archive_sha256":digest,"url":source["url"],
        "source_header_sha256":sha256_file(headers[0]),"eog_channels":eog,
        "reference":"original raw publisher reference retained","units":"volts (MNE conversion)",
        "low_ocular_policy":"scoring RMS <= recipient calibration RMS; fixed before algorithm evaluation",
        "calibration_end_sample_native":cut,"native_clean_status":"unknown; controlled target is retained recipient EEG"}
    raw.close()
    return record,calibration[:count],[candidates[i] for i in positions]


def donor_mapping(calibration,lags=(-10,0,10),penalty=.01):
    references=calibration["references"]; eeg=calibration["eeg"]
    design,valid=lag_matrix(references,lags)
    valid &= boundary_valid(eeg.shape[-1],calibration["boundaries"],lags)
    # Only calibration trials; lagged columns never cross trial joins.
    mean=design[:,valid].mean(axis=1); scale=np.maximum(design[:,valid].std(axis=1),1e-12)
    standardized=(design-mean[:,None])/scale[:,None]
    covariance=standardized[:,valid]@standardized[:,valid].T/valid.sum()
    cross=standardized[:,valid]@(eeg[:,valid]-eeg[:,valid].mean(axis=1,keepdims=True)).T/valid.sum()
    coefficients=np.linalg.solve(covariance+penalty*np.eye(len(design)),cross).T
    return {"lags":list(lags),"mean":mean,"scale":scale,"coefficients":coefficients,
            "donor_calibration_digest":canonical_hash({"shape":list(eeg.shape),"eeg_sum":float(eeg.sum()),"ref_sum":float(references.sum())})}


def apply_mapping(mapping,references):
    design,valid=lag_matrix(references,mapping["lags"])
    field=mapping["coefficients"]@((design-mapping["mean"][:,None])/mapping["scale"][:,None])
    # Unavailable delayed data remains zero and is disclosed in the recipe.
    field[:,~valid]=0
    return field


def build_corpus(input_root,output,config,profile):
    parent=unique_parent(input_root,"audit_summary.json").parent
    state=json.loads((parent/"execution_state.json").read_text())
    if state["status"]!="complete": raise ValueError("Audit parent incomplete")
    split=json.loads((parent/"split_manifest.json").read_text())
    external=json.loads((parent/"external_sources.json").read_text())
    atomic_json(output/"split_manifest.json",split)
    arrays=output/"arrays"; arrays.mkdir()
    calibration_output=output/"calibration"; calibration_output.mkdir()
    donors=[]; examples=[]
    for record_path in sorted((parent/"source_cache").glob("*/record.json")):
        record=json.loads(record_path.read_text()); directory=record_path.parent
        target=calibration_output/record["record_id"]; target.mkdir()
        shutil.copy2(directory/"calibration.npz",target/"calibration.npz")
        shutil.copy2(record_path,target/"record.json")
        if record["dataset"]=="osf":
            cal=dict(np.load(directory/"calibration.npz",allow_pickle=False))
            donors.append({"record":record,"directory":directory,"mapping":donor_mapping(cal)})
        for i,window in enumerate(record["windows"]):
            name=record["record_id"]+f"-native-{i:03d}.npz"
            shutil.copy2(directory/window["array"],arrays/name)
            examples.append({"example_id":name[:-4],"array_path":"arrays/"+name,"dataset":record["dataset"],
                "recipient":record["participant"] or record["record_id"],"donor":None,"partition":record["partition"],
                "condition":window["condition"],"target_kind":"real_proxy" if record["dataset"]=="osf" else "legacy_paired",
                "calibration":"calibration/"+record["record_id"]+"/calibration.npz"})
    sources=[s for s in external["lemon"] if split["partitions"]["lemon"]["lemon:"+s["participant"]]["role"]=="development"]
    if profile=="pilot": sources=sources[:config["pilot"]["lemon_participants"]]
    temporary=tempfile.mkdtemp(prefix="lemon-raw-"); recipients=[]; failures=[]
    for source in sources:
        print("LEMON_DOWNLOAD",source["participant"],flush=True)
        try:
            record,cal,windows=prepare_lemon(source,temporary,config,profile)
            partition=split["partitions"]["lemon"][record["participant"]]
            record["partition"]=partition; recipients.append(record)
            allowed=[d for d in donors if d["record"]["partition"]["role"]==partition["role"] and d["record"]["partition"]["fold"]==partition["fold"]]
            if not allowed: raise ValueError("No donor from same frozen crossfit bucket")
            for recipient_index,recipient in enumerate(windows):
                donor=allowed[recipient_index%len(allowed)]
                meta=dict(np.load(donor["directory"]/"calibration.npz",allow_pickle=False))
                donor_names=meta["channel_names"].tolist(); lookup={n.upper():i for i,n in enumerate(donor_names)}
                common=[(i,lookup[n.upper()]) for i,n in enumerate(record["channel_names"]) if n.upper() in lookup]
                if len(common)<8: raise ValueError("Insufficient exact-name montage intersection")
                rids,dids=zip(*common); names=[record["channel_names"][i] for i in rids]
                target=recipient["eeg"][list(rids)]; metadata=channel_metadata(names)
                calid=f"{source['participant']}-{donor['record']['record_id']}"
                caldir=calibration_output/calid; caldir.mkdir(exist_ok=True)
                # Controlled calibration uses the unscored recipient prefix plus
                # the donor's unscored ocular calibration. No scored target fits
                # an ICA/MWF operator, and both EEG sources use recipient units.
                donorcal={key:value for key,value in meta.items() if key in ("eeg","references","boundaries","trial_types")}
                duration=min(cal.shape[-1],donorcal["references"].shape[-1])
                calibration_field=apply_mapping(donor["mapping"],donorcal["references"][:,:duration])[list(dids)]
                recipient_cal=cal[list(rids),:duration]
                calibration_scale=np.linalg.norm(recipient_cal)/max(np.linalg.norm(calibration_field),1e-20)
                donorcal["eeg"]=recipient_cal+calibration_scale*calibration_field
                donorcal["references"]=donorcal["references"][:,:duration]
                donorcal["trial_types"]=donorcal["trial_types"][:duration]
                donorcal["boundaries"]=donorcal["boundaries"][donorcal["boundaries"]<duration]
                np.savez_compressed(caldir/"calibration.npz",**donorcal,**metadata)
                for condition in ("clean","blink","lateral","mixed"):
                    ocular_windows=[w for w in donor["record"]["windows"] if w["condition"] in (("blink","lateral") if condition in ("mixed","clean") else (condition,))]
                    if not ocular_windows: continue
                    selected=ocular_windows[recipient_index%len(ocular_windows)]
                    ocular=dict(np.load(donor["directory"]/selected["array"],allow_pickle=False))
                    references=ocular["references"].copy()
                    if condition=="mixed":
                        other=[w for w in ocular_windows if w["condition"]!=selected["condition"]]
                        if not other: continue
                        refs2=np.load(donor["directory"]/other[0]["array"],allow_pickle=False)["references"]
                        references=(references+refs2)/np.sqrt(2.)
                    field=apply_mapping(donor["mapping"],references)[list(dids)]
                    for snr in ([None] if condition=="clean" else [-5.,0.,5.]):
                        desired=np.linalg.norm(target)/(10**(snr/20)) if snr is not None else 0.
                        amplitude=desired/max(np.linalg.norm(field),1e-20)
                        artifact=field*amplitude
                        suffix="clean" if snr is None else f"snr{int(snr):+d}"
                        name=f"{source['participant']}-{recipient_index:03d}-{condition}-{suffix}.npz"
                        np.savez_compressed(arrays/name,eeg=target+artifact,paired_reference=target,
                            true_artifact=artifact,references=references,**metadata)
                        examples.append({"example_id":name[:-4],"array_path":"arrays/"+name,"dataset":"controlled_LEMON_OSF",
                            "recipient":record["participant"],"donor":donor["record"]["participant"],"partition":partition,
                            "condition":condition,"input_snr_db":snr,"target_kind":"controlled_recipient_reference",
                            "calibration":"calibration/"+calid+"/calibration.npz",
                            "recipe":{"field":"donor-calibration ridge-projected lagged HEOG/VEOG coherent spatial field",
                                "lags":donor["mapping"]["lags"],"donor_calibration_digest":donor["mapping"]["donor_calibration_digest"],
                                "donor_scoring_trial":selected["trial"],"recipient_start":recipient["start"],
                                "amplitude":float(amplitude),"exact_channel_intersection":names,"lag_edges":"zero",
                                "caveat":"reference-generated mixture family; native source target cleanliness is unknown"}})
            print("LEMON_DONE",source["participant"],len(examples),flush=True)
        except Exception as error:
            failures.append({"source":source["participant"],"error":f"{type(error).__name__}: {error}"})
            print("LEMON_FAILED",failures[-1],flush=True)
        finally:
            raw_folder=(Path(temporary)/source["participant"]).resolve()
            if not raw_folder.is_relative_to(Path(temporary).resolve()):
                raise ValueError("Raw cleanup path escapes task temporary directory")
            if raw_folder.exists(): shutil.rmtree(raw_folder)
        write_jsonl(output/"recipient_manifest.jsonl",recipients)
        write_jsonl(output/"corpus_manifest.jsonl",examples)
        atomic_json(output/"source_failures.json",failures)
    controlled=[r for r in examples if r["target_kind"]=="controlled_recipient_reference"]
    summary={"examples":len(examples),"controlled_examples":len(controlled),"recipient_sources":len(set(r["recipient"] for r in controlled)),
        "donor_sources":len(set(r["donor"] for r in controlled)),"download_failures":failures,"profile":profile,
        "split_hash":split["hash"],"parent_audit_hash":sha256_file(parent/"audit_summary.json"),
        "reserved_confirmation_opened":False,"native_clean_target":"unknown; controlled reconstruction is defined relative to retained recipient signal"}
    atomic_json(output/"corpus_summary.json",summary)
    if not controlled: raise RuntimeError("No eligible fresh controlled corpus; cannot proceed to SNR experiments")
