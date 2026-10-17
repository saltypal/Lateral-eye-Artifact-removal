"""Verified source adapters and frozen participant partitions, executed on Kaggle."""
import hashlib
import json
from pathlib import Path
import re
import urllib.request
import numpy as np
from .contracts import canonical_hash
from .io import atomic_json, sha256_file, write_jsonl
from .osf_reader import read_osf
from .signal import preprocess, calibration_trials

FRONTAL = {"FP1", "FP2", "FPZ", "AFZ", "FZ"} | {f"{p}{n}" for p in ("AF", "F") for n in range(1,11)}
POSTERIOR = {"PZ", "POZ", "OZ", "IZ", "O1", "O2"} | {f"{p}{n}" for p in ("P", "PO") for n in range(1,11)}
_preprocess = preprocess


def _region(name):
    name = name.strip().upper()
    if name in FRONTAL: return 0
    if name in POSTERIOR: return 1
    if re.fullmatch(r"(?:FC|FT|C|CP|TP|T)(?:\d+|Z)", name): return 2
    return 3


def _hemisphere(name):
    name = name.strip().upper()
    if _region(name) == 3: return 3
    if name.endswith("Z"): return 2
    return 0 if int(re.search(r"\d+$",name).group()) % 2 else 1


def eog_axes(item):
    """Only named, documented derivatives; EOG1/EOG2 do not establish orientation."""
    names = [name.upper().replace("-", "").replace("_", "") for name in item["names"]]
    indices = []
    for aliases in ({"HEOG", "EOGH"}, {"VEOG", "EOGV"}):
        matches = [i for i,name in enumerate(names) if name in aliases and i in item["eog_indices"]]
        if len(matches) != 1:
            raise ValueError("HEOG/VEOG orientation absent or ambiguous")
        indices.append(matches[0])
    return indices


def channel_metadata(names, locations=None):
    coords, valid = [], []
    for i in range(len(names)):
        point = []
        for axis in ("X", "Y", "Z"):
            value = np.asarray((locations or [{}]*len(names))[i].get(axis,[])).reshape(-1)
            point.append(float(value[0]) if value.size == 1 and np.isfinite(value[0]) else 0.)
        ok = bool(np.linalg.norm(point) > 0)
        coords.append(point); valid.append(ok)
    return {"channel_names":np.asarray(names,dtype="U64"),
        "regions":np.asarray([_region(n) for n in names],dtype=np.int64),
        "hemispheres":np.asarray([_hemisphere(n) for n in names],dtype=np.int64),
        "coordinates":np.asarray(coords,dtype=np.float32),"coordinate_mask":np.asarray(valid),
        "mask":np.ones(len(names),dtype=bool)}


def freeze_groups(ids, seed=42, fraction=.2, folds=5):
    """Freeze identities before extracting any windows; deterministic hash shuffle."""
    ids = sorted(set(ids),key=lambda x:hashlib.sha256(f"{seed}:{x}".encode()).hexdigest())
    if len(ids) < 5: raise ValueError("At least five independent identities required")
    reserved = max(1,int(np.ceil(len(ids)*fraction)))
    result = {}
    for i,identity in enumerate(ids):
        role = "confirmation" if i < reserved else "development"
        result[identity] = {"role":role,"fold":None if i < reserved else (i-reserved)%folds}
    return result


def lemon_sources(count=40):
    # Publisher page embeds official archive URLs in checkbox values. HTTP is
    # metadata only: its historical underscore hostname has a mismatched TLS cert.
    page = "http://fcon_1000.projects.nitrc.org/indi/retro/MPI_LEMON/downloads/download_EEG.html"
    html = urllib.request.urlopen(page,timeout=60).read().decode()
    urls = sorted(set(re.findall(r'https://fcp-indi\.s3\.amazonaws\.com/[^"<>\s]+/EEG_Raw_BIDS_ID/sub-\d+\.tar\.gz',html)))
    if len(urls) < count: raise ValueError("Publisher raw source list is incomplete")
    return [{"participant":Path(url).name.split(".")[0],"url":url,"dataset":"raw_LEMON",
             "historical_exposure":"not previously used by this project","metadata_page":page} for url in urls[:count]]


def audit_sources(root, output, config, profile):
    """Inventory all original sessions; waveform caches contain development only."""
    output = Path(output); caches=output/"source_cache"; caches.mkdir()
    readme = root/"Dataset1_OSF/readme.txt"
    description = readme.read_text(encoding="utf-8-sig")
    if "participant ids are unique across all studies" not in description:
        raise ValueError("Global participant identity evidence missing")
    files = sorted((root/"Dataset1_OSF").rglob("*_prep.set"))
    identities = []
    for path in files:
        match = re.fullmatch(r"(study\d+)_(p\d+)_prep",path.stem)
        if match is None: raise ValueError("Unexpected original source naming")
        identities.append("osf:"+match[2])
    partitions = {"osf":freeze_groups(identities,config["seed"],config["confirmation_fraction"],config["outer_folds"])}
    lemon = lemon_sources(40)
    partitions["lemon"] = freeze_groups(["lemon:"+r["participant"] for r in lemon],config["seed"],config["confirmation_fraction"],config["outer_folds"])
    atomic_json(output/"split_manifest.json",{"policy":"freeze recipients and donors independently before windows",
        "partitions":partitions,"hash":canonical_hash(partitions),"seed":config["seed"],
        "limits":"OSF historical exposure remains development exposure; Klados participant identities unknown."})
    atomic_json(output/"external_sources.json",{"lemon":lemon,"magdeburg":{
        "doi":"10.24352/UB.OVGU-2020-155","participant_count":18,"fs":250,"eeg_channels":30,
        "role":"independent waveform evaluation after model freeze","waveforms_loaded":False,
        "reuse_license":"must verify source license before redistributing"}})
    rows=[]; selected={}; failures=[]
    for path,identity in zip(files,identities):
        row={"record_id":path.stem,"participant":identity,"dataset":"osf","path":str(path.relative_to(root)),
             "source_sha256":sha256_file(path),"partition":partitions["osf"][identity],
             "historical_exposure":"dataset used in previous development","paired_clean":False}
        try:
            item=read_osf(path); refs=eog_axes(item)
            names=[item["names"][i] for i in item["eeg_indices"]]
            row.update(native_fs=item["fs"],shape=list(item["data"].shape),channel_names=names,
                reference=item["reference"],trial_labels=item["trial_labels"],eog_names=[item["names"][i] for i in refs],
                previous_filtering="source preprocessed; see publisher protocol; original reference retained",
                region_counts={str(k):int(sum(_region(n)==k for n in names)) for k in range(4)})
            if row["partition"]["role"]=="confirmation":
                row["waveform_role"]="reserved; no cache generated"
            else:
                take = profile=="full" or selected.get(item["study"],0)<config["pilot"]["osf_per_study"]
                if take:
                    selected[item["study"]]=selected.get(item["study"],0)+1
                    cache_osf(item,refs,caches,row,config,profile)
            row["eligible"]=True
        except Exception as error:
            row.update(eligible=False,error=f"{type(error).__name__}: {error}")
            failures.append(row["record_id"])
        rows.append(row); print("AUDIT",row["record_id"],row["eligible"],flush=True)
        write_jsonl(output/"recording_manifest.jsonl",rows)
    klados=root/"klados"
    shapes={}
    for path in sorted(klados.glob("klados_*.npy")):
        data=np.load(path,mmap_mode="r",allow_pickle=False)
        shapes[path.name]={"shape":list(data.shape),"dtype":str(data.dtype),"sha256":sha256_file(path)}
    atomic_json(output/"klados_audit.json",{"arrays":shapes,"metadata":"channel order, units and participants unknown",
        "role":"historically exposed paired engineering comparison; no anatomical assignments",
        "publisher_alignment":"legacy export audit identified 53 records cropped to 5401 samples; revalidation required"})
    cache_klados(klados,caches,config,profile)
    summary={"sessions":len(rows),"participants":len(partitions["osf"]),"eligible_sessions":sum(r["eligible"] for r in rows),
        "source_failures":failures,"cached_development_sessions":sum(selected.values()),"profile":profile,
        "split_hash":canonical_hash(partitions),"confirmation_waveform_cache_generated":False}
    atomic_json(output/"audit_summary.json",summary)
    return summary


def cache_osf(item,refs,caches,row,config,profile):
    indices=item["eeg_indices"]+refs
    trials=[trial[indices] for trial in item["data"]]
    block=item["annotations"].get("BLOCK")
    if block is not None:
        labels=np.round(item["data"][:,block,0]).astype(int)
    else: labels=np.zeros(len(trials),int)
    if np.any(labels==1) and np.any(labels==2):
        cal_ids=np.flatnonzero(labels==1).tolist(); score_ids=np.flatnonzero(labels==2).tolist()
        cal=[trials[i] for i in cal_ids]; score=[trials[i] for i in score_ids]
        policy="declared block 1 calibration, block 2 scoring; independent trial filtering"
    else:
        cal,score,error=calibration_trials(trials,item["fs"])
        if error: raise ValueError(error)
        cal_ids=list(range(len(cal))); score_ids=list(range(len(cal),len(trials)))
        policy="chronological prefix with whole-trial boundaries"
    names=[item["names"][i] for i in item["eeg_indices"]]
    metadata=channel_metadata(names,[item["locations"][i] for i in item["eeg_indices"]])
    cal_arrays=[preprocess(x,item["fs"]) for x in cal]
    # Covariance/ICA concatenate already independently filtered calibration trials.
    # Lag estimators receive a validity mask so fitting never crosses joins.
    joined=np.concatenate(cal_arrays,axis=-1)
    boundaries=np.cumsum([x.shape[-1] for x in cal_arrays])[:-1]
    trial_types=[(item["trial_labels"] or [0]*len(trials))[i] for i in cal_ids]
    cal_labels=np.concatenate([np.full(x.shape[-1],t,int) for x,t in zip(cal_arrays,trial_types)])
    directory=caches/row["record_id"]; directory.mkdir()
    np.savez_compressed(directory/"calibration.npz",eeg=joined[:-2],references=joined[-2:],
        boundaries=boundaries,trial_types=cal_labels,**metadata)
    windows=[]; counts={}; limit=config["pilot"]["windows_per_condition"] if profile=="pilot" else 8
    for trial,trial_id in zip(score,score_ids):
        kind=(item["trial_labels"] or [0]*len(trials))[trial_id]
        values=preprocess(trial,item["fs"])
        for start in range(0,values.shape[-1]-config["window"]+1,config["hop"]):
            if counts.get(kind,0)>=limit: break
            name=f"trial-{trial_id:04d}-start-{start:05d}.npz"
            np.savez_compressed(directory/name,eeg=values[:-2,start:start+config["window"]],
                references=values[-2:,start:start+config["window"]],**metadata)
            windows.append({"array":name,"trial":trial_id,"start":start,"condition":{1:"rest",2:"lateral",3:"vertical",4:"blink"}.get(kind,"unknown")})
            counts[kind]=counts.get(kind,0)+1
    row["calibration_policy"]=policy
    row["cache"]="source_cache/"+row["record_id"]
    atomic_json(directory/"record.json",{**row,"windows":windows,"calibration_trial_ids":cal_ids,"scoring_trial_ids":score_ids})


def cache_klados(root,caches,config,profile):
    arrays={key:np.load(root/f"klados_{key}.npy",mmap_mode="r",allow_pickle=False) for key in ("pure_eeg","contaminated_eeg","heog","veog")}
    clean=arrays["pure_eeg"]; dirty=arrays["contaminated_eeg"]
    if clean.shape!=dirty.shape or clean.ndim!=3: raise ValueError("Klados paired layout invalid")
    records=[]; limit=4 if profile=="pilot" else len(clean)
    for i in range(min(len(clean),limit)):
        refs=np.stack([arrays[k][i].reshape(-1,clean.shape[-1])[0] for k in ("heog","veog")])
        combined=np.concatenate([clean[i],dirty[i],refs],axis=0)
        cal,score,error=calibration_trials([combined],200)
        if error: raise ValueError("Klados: "+error)
        calibration=preprocess(cal[0],200); scoring=preprocess(score[0],200)
        count=clean.shape[1]; meta=channel_metadata([f"UNKNOWN_{c}" for c in range(count)])
        name=f"klados-{i:03d}"; directory=caches/name; directory.mkdir()
        np.savez_compressed(directory/"calibration.npz",eeg=calibration[count:2*count],references=calibration[-2:],
            boundaries=np.asarray([],int),trial_types=np.zeros(calibration.shape[-1],int),**meta)
        windows=[]
        for start in list(range(0,scoring.shape[-1]-config["window"]+1,config["hop"]))[:(1 if profile=="pilot" else 4)]:
            path=f"paired-{start:05d}.npz"; segment=scoring[:,start:start+config["window"]]
            np.savez_compressed(directory/path,eeg=segment[count:2*count],paired_reference=segment[:count],references=segment[-2:],**meta)
            windows.append({"array":path,"condition":"mixed","start":start})
        record={"record_id":name,"dataset":"klados","participant":None,"partition":{"role":"development","fold":i%config["outer_folds"]},
            "target_kind":"legacy_paired","participant_verified":False,"windows":windows,"source_row":i,
            "clean_hash":hashlib.sha256(np.ascontiguousarray(clean[i]).tobytes()).hexdigest()}
        atomic_json(directory/"record.json",record); records.append(record)
    write_jsonl(caches/"klados_manifest.jsonl",records)
