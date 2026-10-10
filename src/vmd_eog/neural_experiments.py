"""Kaggle-only paired optimization, source-disjoint validation and evidence."""
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import random
import time
import numpy as np
import pandas as pd
import torch
from .artifacts import verify_parent
from .autovmd import balanced_rows,source_identities
from .contracts import canonical_hash
from .experiments import corpus_parent
from .io import atomic_json,read_jsonl,sha256_file
from .metrics import paired
from .neural import DeploymentStudent,RegionalBandRouter,input_scale,build_model
from .neural_losses import paired_objective
from .paper_metrics import paired_channels,resting_spectrum


def inner_partition(rows,outer_fold,inner_fold=0):
    """Freeze inner groups from global source buckets; never split windows."""
    remaining = sorted({row["partition"]["fold"] for row in rows}-{outer_fold})
    if len(remaining) != 4 or inner_fold not in range(3):
        raise ValueError("Need five outer buckets and a declared inner fold")
    # Four source buckets are allocated 2/1/1 to three inner folds.
    validation_buckets = {bucket for index,bucket in enumerate(remaining) if index%3 == inner_fold}
    training = [row for row in rows if row["partition"]["fold"] in set(remaining)-validation_buckets]
    validation = [row for row in rows if row["partition"]["fold"] in validation_buckets]
    outer = [row for row in rows if row["partition"]["fold"] == outer_fold]
    if not training or not validation or not outer:
        raise ValueError("Empty source-isolated partition")
    groups = [source_identities(partition) for partition in (training,validation,outer)]
    if any(groups[a]&groups[b] for a,b in ((0,1),(0,2),(1,2))):
        raise ValueError("Recipient or donor leakage across neural partitions")
    return training,validation,outer


class PairedWindows(torch.utils.data.Dataset):
    def __init__(self,parent,rows,config,experiment,input_root):
        self.parent,self.rows,self.config,self.experiment = parent,rows,config,experiment
        self.cache = {}
        self.source_hashes = {}
        for index in Path(input_root).rglob("mode_cache_index.jsonl"):
            verify_parent(index.parent,(index.name,))
            for entry in read_jsonl(index):
                key = (entry["example_id"],entry["recipe"]["K"],entry["recipe"]["alpha"],entry["scope"])
                if key in self.cache and self.cache[key][1]["sha256"] != entry["sha256"]:
                    raise ValueError("Ambiguous incompatible mode cache")
                self.cache[key] = (index.parent/entry["path"],entry)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self,index):
        row = self.rows[index]
        source = self.parent/row["array_path"]
        with np.load(source,allow_pickle=False) as arrays:
            data = {name:torch.from_numpy(np.asarray(arrays[name]).copy()) for name in
                    ("eeg","paired_reference","mask","regions","hemispheres","references")}
        data["eeg"],data["paired_reference"],data["references"] = (data[name].float() for name in ("eeg","paired_reference","references"))
        data["mask"] = data["mask"].bool()
        arm = self.experiment["arm"]
        if arm in ("regional","no_context","all_vmd"):
            scope = "all" if arm == "all_vmd" else "frontal"
            key = (row["example_id"],self.experiment["K"],self.experiment["alpha"],scope)
            if key not in self.cache:
                raise ValueError("Requested neural recipe has no verified cache for " + row["example_id"])
            path,entry = self.cache[key]
            digest = self.source_hashes.setdefault(str(source),sha256_file(source))
            if digest != entry["source_sha256"] or sha256_file(path) != entry["sha256"]:
                raise ValueError("Cached components do not match source/recipe identity")
            with np.load(path,allow_pickle=False) as bundle:
                data["bundle"] = {name:torch.from_numpy(bundle[name].copy()) for name in ("components","descriptors","valid")}
        return data


def move_window(data,device):
    tensors = {key:value.unsqueeze(0).to(device) for key,value in data.items() if key != "bundle"}
    bundle = None
    if "bundle" in data:
        bundle = {key:value.unsqueeze(0).to(device) for key,value in data["bundle"].items()}
        bundle["scale"] = input_scale(tensors["eeg"]).unsqueeze(-2)
    return tensors,bundle


def forward(model,data,bundle,mask=None):
    metadata = {key:data[key] for key in ("regions","hemispheres")}
    mask = data["mask"] if mask is None else mask
    if isinstance(model,DeploymentStudent):
        return model(data["eeg"],mask,metadata)
    return model(data["eeg"],mask,metadata,bundle)


def validate(model,dataset,device,config,save_directory=None):
    model.eval()
    records = []
    if save_directory:
        save_directory.mkdir(parents=True,exist_ok=True)
    failures = []
    with torch.no_grad():
        for index,row in enumerate(dataset.rows):
            data,bundle = move_window(dataset[index],device)
            prediction = forward(model,data,bundle)["cleaned"][0].cpu().numpy()
            original = data["eeg"][0].cpu().numpy()
            target = data["paired_reference"][0].cpu().numpy()
            if not np.isfinite(prediction).all():
                failures.append({"example_id":row["example_id"],"failure":"nonfinite neural estimate","fallback":"input"})
                prediction = original.copy()
            valid = data["mask"][0].cpu().numpy()
            channel_metrics = paired_channels(prediction[valid],target[valid],original[valid])
            diagnostic = paired(prediction[valid],target[valid],config["fs"])
            # Spectral RRMSE is a labelled Welch-PSD adaptation, not an
            # undocumented claim of matching an unavailable author function.
            spectrum = resting_spectrum(prediction[valid],target[valid],config["fs"],normalization_band=(1,40))
            spectral_error = np.sqrt(((spectrum["psd_after"]-spectrum["psd_before"])**2).sum(-1)/np.maximum((spectrum["psd_before"]**2).sum(-1),1e-30))
            for channel in range(int(valid.sum())):
                records.append({"example_id":row["example_id"],"recipient":row["recipient"],"donor":row["donor"],
                    "condition":row["condition"],"input_level":row.get("input_snr_db"),"channel":channel,
                    **{key:float(value[channel]) for key,value in channel_metrics.items()},
                    "rrmse_spectral_welch_adaptation":float(spectral_error[channel]),
                    "clean_relative_error":diagnostic["relative_error"] if row["condition"] == "clean" else np.nan,
                    "clean_alpha_db":diagnostic["alpha_db"] if row["condition"] == "clean" else np.nan,
                    "clean_beta_db":diagnostic["beta_db"] if row["condition"] == "clean" else np.nan,
                    "clean_covariance":diagnostic["covariance"] if row["condition"] == "clean" else np.nan})
            if save_directory:
                np.savez_compressed(save_directory/(row["example_id"]+".npz"),cleaned=prediction,
                    input=original,paired_reference=target,mask=valid,regions=data["regions"][0].cpu().numpy())
    frame = pd.DataFrame(records)
    dirty = frame[frame.condition != "clean"]
    source_means = dirty.groupby(["recipient","condition","input_level"]).snr_energy_db.mean().groupby("recipient").mean()
    if not np.isfinite(source_means).all():
        raise ValueError("Undefined dirty-window SNR needs explicit diagnosis")
    clean = frame[frame.condition == "clean"].groupby("recipient")[["clean_relative_error","clean_alpha_db","clean_beta_db","clean_covariance"]].mean()
    if len(clean) == 0:
        raise ValueError("Validation has no clean controls")
    limits = config["preservation"]
    gates = {"clean_change":bool(clean.clean_relative_error.max() <= limits["clean_change"]),
        "alpha":bool(clean.clean_alpha_db.max() <= limits["alpha_db"]),
        "beta":bool(clean.clean_beta_db.max() <= limits["beta_db"]),
        "covariance":bool(clean.clean_covariance.max() <= limits["covariance"])}
    summary = {"snr_db":float(source_means.mean()),"clean_gates":gates,"preservation_passed":all(gates.values()),
        "paired_correlation_comparator_gate":"pending matched baseline review", "scientific_qualified":False,
        "worst_source_clean_relative_error":float(clean.clean_relative_error.max()),"failures":failures}
    return frame,summary


def train_experiment(input_root,output,config,profile,experiment):
    parent,rows = corpus_parent(input_root)
    experiment = {"arm":"fixed","outer_fold":0,"inner_fold":0,"seed":42,"loss_profile":"mse",
                  "learning_rate":.001,"epochs":5 if profile == "pilot" else 80,"evaluate_outer":False,**experiment}
    if experiment["seed"] not in config["neural"]["seeds"]:
        raise ValueError("Seed outside declared campaign")
    if not 1 <= experiment["epochs"] <= config["neural"]["max_epochs"]:
        raise ValueError("Epoch budget outside declared campaign")
    controlled = [row for row in rows if row["target_kind"] == "controlled_recipient_reference"]
    if profile == "pilot" or experiment.get("selection_only"):
        controlled = balanced_rows(controlled)
    training,validation,outer = inner_partition(controlled,experiment["outer_fold"],experiment["inner_fold"])
    if experiment.get("refit_outer_training"):
        raise ValueError("Outer refit must use an independently selected epoch budget and disable inner early stopping; not implemented")
    atomic_json(output/"neural_partitions.json", {name:[{"example_id":row["example_id"],"recipient":row["recipient"],"donor":row["donor"]} for row in partition]
        for name,partition in (("training",training),("validation",validation),("outer",outer))})
    seed = experiment["seed"]
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True,warn_only=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Training notebook has no CUDA GPU; publish a GPU version")
    fixture_files = list(Path(input_root).rglob("neural_fixture_summary.json"))
    if len(fixture_files) != 1:
        raise RuntimeError("Attach one completed neural/CUDA contract fixture before training")
    fixture = fixture_files[0]
    verify_parent(fixture.parent,(fixture.name,"environment.json"))
    fixture_summary = json.loads(fixture.read_text())
    environment = json.loads((fixture.parent/"environment.json").read_text())
    if fixture_summary.get("passed") is not True or fixture_summary.get("device") != "cuda":
        raise RuntimeError("GPU numerical fixture has not passed")
    if environment["packages"]["torch"] != torch.__version__:
        raise RuntimeError("Torch version differs from frozen GPU fixture environment")
    model = build_model(experiment["arm"],config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(),lr=experiment["learning_rate"],weight_decay=config["neural"]["weight_decay"])
    scaler = torch.amp.GradScaler("cuda")
    train_data = PairedWindows(parent,training,config,experiment,input_root)
    val_data = PairedWindows(parent,validation,config,experiment,input_root)
    counts = {}
    for row in training:
        key = (row["recipient"],row["condition"],row.get("input_snr_db"))
        counts[key] = counts.get(key,0)+1
    weights = [(.25 if row["condition"] == "clean" else .75/9)/counts[(row["recipient"],row["condition"],row.get("input_snr_db"))] for row in training]
    sampler = torch.utils.data.WeightedRandomSampler(weights,len(training),replacement=True,generator=torch.Generator().manual_seed(seed))
    accumulation = config["neural"]["effective_batch"]
    best = None
    stale,history = 0,[]
    started = time.perf_counter()
    for epoch in range(experiment["epochs"]):
        model.train(); optimizer.zero_grad(set_to_none=True)
        total_loss = 0.
        sampled = list(sampler)
        for step,index in enumerate(sampled):
            data,bundle = move_window(train_data[index],device)
            mask = data["mask"].clone()
            mask &= torch.rand(mask.shape,device=device) >= config["neural"]["channel_drop_probability"]
            if random.random() < config["neural"]["frontal_drop_probability"]:
                mask &= data["regions"] != 0
            if not mask.any():
                mask = data["mask"].clone()
            group_size = min(accumulation,len(sampled)-(step//accumulation)*accumulation)
            with torch.autocast(device_type="cuda",dtype=torch.float16):
                prediction = forward(model,data,bundle,mask)["cleaned"]
            # FFT and covariance objectives stay in float32 outside autocast.
            loss,_ = paired_objective(prediction.float(),data["paired_reference"],data["eeg"],mask,
                torch.tensor([training[index]["condition"] == "clean"],device=device),config,experiment["loss_profile"],data["references"])
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite training objective; retain failure evidence")
            scaler.scale(loss/group_size).backward()
            total_loss += float(loss.detach())
            if (step+1)%accumulation == 0 or step+1 == len(sampled):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(),config["neural"]["gradient_clip"])
                scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
        frame,summary = validate(model,val_data,device,config)
        score = (summary["preservation_passed"],summary["snr_db"])
        improved = best is None or score > best["score"]
        if improved:
            best = {"score":score,"epoch":epoch+1,"state":{key:value.detach().cpu().clone() for key,value in model.state_dict().items()},"summary":summary}
            stale = 0
            torch.save({"state_dict":best["state"],"experiment":experiment,"configuration_hash":canonical_hash(config),
                "corpus_manifest_sha256":sha256_file(parent/"corpus_manifest.jsonl"),"epoch":epoch+1},output/"best_model.pt")
        else:
            stale += 1
        history.append({"epoch":epoch+1,"train_loss":total_loss/len(sampled),"snr_db":summary["snr_db"],
                        "clean_preservation_passed":summary["preservation_passed"],"runtime_s":time.perf_counter()-started})
        pd.DataFrame(history).to_csv(output/"training_history.csv",index=False)
        torch.save({"state_dict":model.state_dict(),"optimizer":optimizer.state_dict(),"scaler":scaler.state_dict(),
            "epoch":epoch+1,"torch_rng":torch.get_rng_state(),"cuda_rng":torch.cuda.get_rng_state_all(),
            "sampler_rng":sampler.generator.get_state(),"experiment":experiment},output/"last_checkpoint.pt")
        print("NEURAL_EPOCH",epoch+1,history[-1],flush=True)
        if stale >= config["neural"]["patience"]:
            break
    model.load_state_dict(best["state"])
    frame,summary = validate(model,val_data,device,config,output/"validation_predictions")
    frame.to_csv(output/"neural_validation_channels.csv.gz",index=False)
    if experiment["evaluate_outer"]:
        outer_data = PairedWindows(parent,outer,config,experiment,input_root)
        outer_frame,outer_summary = validate(model,outer_data,device,config,output/"outer_predictions")
        outer_frame.to_csv(output/"neural_outer_channels.csv.gz",index=False)
        atomic_json(output/"neural_outer_summary.json",outer_summary)
    summary.update({"campaign_id":config["campaign_id"],"experiment":experiment,"best_epoch":best["epoch"],
        "epochs_completed":len(history),"runtime_s":time.perf_counter()-started,"parameter_count":sum(p.numel() for p in model.parameters()),
        "checkpoint_sha256":sha256_file(output/"best_model.pt"),"reserved_confirmation_opened":False,
        "paired_reference":"retained low-ocular recipient EEG; no artifact-free claim", "screening_only":profile == "pilot"})
    atomic_json(output/"training_summary.json",summary)
    import matplotlib.pyplot as plt
    figure,axes = plt.subplots(1,2,figsize=(11,4))
    axes[0].plot([row["epoch"] for row in history],[row["train_loss"] for row in history])
    axes[0].set(xlabel="Epoch",ylabel="Paired objective",title="Training")
    axes[1].plot([row["epoch"] for row in history],[row["snr_db"] for row in history])
    axes[1].set(xlabel="Epoch",ylabel="Source-balanced output SNR (dB)",title="Inner development validation")
    figure.tight_layout(); figure.savefig(output/"learning_curves.png",dpi=150); plt.close(figure)


def run_fixture(output,config):
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    eeg = torch.randn(1,4,512,device=device)
    target = .7*eeg
    mask = torch.ones(1,4,dtype=torch.bool,device=device)
    metadata = {"regions":torch.tensor([[0,0,1,2]],device=device),"hemispheres":torch.tensor([[0,1,2,3]],device=device)}
    outcomes = []
    from .autovmd import AutoVMD,FrozenVMDRecipe
    from .neural import input_scale
    transformed = AutoVMD(config).transform(eeg[0].cpu().numpy(),FrozenVMDRecipe(5,1000),scope="all")
    mode_bundle = {key:torch.from_numpy(transformed[key]).unsqueeze(0).to(device) for key in ("components","descriptors","valid")}
    mode_bundle["scale"] = input_scale(eeg).unsqueeze(-2)
    for arm in ("fixed","raw","student","regional","no_context","all_vmd"):
        model = build_model(arm,config).to(device)
        optimizer = torch.optim.AdamW(model.parameters(),lr=.001)
        model.eval()
        data = {"eeg":eeg,"mask":mask,**metadata}
        torch.testing.assert_close(forward(model,data,mode_bundle)["cleaned"],eeg,rtol=0,atol=0)
        model.train()
        for step in range(3):
            optimizer.zero_grad()
            prediction = forward(model,data,mode_bundle)["cleaned"]
            loss,_ = paired_objective(prediction,target,eeg,mask,torch.zeros(1,dtype=torch.bool,device=device),config)
            loss.backward(); optimizer.step()
        if not any(parameter.grad is not None and parameter.grad.abs().sum() > 0 for name,parameter in model.named_parameters() if 'heads' not in name):
            raise RuntimeError("Correction heads did not propagate encoder gradients")
        model.eval()
        checkpoint = output/(arm+"_fixture.pt")
        torch.save(model.state_dict(),checkpoint)
        reloaded = build_model(arm,config).eval()
        reloaded.load_state_dict(torch.load(checkpoint,weights_only=True,map_location="cpu"))
        cpu_metadata = {key:value.cpu() for key,value in metadata.items()}
        cpu_data = {"eeg":eeg.cpu(),"mask":mask.cpu(),**cpu_metadata}
        cpu_bundle = {key:value.cpu() for key,value in mode_bundle.items()}
        torch.testing.assert_close(forward(model,data,mode_bundle)["cleaned"].cpu(),forward(reloaded,cpu_data,cpu_bundle)["cleaned"],rtol=1e-4,atol=1e-5)
        outcomes.append({"arm":arm,"passed":True,"parameters":sum(p.numel() for p in model.parameters()),"final_objective":float(loss.detach())})
    atomic_json(output/"neural_fixture_summary.json",{"passed":True,"synthetic_software_only":True,"device":device.type,"torch_version":torch.__version__,"models":outcomes})


def select_search(input_root,output,config,experiment):
    """Aggregate only a complete, source-compatible declared screening rung."""
    parents = list(Path(input_root).rglob("training_summary.json"))
    required = experiment.get("required_candidates",40)
    records = []
    for path in parents:
        verify_parent(path.parent,(path.name,"neural_partitions.json"))
        summary = json.loads(path.read_text())
        spec = summary["experiment"]
        if summary["campaign_id"] != config["campaign_id"] or spec["outer_fold"] != experiment["outer_fold"]:
            raise ValueError("Search parent belongs to another fold/campaign")
        if summary["reserved_confirmation_opened"] is not False or spec.get("evaluate_outer"):
            raise ValueError("Outer/confirmation evidence cannot select a candidate")
        records.append({"K":spec["K"],"alpha":spec["alpha"],"snr_db":summary["snr_db"],
            "preservation_passed":summary["preservation_passed"],"runtime_s":summary["runtime_s"],"epochs":summary["epochs_completed"]})
    if len(records) != required or len({(row["K"],row["alpha"]) for row in records}) != required:
        raise ValueError("Search rung is incomplete or duplicated")
    frame = pd.DataFrame(records).sort_values(["preservation_passed","snr_db"],ascending=[False,False])
    frame.to_csv(output/"autovmd_selection.csv",index=False)
    atomic_json(output/"autovmd_selection_summary.json",{"campaign_id":config["campaign_id"],
        "candidates":len(records),"promoted":frame.head(experiment.get("keep",8)).to_dict("records"),
        "scientific_qualified":False,"reserved_confirmation_opened":False})
