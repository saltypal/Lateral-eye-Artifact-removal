"""Frozen, source-aware AutoVMD and immutable decomposition caches."""
from dataclasses import dataclass,asdict
from itertools import product
import time
import numpy as np
from .contracts import canonical_hash
from .experiments import corpus_parent
from .frontal import decompose
from .io import atomic_json,sha256_file,write_jsonl


@dataclass(frozen=True)
class FrozenVMDRecipe:
    K:int
    alpha:float
    fs:int=200
    tolerance:float=1e-6
    max_iterations:int=2000
    initialization:str="uniform"
    normalization:str="input_RMS"

    def identity(self):
        return canonical_hash(asdict(self))


def source_identities(rows):
    return {str(row[kind]) for row in rows for kind in ("recipient","donor") if row.get(kind) is not None}


def balanced_rows(rows):
    """One stable example per recipient/condition/input-level; no target ranking."""
    selected = {}
    for row in sorted(rows,key=lambda item:item["example_id"]):
        if row["target_kind"] != "controlled_recipient_reference":
            continue
        key = (row["recipient"],row["condition"],row.get("input_snr_db"))
        selected.setdefault(key,row)
    return list(selected.values())


class AutoVMD:
    def __init__(self,config):
        self.config = config

    def transform(self,eeg,recipe,regions=None,scope="frontal"):
        """Only input EEG is consumed. Return explicit raw fallback on failure."""
        values = np.asarray(eeg,dtype=np.float32)
        if values.ndim != 2 or not np.isfinite(values).all():
            raise ValueError("Need finite [channels,samples] EEG")
        if recipe.fs != self.config["fs"]:
            raise ValueError("Recipe sampling rate differs")
        channels,length = values.shape
        number = max(6,recipe.K+1)
        components = np.zeros((channels,number,length),dtype=np.float32)
        descriptors = np.zeros((channels,number,5),dtype=np.float32)
        valid = np.zeros((channels,number),dtype=bool)
        selected = np.ones(channels,dtype=bool) if scope == "all" else np.asarray(regions) == 0
        if selected.shape != (channels,):
            raise ValueError("Verified region vector is required for frontal cache")
        diagnostics = []
        solver = dict(self.config)
        solver["classical_grid"] = {**solver["classical_grid"],"tolerance":recipe.tolerance,"max_iterations":recipe.max_iterations}
        for channel in np.flatnonzero(selected):
            try:
                modes,residual,info = decompose(values[channel],recipe.K,recipe.alpha,solver)
                pieces = np.concatenate((modes,residual[None]),axis=0)
                if not np.isfinite(pieces).all():
                    raise RuntimeError("Nonfinite mode vectors")
                components[channel,:recipe.K+1] = pieces
                valid[channel,:recipe.K+1] = True
                power = np.abs(np.fft.rfft(pieces,axis=-1))**2
                energy = power.sum(-1)
                frequencies = np.fft.rfftfreq(length,1/recipe.fs)
                centers = (power*frequencies).sum(-1)/np.maximum(energy,1e-20)
                centers[:recipe.K] = info["centers_hz"]
                widths = np.sqrt((power*(frequencies-centers[:,None])**2).sum(-1)/np.maximum(energy,1e-20))
                descriptor = descriptors[channel,:recipe.K+1]
                descriptor[:,0] = centers/recipe.fs
                descriptor[:,1] = widths/recipe.fs
                descriptor[:,2] = energy/max(energy.sum(),1e-20)
                descriptor[:,3] = float(info["hit_iteration_limit"])
                descriptor[-1,4] = 1
                diagnostics.append({"channel":int(channel),"failure":None,**info})
            except (ValueError,RuntimeError,FloatingPointError) as error:
                components[channel,0] = values[channel]
                valid[channel,0] = True
                descriptors[channel,0,2:] = (1,1,1)
                diagnostics.append({"channel":int(channel),"failure":str(error),"fallback":"raw"})
        return {"components":components,"descriptors":descriptors,"valid":valid,"diagnostics":diagnostics}

    def select(self,training_ids,validation_ids,search_config):
        """Successive screening callback runs neural fits inside Kaggle.

        The callback receives a frozen recipe, budget and disjoint source IDs.
        Nested fold orchestration must supply training/validation rows here;
        there is no default random window split.
        """
        rows = search_config["rows"]
        lookup = {row["example_id"]:row for row in rows}
        training = [lookup[identity] for identity in training_ids]
        validation = [lookup[identity] for identity in validation_ids]
        if set(training_ids)&set(validation_ids) or source_identities(training)&source_identities(validation):
            raise ValueError("AutoVMD fitting/selection sources overlap")
        if any(row["partition"]["role"] != "development" for row in training+validation):
            raise ValueError("AutoVMD cannot select on confirmation")
        excluded = set(search_config.get("excluded_sources",[]))
        if source_identities(training+validation)&excluded:
            raise ValueError("Outer evaluation sources entered AutoVMD selection")
        grid = self.config["classical_grid"]
        recipes = [FrozenVMDRecipe(k,alpha,self.config["fs"],grid["tolerance"],grid["max_iterations"])
                   for k,alpha in product(grid["K"],grid["alpha"])]
        evidence = []
        evaluator = search_config["evaluate"]
        for budget,keep in zip((5,15,80),(8,2,1)):
            scored = []
            for recipe in recipes:
                result = evaluator(recipe,budget,training_ids,validation_ids)
                if not np.isfinite(result["snr_db"]):
                    raise ValueError("Undefined selection SNR must be diagnosed, not discarded")
                record = {"recipe":asdict(recipe),"epochs":budget,**result}
                evidence.append(record)
                scored.append((recipe,record))
            scored.sort(key=lambda pair:(not pair[1]["preservation_passed"],-pair[1]["snr_db"]))
            best = scored[0][1]
            equivalent = [pair for pair in scored if pair[1]["preservation_passed"] == best["preservation_passed"]
                          and best["snr_db"]-pair[1]["snr_db"] <= .1]
            winner = min(equivalent,key=lambda pair:(pair[0].K,pair[1]["runtime_s"]))
            recipes = [winner[0]]+[pair[0] for pair in scored if pair[0] != winner[0]][:keep-1]
        return recipes[0],evidence


def build_cache(input_root,output,config,experiment):
    parent,rows = corpus_parent(input_root)
    rows = balanced_rows(rows) if experiment.get("selection_only",True) else [row for row in rows if row["target_kind"] == "controlled_recipient_reference"]
    settings = experiment.get("settings") or list(product(config["classical_grid"]["K"],config["classical_grid"]["alpha"]))
    permitted = set(product(config["classical_grid"]["K"],config["classical_grid"]["alpha"]))
    if any(tuple(setting) not in permitted for setting in settings):
        raise ValueError("Cache setting outside declared forty-setting grid")
    scope = experiment.get("scope","frontal")
    if scope not in ("frontal","all"):
        raise ValueError("Unknown cache scope")
    cache = output/"mode_cache"
    cache.mkdir()
    records = []
    solver = AutoVMD(config)
    for k,alpha in settings:
        recipe = FrozenVMDRecipe(k,alpha,config["fs"],config["classical_grid"]["tolerance"],config["classical_grid"]["max_iterations"])
        directory = cache/recipe.identity()
        directory.mkdir()
        for row in rows:
            started = time.perf_counter()
            source = parent/row["array_path"]
            data = np.load(source,allow_pickle=False)
            bundle = solver.transform(data["eeg"],recipe,data["regions"],scope)
            destination = directory/(row["example_id"]+".npz")
            np.savez_compressed(destination,components=bundle["components"],descriptors=bundle["descriptors"],valid=bundle["valid"])
            records.append({"example_id":row["example_id"],"source_sha256":sha256_file(source),
                "corpus_manifest_sha256":sha256_file(parent/"corpus_manifest.jsonl"),
                "recipe":asdict(recipe),"recipe_hash":recipe.identity(),"scope":scope,
                "path":str(destination.relative_to(output)),"sha256":sha256_file(destination),
                "runtime_s":time.perf_counter()-started,"diagnostics":bundle["diagnostics"]})
            print("VMD_CACHE",k,alpha,row["example_id"],flush=True)
        write_jsonl(output/"mode_cache_index.jsonl",records)
    atomic_json(output/"autovmd_cache_summary.json",{"examples":len(rows),"settings":len(settings),
        "scope":scope,"selection_only":experiment.get("selection_only",True),"entries":len(records),
        "reserved_confirmation_opened":False,"configuration_hash":canonical_hash(config)})
