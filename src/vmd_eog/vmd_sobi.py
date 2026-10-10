"""Bounded VMD–SOBI comparison; mechanism evidence cannot authorize a teacher."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .artifacts import verify_parent
from .corpus import unique_parent
from .experiments import corpus_parent,selection_rows
from .io import atomic_json,read_jsonl,write_jsonl
from .paper_metrics import paired_channels
from .reference import correlations,project
from .sobi import fit_sobi,approximate_entropy


DELAY_BANKS=((1,2,4,8,16,32),(1,5,10,20,40,80))


def source_artifact(expert, references, threshold, kind, lags, penalty):
    """Subtract reconstructed selected source contributions; retain other modes."""
    association=np.max(np.abs(correlations(expert.sources,references,20)),axis=1)
    selected=association>=threshold
    entropy=np.array([approximate_entropy(source) for source in expert.sources])
    if kind=="entropy_eog":
        selected &= entropy<.4
    if kind=="projected":
        contributions=project(expert.sources,references,lags,penalty)[0]
    elif kind in ("whole","entropy_eog"):
        contributions=expert.sources
    else:
        raise ValueError("Unknown mode-domain source correction")
    mode_artifact=expert.mixing[:,selected]@contributions[selected]
    return mode_artifact.sum(axis=0),{"association":association.tolist(),
        "approximate_entropy":entropy.tolist(),"selected_sources":np.flatnonzero(selected).tolist()}


def run_sobi_comparison(input_root,output,config,profile):
    corpus,rows=corpus_parent(input_root)
    parent=unique_parent(input_root,"vmd_summary.json").parent
    verify_parent(parent,("selected_frontal.json","mode_diagnostics.jsonl"))
    recipe=json.loads((parent/"selected_frontal.json").read_text())["vmd_projected"]
    if recipe is None:
        raise ValueError("No archived VMD recipe to inspect")
    infos={}
    for row in read_jsonl(parent/"mode_diagnostics.jsonl"):
        if row["K"]==recipe["K"] and row["alpha"]==recipe["alpha"]:
            infos.setdefault(row["example_id"],[]).append(row)
    examples=selection_rows(rows)
    if profile=="pilot":
        examples=examples[:12]
    predictions=output/"predictions"
    predictions.mkdir()
    scores=[]
    diagnostics=[]
    failures=[]
    for row in examples:
        data=dict(np.load(corpus/row["array_path"],allow_pickle=False))
        references=data["references"]
        for info in infos.get(row["example_id"],[]):
            verify_parent(parent,(info["array"],))
            saved=dict(np.load(parent/info["array"],allow_pickle=False))
            channel=data["channel_names"].tolist().index(info["channel"])
            raw=np.asarray(data["eeg"][channel],float)
            target=data["paired_reference"][channel]
            outer_association=float(np.max(np.abs(correlations(raw,references,20))))
            outer_gate=outer_association>=recipe["threshold"]
            direct=project(raw,references,recipe["lags"],recipe["penalty"])[0][0]
            corrections={"identity":np.zeros_like(raw),"direct_matched":direct*outer_gate*recipe["strength"]}
            unusable=info["hit_iteration_limit"] or bool(info.get("failure"))
            for bank,lags in enumerate(DELAY_BANKS):
                expert=None
                failure=None
                try:
                    if unusable:
                        raise RuntimeError("Archived VMD fit is unusable")
                    expert=fit_sobi(saved["modes"],lags=lags)
                    if not expert.diagnostics["converged"]:
                        raise RuntimeError("SOBI joint diagonalization reached sweep cap")
                    diagnostics.append({"example_id":row["example_id"],"channel":info["channel"],
                        "bank":bank,"raw_channel_association":outer_association,**expert.diagnostics})
                except Exception as error:
                    failure=f"{type(error).__name__}: {error}"
                    failures.append({"example_id":row["example_id"],"channel":info["channel"],"bank":bank,
                        "failure":failure,"status":"passthrough"})
                for threshold in config["classical_grid"]["thresholds"]:
                    for kind in ("whole","projected","entropy_eog"):
                        method=f"vmd_sobi_bank_{bank}_{kind}_threshold_{threshold:g}"
                        artifact=np.zeros_like(raw)
                        if failure is None and outer_gate:
                            artifact,selection=source_artifact(expert,references,threshold,kind,
                                recipe["lags"],recipe["penalty"])
                            artifact*=recipe["strength"]
                            diagnostics.append({"example_id":row["example_id"],"channel":info["channel"],
                                "bank":bank,"kind":kind,"threshold":threshold,**selection})
                        corrections[method]=artifact
            outputs={method:raw-artifact for method,artifact in corrections.items()}
            np.savez_compressed(predictions/(row["example_id"]+"-"+info["channel"]+".npz"),
                original=raw,target=target,references=references,**outputs)
            for method,cleaned in outputs.items():
                metrics=paired_channels(cleaned,target)
                scores.append({"example_id":row["example_id"],"recipient":row["recipient"],
                    "donor":row["donor"],"condition":row["condition"],"channel":info["channel"],
                    "method":method,**{key:float(values[0]) for key,values in metrics.items()},
                    "inference_target_access":False})
        print("VMD_SOBI",row["example_id"],flush=True)
        write_jsonl(output/"sobi_fit_failures.jsonl",failures)
    frame=pd.DataFrame(scores)
    frame.to_csv(output/"vmd_sobi_channels.csv",index=False)
    strict_mean=lambda values:np.asarray(values,float).mean()
    metrics=["snr_energy_db","rrmse_time","mse","pearson_cc"]
    source=frame[frame.condition!="clean"].groupby(["recipient","method"])[metrics].agg(strict_mean)
    summary=source.groupby("method")[metrics].agg(strict_mean)
    summary.to_csv(output/"vmd_sobi_method_means.csv")
    clean=frame[frame.condition=="clean"].groupby(["recipient","method"]).rrmse_time.agg(strict_mean).groupby("method").max()
    clean.rename("worst_recipient_clean_rrmse").to_csv(output/"vmd_sobi_clean_preservation.csv")
    write_jsonl(output/"sobi_diagnostics.jsonl",diagnostics)
    from .vmd_diagnosis import plot_diagnosis
    plot_diagnosis(summary,clean,output)
    (output/"vmd_correction_controls.png").rename(output/"vmd_sobi_controls.png")
    atomic_json(output/"vmd_sobi_summary.json",{"profile":profile,"examples":frame.example_id.nunique(),
        "failure_count":len(failures),"archived_vmd_recipe":recipe,"sobi_delay_banks":[list(bank) for bank in DELAY_BANKS],
        "entropy":{"embedding":2,"radius_fraction":.15,"threshold":.4,"self_matches":True},
        "scope":"preferred frontal mechanism pilot/comparison; global development recipe, not grouped teacher selection",
        "paper_replication":False,"adaptation":"EOG gate/selection, explicit SOBI delays, archived grid VMD rather than paper SVM/GA",
        "reserved_confirmation_opened":False,"teacher_selection_authorized":False,"model_authorization":False})
