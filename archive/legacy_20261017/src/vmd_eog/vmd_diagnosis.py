"""Inspect VMD correction failure mechanisms without changing old results."""
import json
from pathlib import Path
import numpy as np
import pandas as pd

from .artifacts import verify_parent
from .corpus import unique_parent
from .experiments import corpus_parent, selection_rows
from .io import atomic_json, read_jsonl
from .paper_metrics import paired_channels
from .reference import project, calibration_reference_baseline, correlations
from .frontal import mode_features


def correction_controls(mode_projection, residual_projection, selected_modes,
                        channel_gate, strength, unusable):
    """Isolate mode rejection, residual retention and convergence policy.

    Inputs contain EEG/EOG projections only. Targets never select a mode or
    correction. The all-components control is a linearity check, not a claim
    that VMD adds information to ordinary reference regression.
    """
    modes = np.asarray(mode_projection, float)
    residual = np.asarray(residual_projection, float).reshape(-1)
    selected = np.asarray(selected_modes, bool)
    if modes.ndim != 2 or selected.shape != (len(modes),):
        raise ValueError("Mode projections and selection mask disagree")
    if residual.shape != (modes.shape[-1],):
        raise ValueError("Residual and modes have different sample lengths")
    all_modes = modes.sum(axis=0)
    selected_correction = modes[selected].sum(axis=0)
    valid = 0. if unusable else 1.
    gated_strength = float(channel_gate)*strength
    return {
        "vmd_all_modes_channel_gate": valid*all_modes*gated_strength,
        "vmd_selected_plus_residual": valid*(selected_correction+residual)*strength,
        "vmd_all_components_converged": valid*(all_modes+residual)*gated_strength,
        "vmd_all_components_closure": (all_modes+residual)*gated_strength,
    }


def run_diagnosis(input_root, output, config, profile):
    """Use saved vectors, never refit VMD or tune on reserved sources."""
    corpus, rows=corpus_parent(input_root)
    vmd=unique_parent(input_root,"vmd_summary.json").parent
    verify_parent(vmd,("selected_frontal.json","mode_diagnostics.jsonl"))
    selected=json.loads((vmd/"selected_frontal.json").read_text())["vmd_projected"]
    if selected is None:
        raise ValueError("No preliminary projected recipe available for diagnosis")
    infos=[row for row in read_jsonl(vmd/"mode_diagnostics.jsonl")
           if row["K"]==selected["K"] and row["alpha"]==selected["alpha"]]
    lookup={}
    for info in infos:
        lookup.setdefault(info["example_id"],[]).append(info)
    records=[]
    closure=[]
    mode_rows=[]
    for row in selection_rows(rows):
        data=dict(np.load(corpus/row["array_path"],allow_pickle=False))
        calibration=dict(np.load(corpus/row["calibration"],allow_pickle=False))
        baseline=calibration_reference_baseline(calibration["references"],calibration["boundaries"],selected["lags"])
        for info in lookup.get(row["example_id"],[]):
            verify_parent(vmd,(info["array"],))
            saved=dict(np.load(vmd/info["array"],allow_pickle=False))
            index=data["channel_names"].tolist().index(info["channel"])
            raw=np.asarray(data["eeg"][index],float)
            target=data["paired_reference"][index]
            modes,residual=saved["modes"],saved["residual"]
            references=data["references"]
            selected_modes=saved["features"][:,0]>=selected["threshold"]
            current_features=mode_features(modes,references,config["fs"])
            unusable=info["hit_iteration_limit"] or bool(info.get("failure"))
            artifacts={}
            direct,_=project(raw,references,selected["lags"],selected["penalty"])
            centered_modes,_=project(modes,references,selected["lags"],selected["penalty"])
            residual_projection,_=project(residual,references,selected["lags"],selected["penalty"])
            closure_error=direct[0]-(centered_modes.sum(0)+residual_projection[0])
            closure.append({"example_id":row["example_id"],"channel":info["channel"],
                "projection_closure_relative_error":float(np.linalg.norm(closure_error)/max(np.linalg.norm(direct),1e-30)),
                "reconstruction_relative_error":float(np.linalg.norm(raw-modes.sum(0)-residual)/max(np.linalg.norm(raw),1e-30)),
                "selected_modes":int(selected_modes.sum()),"total_modes":len(modes),
                "discarded_mode_projection_rms":float(np.sqrt(np.mean(centered_modes[~selected_modes].sum(0)**2))),
                "selected_mode_projection_rms":float(np.sqrt(np.mean(centered_modes[selected_modes].sum(0)**2))),
                "iteration_cap":unusable,"residual_projection_rms":float(np.sqrt(np.mean(residual_projection**2))),
                "input_true_artifact_mean":float(np.mean(data["true_artifact"][index])),
                "inference_uses_true_artifact":False})
            for mode_index, projection in enumerate(centered_modes):
                mode_rows.append({"example_id":row["example_id"],"recipient":row["recipient"],
                    "condition":row["condition"],"channel":info["channel"],"mode":mode_index+1,
                    "center_hz":float(saved["centers_hz"][mode_index]),
                    "eog_association":float(saved["features"][mode_index,0]),
                    "unit_invariant_eog_association":float(current_features[mode_index,0]),
                    "selected":bool(selected_modes[mode_index]),
                    "projected_rms":float(np.sqrt(np.mean(projection**2))),
                    "iteration_cap":unusable})
            association=float(np.max(np.abs(correlations(raw,references,20))))
            channel_gate=association>=selected["threshold"]
            anchored_direct,_=project(raw,references,selected["lags"],selected["penalty"],reference_baseline=baseline)
            anchored_modes,_=project(modes,references,selected["lags"],selected["penalty"],reference_baseline=baseline)
            artifacts["identity"]=np.zeros_like(raw)
            artifacts["direct_matched_centered"]=direct[0]*channel_gate*selected["strength"]
            artifacts["direct_matched_calibration_baseline"]=anchored_direct[0]*channel_gate*selected["strength"]
            artifacts["vmd_original_projected"]=np.zeros_like(raw) if unusable else (centered_modes*selected_modes[:,None]).sum(0)*selected["strength"]
            artifacts["vmd_calibration_baseline"]=np.zeros_like(raw) if unusable else (anchored_modes*selected_modes[:,None]).sum(0)*selected["strength"]
            # Closure is an algebraic control, not an independent VMD gain.
            artifacts.update(correction_controls(centered_modes,residual_projection,selected_modes,
                channel_gate,selected["strength"],unusable))
            # Declared-grid threshold ablations diagnose sensitivity only.
            # This notebook does not select a replacement teacher from them.
            for threshold in config["classical_grid"]["thresholds"]:
                mask=saved["features"][:,0]>=threshold
                artifacts[f"vmd_mode_threshold_{threshold:g}"]=np.zeros_like(raw) if unusable else centered_modes[mask].sum(0)*selected["strength"]
                current_mask=current_features[:,0]>=threshold
                artifacts[f"vmd_unit_invariant_threshold_{threshold:g}"]=np.zeros_like(raw) if unusable else centered_modes[current_mask].sum(0)*selected["strength"]
            # A stricter raw-channel trigger may protect clean inputs while
            # allowing lower-association contributions inside ocular windows.
            # The paper uses an SVM trigger; this is an EOG-assisted adaptation.
            for bank_index,lags in enumerate(config["classical_grid"]["lag_banks"]):
                projected_modes,_=project(modes,references,lags,selected["penalty"])
                matched_direct,_=project(raw,references,lags,selected["penalty"])
                artifacts[f"direct_bank_{bank_index}"]=matched_direct[0]*channel_gate*selected["strength"]
                for outer in config["classical_grid"]["thresholds"]:
                    for inner in config["classical_grid"]["thresholds"]:
                        name=f"vmd_dual_bank_{bank_index}_outer_{outer:g}_mode_{inner:g}"
                        artifacts[name]=dual_gate_correction(projected_modes,current_features[:,0],
                            association,outer,inner,selected["strength"],unusable)
            for method,artifact in artifacts.items():
                cleaned=raw-artifact
                metrics=paired_channels(cleaned,target)
                error=np.asarray(cleaned-target,float)
                mean_error=float(error.mean())
                mse=float(np.mean(error**2))
                centered_mse=float(np.mean((error-mean_error)**2))
                records.append({"example_id":row["example_id"],"recipient":row["recipient"],"donor":row["donor"],
                    "condition":row["condition"],"channel":info["channel"],"method":method,
                    **{key:float(values[0]) for key,values in metrics.items()},
                    "mean_error":mean_error,"bias_mse":mean_error**2,"centered_mse":centered_mse,
                    "mse_partition_error":abs(mse-mean_error**2-centered_mse),
                    "mean_bias_fraction":mean_error**2/mse if mse>0 else np.nan,
                    "iteration_cap":unusable,"selected_modes":int(selected_modes.sum()),
                    "inference_target_access":False})
        print("VMD_DIAGNOSIS",row["example_id"],flush=True)
    frame=pd.DataFrame(records)
    frame.to_csv(output/"vmd_diagnosis_channels.csv",index=False)
    pd.DataFrame(closure).to_csv(output/"vmd_projection_closure.csv",index=False)
    pd.DataFrame(mode_rows).to_csv(output/"vmd_mode_projection_diagnostics.csv",index=False)
    fields=["snr_energy_db","rrmse_time","pearson_cc","mean_bias_fraction"]
    dirty=frame[frame.condition!="clean"].groupby(["recipient","method"])[fields].agg(lambda values:np.asarray(values,float).mean())
    summary=dirty.groupby("method")[fields].agg(lambda values:np.asarray(values,float).mean())
    summary.to_csv(output/"vmd_diagnosis_method_means.csv")
    clean=frame[frame.condition=="clean"].groupby(["recipient","method"]).rrmse_time.mean().groupby("method").max()
    clean.rename("worst_recipient_clean_rrmse").to_csv(output/"vmd_diagnosis_clean_preservation.csv")
    plot_diagnosis(summary,clean,output)
    atomic_json(output/"vmd_diagnosis_summary.json",{
        "profile":profile,"recipe":selected,"examples":frame.example_id.nunique(),
        "scope":"preliminary preferred frontal selection channels; mechanism diagnosis, not final teacher approval",
        "hypotheses":["Per-mode gating excludes ocular contributions", "VMD residual contains projected ocular activity",
                      "Zero-mean scoring projection retains ocular baseline excursions"],
        "experimental_baseline":"unscored calibration lag-major EOG means; coefficients still fit scoring EEG without targets",
        "two_stage_detection":"declared raw-channel and mode association threshold banks; EOG-assisted adaptation, not paper SVM replication",
        "dual_gate_selection_status":"mechanism ablation only; no replacement recipe selected or authorized",
        "mean_removed_snr_is_acceptance_metric":False,
        "reserved_confirmation_opened":False,"model_authorization":False,
        "max_projection_closure_relative_error":max(row["projection_closure_relative_error"] for row in closure),
        "max_reconstruction_relative_error":max(row["reconstruction_relative_error"] for row in closure)})


def plot_diagnosis(summary, clean, output):
    """Executed on Kaggle; juxtapose recovery with clean-input modification."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    methods=[method for method in summary.index if "vmd_dual_" not in method or
             method.startswith("vmd_dual_bank_0_outer_0.6_")]
    fig,axes=plt.subplots(1,2,figsize=(16,11),sharey=True)
    axes[0].scatter(summary.loc[methods,"snr_energy_db"],np.arange(len(methods)))
    axes[0].set_xlabel("Fz development output SNR (dB), recipient mean")
    axes[1].scatter(100*clean.reindex(methods),np.arange(len(methods)))
    axes[1].axvline(1.,color="#555555",linestyle="--")
    axes[1].set_xlabel("Worst recipient clean RRMSE (%)")
    axes[0].set_yticks(np.arange(len(methods)),methods)
    axes[0].invert_yaxis()
    for axis in axes:
        axis.grid(axis="x",alpha=.15)
    fig.suptitle("Correction mechanism controls: fixed K/alpha, preliminary Fz subset")
    fig.tight_layout()
    fig.savefig(output/"vmd_correction_controls.png",dpi=160)
    plt.close(fig)


def dual_gate_correction(projections, mode_associations, channel_association,
                         outer_threshold, mode_threshold, strength, unusable=False):
    """Estimate only from EEG/EOG; mode sensitivity cannot bypass the outer gate."""
    projections=np.asarray(projections,float)
    associations=np.asarray(mode_associations,float)
    if projections.ndim!=2 or associations.shape!=(len(projections),):
        raise ValueError("One association per mode projection is required")
    if not np.isfinite(projections).all() or not np.isfinite(associations).all() or not np.isfinite(channel_association):
        raise ValueError("Finite inference projections and associations required")
    if unusable or abs(channel_association)<outer_threshold:
        return np.zeros(projections.shape[-1])
    return strength*projections[np.abs(associations)>=mode_threshold].sum(axis=0)
