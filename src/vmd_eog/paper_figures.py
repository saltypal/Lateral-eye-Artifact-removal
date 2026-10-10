"""Paper-metric figures with explicit participant and aggregation identities."""
import json
import numpy as np
import pandas as pd


LABELS={"identity":"Input / identity", "direct":"Direct EOG regression",
    "shared_vmd":"Shared VMD", "regional_mwf":"VMD + posterior MWF",
    "regional_ica":"VMD + posterior ICA", "regional_no_context":"VMD + posterior-only ICA",
    "regional_mwf_direct_frontal":"Regression + posterior MWF",
    "regional_ica_direct_frontal":"Regression + posterior ICA"}
METRICS=("rrmse_time","mse","pearson_cc","snr_energy_db")


def controlled_participants(source_table):
    """Equal input-level weight within each participant and ocular condition.

    Source tables already contain channel-first means. Strict means retain
    undefined/infinite values rather than quietly changing the cohort.
    """
    selected=source_table[(source_table.dataset=="controlled_LEMON_OSF") &
                          (source_table.condition!="clean")]
    return selected.groupby(["method","recipient","condition"])[list(METRICS)].agg(
        lambda values:np.asarray(values,float).mean()).reset_index()


def participant_interval(values, seed=42):
    """A participant bootstrap interval is unavailable for nonfinite units."""
    values=np.asarray(values,float)
    if not len(values) or not np.isfinite(values).all():
        return np.nan,np.nan,np.nan
    generator=np.random.default_rng(seed)
    draws=generator.choice(values,size=(10000,len(values)),replace=True).mean(axis=1)
    lower,upper=np.quantile(draws,[.025,.975])
    return float(values.mean()),float(lower),float(upper)


def point_interval(axis, values, position, seed):
    mean,lower,upper=participant_interval(values,seed)
    if not np.isfinite([mean,lower,upper]).all():
        axis.text(.02,position,"unavailable: nonfinite units",
                  transform=axis.get_yaxis_transform(),fontsize=8)
        return
    axis.errorbar(mean,position,xerr=[[max(0.,mean-lower)],[max(0.,upper-mean)]],
                  fmt="o",color="#315e82",capsize=3)


def plot_paper_review(regional, native, output, seed):
    """Saved source tables drive every figure; never recompute waveform metrics."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    path=regional/"paper_paired_source_means.csv"
    if path.exists():
        participants=controlled_participants(pd.read_csv(path))
        participants.to_csv(output/"paper_plot_participant_conditions.csv",index=False)
        if len(participants):
            methods=[method for method in LABELS if method in participants.method.values]
            combined=participants.groupby(["method","recipient"])[list(METRICS)].agg(
                lambda values:np.asarray(values,float).mean()).reset_index()
            fig,axes=plt.subplots(2,2,figsize=(16,11),sharey=True)
            names=("Time RRMSE (lower is better)","MSE (V²; lower is better)",
                   "Pearson CC (higher is better)","Channel-first output SNR (dB)")
            for axis,metric,label in zip(axes.flat,METRICS,names):
                for position,method in enumerate(methods):
                    point_interval(axis,combined.loc[combined.method==method,metric],position,seed)
                if metric=="snr_energy_db":
                    axis.axvline(15.,color="#555555",linestyle="--")
                axis.set_xlabel(label)
                axis.grid(axis="x",alpha=.15)
            for axis in axes[:,0]:
                axis.set_yticks(np.arange(len(methods)),[LABELS[m] for m in methods])
            axes[0,0].invert_yaxis()
            fig.suptitle("Primary paired metrics: development recipient means and bootstrap 95% intervals\n"
                         "Channel → example → input level → condition → recipient; controlled LEMON/OSF mixtures")
            fig.tight_layout()
            fig.savefig(output/"paper_primary_metrics.png",dpi=160)
            plt.close(fig)
            fig,axes=plt.subplots(1,3,figsize=(18,7),sharey=True)
            for axis,condition in zip(axes,("blink","lateral","mixed")):
                for position,method in enumerate(methods):
                    values=participants.loc[(participants.method==method)&
                                            (participants.condition==condition),"snr_energy_db"]
                    point_interval(axis,values,position,seed)
                axis.axvline(15.,color="#555555",linestyle="--")
                axis.set(title=condition,xlabel="Channel-first output SNR (dB)")
                axis.grid(axis="x",alpha=.15)
            axes[0].set_yticks(np.arange(len(methods)),[LABELS[m] for m in methods])
            axes[0].invert_yaxis()
            fig.suptitle("Development recipient means; participant-bootstrap 95% intervals; 15 dB project target")
            fig.tight_layout()
            fig.savefig(output/"paper_condition_SNR.png",dpi=160)
            plt.close(fig)
    if native is None:
        return
    participants=pd.read_csv(native/"native_condition_participants.csv")
    chance=json.loads((native/"native_chance_summary.json").read_text())
    fig,axes=plt.subplots(1,4,figsize=(19,6),sharey=True)
    methods=["identity","direct","regional_mwf","regional_ica"]
    for axis,condition in zip(axes,("rest","blink","lateral","vertical")):
        metric="rest_rmse" if condition=="rest" else "eeg_eog_abs_r_after"
        for position,method in enumerate(methods):
            values=participants.loc[(participants.method==method)&
                                    (participants.condition==condition),metric]
            point_interval(axis,values,position,seed)
        axis.set_title(condition)
        axis.set_xlabel("Rest RMSE (source units unverified)" if condition=="rest" else "Mean absolute EEG–EOG Pearson")
        axis.grid(axis="x",alpha=.15)
        for record in chance.get("conditions",[]):
            if record["condition"]==condition and record.get("status")=="computed adapted null":
                axis.axvline(record["p95"],color="#555555",linestyle=":")
    axes[0].set_yticks(np.arange(len(methods)),[LABELS[m] for m in methods])
    axes[0].invert_yaxis()
    fig.suptitle("Native OSF: complete-condition metrics, participant means and bootstrap 95% intervals\n"
                 "Dotted thresholds, if available, use an adapted eight-second null; native reconstruction SNR is unavailable")
    fig.tight_layout()
    fig.savefig(output/"paper_native_conditions.png",dpi=160)
    plt.close(fig)
