"""Frontal comparison evidence; run only in the Kaggle experiment notebook."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .data import save_json
from .pretrained_review import verified_table_parent


def bootstrap_difference(first, second, seed=42, repeats=2000):
    """Paired cluster resampling, never channel/window pseudoreplication."""
    first,second=first.align(second,join='inner')
    if len(first)<2 or not np.isfinite(first).all() or not np.isfinite(second).all():
        raise ValueError('Insufficient finite common clusters for paired uncertainty')
    differences=(first-second).to_numpy()
    rng=np.random.default_rng(seed)
    means=differences[rng.integers(0,len(differences),(repeats,len(differences)))].mean(axis=1)
    return {'clusters':len(first),'mean_difference':float(differences.mean()),
            'ci95_lower':float(np.quantile(means,.025)),
            'ci95_upper':float(np.quantile(means,.975))}


def report(output,input_root):
    import matplotlib.pyplot as plt
    output=Path(output)
    verify=verified_table_parent(input_root,'frontal-six-full-001','c19637465e5a5c29991c2a988f25e6527a156f09')
    paired=pd.read_csv(output/'klados_all_per_channel.csv')
    clean=pd.concat([pd.read_csv(verify('clean_preservation.csv')),
                     pd.read_csv(output/'new_klados_clean_controls.csv')],ignore_index=True)
    clean.groupby(['method','record'],as_index=False).mean(numeric_only=True).to_csv(output/'klados_record_clean_preservation.csv',index=False)
    records=paired.groupby(['method','record'],as_index=False).mean(numeric_only=True)
    records.to_csv(output/'klados_per_record.csv',index=False)
    controls=clean.groupby(['method','record']).mean(numeric_only=True)
    native=pd.read_csv(output/'osf_participant_correlations.csv')
    rest=pd.read_csv(output/'osf_rest_worst_participant.csv').set_index('method')
    failures=json.loads((output/'method_failures.json').read_text())
    comparator=records[records.method=='direct_calibrated_ridge'].set_index('record')
    gates=[]
    for method in sorted(rest.index):
        row={'method':method,'native_rest_proxy_pass':bool(
            rest.loc[method,'relative_change']<=.01 and rest.loc[method,'alpha_error_db']<=.5
            and rest.loc[method,'beta_error_db']<=.5 and rest.loc[method,'covariance_relative_error']<=.02),
            'native_worst_participant_rest_change':float(rest.loc[method,'relative_change']),
            'paired_preservation_pass':False,'paired_evidence_available':method in records.method.unique(),
            'production_qualified':False}
        if row['paired_evidence_available']:
            selected=controls.loc[method]
            scores=records[records.method==method].set_index('record')
            row.update(worst_record_clean_change=float(selected.relative_change.max()),
                worst_record_alpha_db=float(selected.alpha_error_db.max()),
                worst_record_beta_db=float(selected.beta_error_db.max()),
                worst_record_covariance=float(selected.covariance_relative_error.max()),
                worst_record_cc_deterioration_vs_calibrated_ridge=float((comparator.pearson_cc-scores.pearson_cc).max()))
            row['paired_preservation_pass']=bool(row['worst_record_clean_change']<=.01 and
                row['worst_record_alpha_db']<=.5 and row['worst_record_beta_db']<=.5 and
                row['worst_record_covariance']<=.02 and row['worst_record_cc_deterioration_vs_calibrated_ridge']<=.005)
        row['new_method_failure_events']=sum(item['method']==method for item in failures)
        # Earlier failures remain documented in their pinned parents; no new qualification inferred.
        row['complete_preservation_evidence_pass']=bool(row['paired_preservation_pass'] and
            row['native_rest_proxy_pass'] and row['new_method_failure_events']==0)
        gates.append(row)
    pd.DataFrame(gates).to_csv(output/'preservation_review.csv',index=False)
    uncertainty=[]
    for method in ['regica','historical_vmd']:
        if method not in records.method.unique():continue
        selected=records[records.method==method].set_index('record')
        for metric in ['pooled_window_snr_db','rmse','pearson_cc']:
            uncertainty.append({'dataset':'Klados','method':method,'comparator':'direct_calibrated_ridge',
                'metric':metric,**bootstrap_difference(selected[metric],comparator[metric])})
    for method in ['sgeyesub','regica']:
        for condition,metric in [('blink','veog_abs_cc_after'),('lateral','heog_abs_cc_after')]:
            selected=native[(native.method==method)&(native.condition==condition)].set_index('participant')
            control=native[(native.method=='direct_eog_regression')&(native.condition==condition)].set_index('participant')
            uncertainty.append({'dataset':'OSF','method':method,'comparator':'direct_eog_regression',
                'condition':condition,'metric':metric,**bootstrap_difference(selected[metric],control[metric])})
    pd.DataFrame(uncertainty).to_csv(output/'paired_cluster_uncertainty.csv',index=False)
    paths=[]
    directory=output/'figures';directory.mkdir(exist_ok=True)
    def save(figure,name):
        figure.tight_layout()
        for extension in ['png','pdf']:figure.savefig(directory/(name+'.'+extension),dpi=150,bbox_inches='tight')
        paths.append(str(directory/(name+'.png')));plt.close(figure)
    summary=pd.read_csv(output/'klados_comparison.csv')
    for metric,label in [('pooled_window_snr_db','Output SNR (dB)'),('rmse','RMSE'),('pearson_cc','Pearson CC with clean EEG')]:
        figure,axis=plt.subplots(figsize=(10,5))
        axis.barh(summary.method,summary[metric]);axis.set(xlabel=label,title='Same Klados frontal scoring windows; equal record weight')
        axis.grid(axis='x',alpha=.2);save(figure,'klados_'+metric)
    table=pd.read_csv(output/'osf_comparison.csv')
    for condition,reference in [('blink','veog'),('lateral','heog'),('vertical','veog')]:
        selected=table[table.condition==condition]
        figure,axis=plt.subplots(figsize=(10,5));axis.barh(selected.method,selected[reference+'_abs_cc_after'])
        axis.axvline(float(selected[reference+'_abs_cc_before'].iloc[0]),color='black',linestyle='--',label='Before')
        axis.set(xlabel='Mean absolute Pearson CC with unchanged '+reference.upper(),xlim=(0,1),title='Native frontal '+condition+'; equal participant weight')
        axis.legend();save(figure,'osf_'+condition)
    figure,axes=plt.subplots(1,2,figsize=(14,5))
    axes[0].barh(rest.index,100*rest.relative_change);axes[0].axvline(1,color='black',linestyle='--')
    axes[0].set(xlabel='Worst participant mean resting modification (%)')
    axes[1].barh(rest.index,rest.covariance_relative_error);axes[1].axvline(.02,color='black',linestyle='--')
    axes[1].set(xlabel='Worst participant mean relative covariance error')
    figure.suptitle('Resting preservation proxy; dashed fixed engineering limits');save(figure,'osf_preservation')
    solver=[]
    for path in output.rglob('solver_diagnostics.json'):
        solver.append({'path':str(path.relative_to(output)),**json.loads(path.read_text())})
    save_json(output/'ica_solver_review.json',solver)
    text='# Frontal full-montage SGEYESUB and REG-ICA comparison\n\n'
    text+='Original author routines through Octave. Spatial fits use the full montage; only frontal outputs are scored. '
    text+='SGEYESUB requires native OSF six-class calibration and is not run on unlabelled Klados. '
    text+='REG-ICA uses calibration-frozen extended Infomax/fractal selection and trial/window-reset original SRLS; this is a declared evaluation adaptation. '
    text+='No VMD on nonfrontal signals, LEMON, AutoVMD search or neural training.\n\n'
    text+=summary.round(6).to_markdown(index=False)+'\n\n'+table.round(6).to_markdown(index=False)+'\n\n'
    text+=pd.DataFrame(gates).round(6).to_markdown(index=False)+'\n\n'
    text+=pd.DataFrame(uncertainty).round(6).to_markdown(index=False)+'\n\n'
    text+='Paired uncertainty resamples eight record clusters (participant identities unknown); native uncertainty resamples four participants, exploratory and unstable. '
    text+='Raw/target scoring arrays match pinned previous comparisons exactly; calibration/training and information access remain different. '
    text+='The newly calibrated ridge control is not the earlier score-window-fitted ridge. '
    text+='Clean targets never fit these new methods. Native rest is a proxy, without reconstruction SNR. '
    text+='Recovery reuses checksum-verified completed outputs and calibration models after input identity checks; actual scored frontal rows use the archived bytes. '
    text+='Permitted one-float32-ULP preprocessing differences are recorded separately from unchanged scientific thresholds. Runtime.csv covers this execution; original fit costs for recovered models remain in the failed parent logs. '
    text+='All failure cases remain as identity with explicit logs; solver step caps are not convergence. '
    text+='Reused development evidence cannot establish production qualification or regional integration.\n'
    (output/'RESULTS.md').write_text(text,encoding='utf-8')
    return paths
