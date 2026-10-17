"""Measured results and separate, unchanged engineering preservation gates."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .frontal_benchmark import METHODS
from .data import save_json


def write_review(output):
    output=Path(output)
    summary=pd.read_csv(output/'comparison.csv').set_index('method')
    records=pd.read_csv(output/'per_record.csv')
    clean=pd.read_csv(output/'clean_preservation.csv').groupby(['method','record']).mean(numeric_only=True)
    failures=json.loads((output/'failures.json').read_text())
    run=json.loads((output/'run_spec.json').read_text())
    completion=json.loads((output/'numerical_completion.json').read_text())
    config=json.loads((output/'configuration.json').read_text())
    comparator=records[records.method=='direct_eog_regression'].set_index('record').pearson_cc
    gates=[]
    for method in METHODS:
        selected=clean.loc[method]
        cc=records[records.method==method].set_index('record').pearson_cc
        row={'method':method,'worst_record_mean_clean_change':float(selected.relative_change.max()),
             'worst_record_mean_alpha_error_db':float(selected.alpha_error_db.max()),
             'worst_record_mean_beta_error_db':float(selected.beta_error_db.max()),
             'worst_record_mean_covariance_error':float(selected.covariance_relative_error.max()),
             'worst_record_cc_deterioration_vs_direct_ridge':float((comparator-cc).max()),
             'mean_output_snr_db':float(summary.loc[method,'pooled_window_snr_db']),
             'failed_windows':sum(x['method']==method for x in failures)}
        values=list(row.values())[1:]
        row['preservation_gates_pass']=bool(all(np.isfinite(x) for x in values) and
            row['worst_record_mean_clean_change']<=.01 and
            row['worst_record_mean_alpha_error_db']<=.5 and
            row['worst_record_mean_beta_error_db']<=.5 and
            row['worst_record_mean_covariance_error']<=.02 and
            row['worst_record_cc_deterioration_vs_direct_ridge']<=.005 and row['failed_windows']==0)
        row['development_15db_target_pass']=row['preservation_gates_pass'] and row['mean_output_snr_db']>=15
        row['production_qualified']=False
        gates.append(row)
    table=pd.DataFrame(gates)
    table.to_csv(output/'engineering_gates.csv',index=False)
    quality=[]
    for path in sorted((output/'windows').rglob('*_diagnostics.json')):
        for detail in json.loads(path.read_text()):
            if (detail.get('mode_cap_reached') or detail.get('vme_iteration_cap_reached') or
                detail.get('converged') is False or detail.get('fcm_converged') is False):
                quality.append({'artifact':str(path.relative_to(output)),**detail})
    save_json(output/'solver_quality_warnings.json',quality)
    columns=['pooled_window_snr_db','pooled_snr_improvement_db','rmse','mae','rrmse_t','rrmse_s','pearson_cc']
    text=f"# Klados frontal six-method results\n\nRun `{run['run_id']}`, SHA `{run['git_sha']}`.\n\n"
    text+=f"Smoke: **{completion['smoke']}**; {completion['records']} records, {completion['test_windows']} windows, seven frontal rows.\n\n"
    text+=summary.loc[METHODS,columns].round(6).to_markdown()+'\n\n'
    text+='Pooled-window SNR averages windows within record, then records in dB. Other metrics average channels/windows within record, then records. The old all-19-row 17.927210 dB is separate.\n\n'
    text+='## Reference access\n\n'
    for method in METHODS:text+=f"- {method}: {config['reference_access'][method]}.\n"
    text+='\n## Engineering preservation, separate from paper metrics\n\n'
    text+='Worst-record mean clean change<=1%, alpha/beta error<=0.5dB, covariance error<=0.02. The declared CC comparator here is matched direct EOG ridge, with worst-record mean deterioration<=0.005. Exceptions fail qualification.\n\n'
    text+=table.round(6).to_markdown(index=False)+'\n\n'
    text+=f"Failed method windows: {len(failures)}. Solver-cap/nonconvergence warning entries: {len(quality)}. See saved evidence; SVMD inner convergence is not exposed by its library.\n\n"
    text+='## Interpretation limits\n\nHistorically exposed Klados record split; participant IDs unknown. No fresh confirmation or production claim. Frontal assignment uses the requested paper-list order with unverified MATLAB row binding. SVMD–FCM and VME–GMETV are documented adaptations with unavailable ocular-paper settings. EEGOAR-Net is fine-tuned on an explicit custom montage, not frozen electrode aliases. No LEMON or AutoVMD search. Runtime rows disclose cached VMD cost and device. Actual executed notebook, predictions, modes, H/V evidence, checkpoints and figures are saved. Smoke scores are not final accuracy.\n'
    (output/'RESULTS.md').write_text(text,encoding='utf-8')
    return table
