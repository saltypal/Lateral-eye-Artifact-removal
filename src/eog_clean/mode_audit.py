"""Evaluate each saved frontal mode against paired EEG and both EOG axes.

Clean targets are used for diagnostics only. They never select a runtime mode.
Low correlation with full clean EEG does not prove that a narrowband mode is
artifact-only; high ocular association does not prove exclusive contamination.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .data import save_json
from .frontal_correlations import verified_parent
from .frontal_figures import LABELS


def associations(vector,clean,eyes):
    """Signed Pearson coefficients in explicit clean/HEOG/VEOG order."""
    signals=[clean,eyes[0],eyes[1]]
    return np.array([np.corrcoef(vector,reference)[0,1] for reference in signals])


def audit_native_references(native_root,manifest,output):
    """Independently verify predictions, then repair saved mode-reference pairing.

    The original before/after table uses the correct unmodified LPF references.
    Its mode CSV used model-preprocessed references. Recompute mode diagnostics
    from the actual H/V arrays embedded alongside the saved mode vectors.
    """
    from .data import sha256
    entries={row['path']:row for row in manifest['files']}
    def verify(path):
        relative=str(path.relative_to(native_root)).replace('\\','/')
        if sha256(path)!=entries[relative]['sha256']:raise ValueError('Native saved array differs')
    source=native_root/'osf_frontal_before_after_per_channel.csv'
    verify(source)
    previous=pd.read_csv(source).set_index(['method','record','trial','channel'])
    checks=[]
    for path in sorted((native_root/'osf').rglob('trial_*_predictions.npz')):
        verify(path)
        trial=int(path.name.split('_')[1]);record=path.parent.name
        with np.load(path,allow_pickle=False) as archive:data={name:archive[name] for name in archive.files}
        for method in ['identity','direct_eog_regression','historical_vmd','svmd_fcm_adaptation',
                       'vme_gmetv_adaptation','eegoarnet_finetuned','gevd_mwf','bandroutenet']:
            for channel,name in enumerate(data['channel_names']):
                row={'record':record,'trial':trial,'method':method,'channel':str(name)}
                for reference in ['heog','veog']:
                    before=float(np.corrcoef(data['raw'][channel],data[reference])[0,1])
                    after=float(np.corrcoef(data[method][channel],data[reference])[0,1])
                    expected=previous.loc[(method,record,trial,name)]
                    if not np.allclose([before,after],
                        [expected[reference+'_cc_before'],expected[reference+'_cc_after']],
                        rtol=5e-6,atol=5e-6,equal_nan=True):
                        raise ValueError('Native saved prediction/reference CC differs from scoring table')
                    row.update({reference+'_cc_before':before,reference+'_cc_after':after})
                checks.append(row)
    if not checks or len(checks)!=len(previous):
        raise ValueError('Native prediction audit did not cover the complete scoring table')
    pd.DataFrame(checks).to_csv(output/'native_prediction_correlations_independently_verified.csv',index=False)
    rows=[]
    for path in sorted((native_root/'osf').rglob('trial_*_modes.npz')):
        verify(path)
        parts=path.stem.split('_');trial,start=int(parts[1]),int(parts[3])
        with np.load(path,allow_pickle=False) as archive:data={name:archive[name] for name in archive.files}
        eyes=np.stack([data['heog'],data['veog']])
        for channel,name in enumerate(data['channel_names']):
            for method in ['historical_vmd','svmd_fcm_adaptation','vme_gmetv_adaptation']:
                prefix=method+'_channel_'+str(channel)+'_'
                vectors=data['modes'][channel] if method=='historical_vmd' else data[prefix+'modes']
                residual=data['residual'][channel] if method=='historical_vmd' else data[prefix+'residual']
                components=[('mode',index+1,vector) for index,vector in enumerate(vectors)]
                components.append(('residual',0,residual))
                for component,index,vector in components:
                    heog=float(np.corrcoef(vector,eyes[0])[0,1])
                    veog=float(np.corrcoef(vector,eyes[1])[0,1])
                    rows.append({'record':path.parent.name,'trial':trial,'start':start,
                        'channel':str(name),'method':method,'component':component,'mode':index,
                        'heog_cc':heog,'veog_cc':veog,'heog_abs_cc':abs(heog),'veog_abs_cc':abs(veog),
                        'reference':'unchanged evaluation references embedded in saved mode NPZ'})
    if not rows:
        raise ValueError('Native component audit found no saved modes')
    pd.DataFrame(rows).to_csv(output/'native_modes_unchanged_reference_correlations.csv',index=False)
    save_json(output/'native_reference_audit.json',{'native_prediction_rows_verified':len(checks),
        'native_component_rows_recomputed':len(rows),
        'before_after_prediction_table_matches_saved_signals':True,
        'mode_diagnostic_reference_repair':'original CSV used model-preprocessed references; recomputed from unchanged saved evaluation references',
        'denoising_repeated':False,'training_repeated':False,'original_native_artifacts_preserved':True})


def run(input_root,source_root,output):
    if not Path('/kaggle/input').exists():raise RuntimeError('Numerical work is Kaggle only')
    output=Path(output)
    config={'parent_run_id':'frontal-six-full-001',
            'parent_git_sha':'c19637465e5a5c29991c2a988f25e6527a156f09'}
    parent,verify=verified_parent(input_root,config)
    labels=json.loads((Path(source_root)/'docs/provenance/klados_montage.json').read_text())
    names=labels['frontal_names']
    rows=[]
    for directory in sorted((parent/'windows').iterdir()):
        key=json.loads((directory/'completion.json').read_text())
        with np.load(verify(str((directory/'predictions.npz').relative_to(parent)).replace('\\','/')),allow_pickle=False) as archive:
            clean,eyes=archive['clean'],archive['eyes']
        with np.load(verify(str((directory/'historical_modes.npz').relative_to(parent)).replace('\\','/')),allow_pickle=False) as archive:
            historical={name:archive[name] for name in ['modes','residual']}
        for channel,name in enumerate(names):
            for method in ['historical_vmd','svmd_fcm_adaptation','vme_gmetv_adaptation']:
                if method=='historical_vmd':
                    modes,residual=historical['modes'][channel],historical['residual'][channel]
                else:
                    path=directory/f'{method}_dirty_{channel}.npz'
                    with np.load(verify(str(path.relative_to(parent)).replace('\\','/')),allow_pickle=False) as archive:
                        modes,residual=archive['modes'],archive['residual']
                components=[('mode',index+1,vector) for index,vector in enumerate(modes)]
                components.append(('residual',0,residual))
                for component,index,vector in components:
                    cc=associations(vector,clean[channel],eyes)
                    frequency=np.fft.rfftfreq(len(vector),1/200)
                    power=np.abs(np.fft.rfft(vector))**2
                    center=float(frequency@power/max(power.sum(),1e-30))
                    rows.append({'method':method,'record':key['record'],'start':key['start'],
                        'channel':name,'source_row_index':labels['frontal_indices'][channel],
                        'component':component,'mode':index,'total_modes':len(modes),
                        'measured_center_hz':center,'clean_eeg_cc':cc[0],
                        'heog_cc':cc[1],'veog_cc':cc[2],
                        'clean_eeg_abs_cc':abs(cc[0]),'heog_abs_cc':abs(cc[1]),'veog_abs_cc':abs(cc[2])})
    frame=pd.DataFrame(rows)
    frame.to_csv(output/'frontal_mode_clean_heog_veog_correlations.csv',index=False)
    metrics=['measured_center_hz','clean_eeg_cc','heog_cc','veog_cc',
             'clean_eeg_abs_cc','heog_abs_cc','veog_abs_cc']
    record=frame.groupby(['method','record','channel','component','mode'],as_index=False)[metrics].mean()
    record.to_csv(output/'frontal_mode_correlations_per_record.csv',index=False)
    summary=record.groupby(['method','channel','component','mode'],as_index=False)[metrics].mean()
    summary.to_csv(output/'frontal_mode_correlation_summary.csv',index=False)
    save_json(output/'audit_completion.json',{'complete':True,'paired_dataset':'Klados',
        'records':int(frame.record.nunique()),'windows':len(list((parent/'windows').iterdir())),
        'frontal_channels':names,'component_rows':len(frame),
        'mode_methods':['historical_vmd','svmd_fcm_adaptation','vme_gmetv_adaptation'],
        'clean_targets_used_for_runtime_selection':False,'label_policy':labels,
        'undefined_correlation_values':int(frame[['clean_eeg_cc','heog_cc','veog_cc']].isna().sum().sum())})
    directory=output/'figures';directory.mkdir(exist_ok=True)
    paths=[]
    def save(figure,name):
        figure.savefig(directory/(name+'.png'),dpi=160,bbox_inches='tight')
        figure.savefig(directory/(name+'.pdf'),bbox_inches='tight')
        paths.append(str(directory/(name+'.png')));plt.close(figure)
    historical=summary[(summary.method=='historical_vmd')&(summary.component=='mode')]
    historical=historical.set_index(['channel','mode']).sort_index()
    figure,axis=plt.subplots(figsize=(10,11))
    columns=['clean_eeg_abs_cc','heog_abs_cc','veog_abs_cc']
    table=historical[columns]
    picture=axis.imshow(table.values,aspect='auto',vmin=0,vmax=1,cmap='magma')
    axis.set(yticks=range(len(table)),yticklabels=[f'{name}, mode {mode}' for name,mode in table.index],
             xticks=[0,1,2],xticklabels=['Paired clean EEG','HEOG','VEOG'],
             title='Frontal K3/alpha2000: mode correlations, equal record weight')
    for row in range(len(table)):
        for column in range(3):axis.text(column,row,f'{table.iloc[row,column]:.2f}',ha='center',va='center',color='white')
    figure.colorbar(picture,ax=axis,label='Mean absolute Pearson CC')
    save(figure,'01_modes_clean_eeg_and_eog')
    figure,axes=plt.subplots(1,3,figsize=(16,5),sharex=True,sharey=True)
    for axis,method in zip(axes,['historical_vmd','svmd_fcm_adaptation','vme_gmetv_adaptation']):
        selected=frame[(frame.method==method)&(frame.component=='mode')]
        x=selected[['heog_abs_cc','veog_abs_cc']].max(axis=1)
        picture=axis.scatter(x,selected.clean_eeg_abs_cc,c=selected.measured_center_hz,
                            vmin=0,vmax=40,s=10,alpha=.5,cmap='viridis')
        axis.set(xlim=(0,1),ylim=(0,1),xlabel='max(|HEOG CC|, |VEOG CC|)',
                 title=LABELS[method]);axis.grid(alpha=.2)
    axes[0].set_ylabel('|Correlation with paired clean EEG|')
    figure.colorbar(picture,ax=list(axes),label='Measured spectral center (Hz)')
    figure.suptitle('Each point is one actual frontal mode; upper-right can contain both EEG and ocular activity')
    save(figure,'02_modes_joint_associations')
    residual=summary[summary.component=='residual'].groupby('method')[columns].mean()
    figure,axis=plt.subplots(figsize=(10,4))
    picture=axis.imshow(residual.values,aspect='auto',vmin=0,vmax=1,cmap='magma')
    axis.set(yticks=range(len(residual)),yticklabels=[LABELS[name] for name in residual.index],
             xticks=[0,1,2],xticklabels=['Paired clean EEG','HEOG','VEOG'],
             title='Retained reconstruction residual associations; residual is not an extra VMD mode')
    figure.colorbar(picture,ax=axis);save(figure,'03_residual_associations')
    text='# Frontal mode correlation audit\n\n'
    text+='Each saved contaminated-input mode is correlated with the paired clean EEG channel, unchanged HEOG and unchanged VEOG. Signed and absolute coefficients, channel names, source rows, actual mode count and measured spectral centers are saved for every record/window. Residuals are audited separately.\n\n'
    text+=historical[columns+['measured_center_hz']].round(6).to_markdown()+'\n\n'
    text+='Clean EEG is available only in paired Klados. It is used for this diagnostic, not for runtime component selection. A narrowband neural mode can correlate weakly with full-band clean EEG; low clean CC is not proof of artifact. High EOG CC is not proof of exclusive ocular content. The unverified paper-list-to-row binding remains explicit. Native OSF has no corresponding clean target.\n'
    (output/'RESULTS.md').write_text(text,encoding='utf-8')
    # Freeze native evidence from the other completed notebook. This is table
    # summarization only; no further native fitting or parameter selection.
    native=[]
    for path in Path(input_root).rglob('artifact_manifest.json'):
        manifest=json.loads(path.read_text())
        if manifest.get('run_id')=='frontal-osf-correlations-001':native.append((path,manifest))
    if len(native)!=1 or not native[0][1]['complete'] or native[0][1]['git_sha']!='b9df0e65e0b7135aae45f9614439272bda88a8f3':
        raise ValueError('Need the exact completed native frontal parent')
    native_path,native_manifest=native[0]
    audit_native_references(native_path.parent,native_manifest,output)
    entries={row['path']:row for row in native_manifest['files']}
    from .data import sha256
    for name in ['osf_frontal_rest_preservation.csv','osf_frontal_correlation_summary.csv','native_completion.json']:
        path=native_path.parent/name
        if sha256(path)!=entries[name]['sha256']:raise ValueError('Native parent payload differs: '+name)
    rest=pd.read_csv(native_path.parent/'osf_frontal_rest_preservation.csv')
    names=['relative_change','resting_input_cc','alpha_error_db','beta_error_db','covariance_relative_error']
    participants=rest.groupby(['method','participant'],as_index=False)[names].mean()
    mean=participants.groupby('method')[names].mean().add_prefix('participant_mean_')
    worst=participants.groupby('method')[names].max().add_prefix('maximum_participant_mean_')
    pd.concat([mean,worst],axis=1).to_csv(output/'native_rest_participant_summary.csv')
    participants.to_csv(output/'native_rest_per_participant.csv',index=False)
    native_table=pd.read_csv(native_path.parent/'osf_frontal_correlation_summary.csv')
    native_table.to_csv(output/'native_frontal_before_after_summary.csv',index=False)
    save_json(output/'native_parent_identity.json',{'run_id':'frontal-osf-correlations-001',
        'source_sha':native_manifest['git_sha'],
        'kernel':'satyapaladugu/frontal-klados-and-osf-correlations/1',
        'reference_evaluation':'unchanged source HEOG/VEOG; native has no paired clean target'})
    with (output/'RESULTS.md').open('a',encoding='utf-8') as stream:
        stream.write('\n## Native frontal before/after correlation\n\n'+native_table.round(6).to_markdown(index=False))
        stream.write('\n\n## Native resting-input preservation proxies\n\n'+pd.concat([mean,worst],axis=1).round(6).to_markdown())
        stream.write('\n\n## Native reference consistency audit\n\nOriginal before/after prediction correlations were independently verified against saved unmodified HEOG/VEOG. Original mode diagnostics used model-preprocessed references; the corrected per-component file recomputes those coefficients using the unchanged references embedded beside the saved modes. No denoising, training or scoring-table result changed. Original evidence remains preserved.\n')
    print('FRONTAL_MODE_CLEAN_EOG_AUDIT_COMPLETE',len(frame),flush=True)
    return paths
