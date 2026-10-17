"""Matched frontal/posterior composition; all numerical work stays on Kaggle."""
from pathlib import Path
import itertools
import json
import tempfile
import time
import numpy as np
import pandas as pd
from .data import save_json,sha256,prepare_sources,klados,preprocess,region
from .methods import fit_ridge,fit_mwf
from .metrics import paired,preservation
from .central import correlation_rows,channel_group
from .pretrained_review import verified_table_parent
from .spatial_report import bootstrap_difference


def compose(original,frontal,posterior,frontal_indices,posterior_indices):
    """Two predictions relative to original EEG; disjoint masks, one subtraction."""
    first=list(frontal_indices);second=list(posterior_indices)
    if (len(set(first))!=len(first) or len(set(second))!=len(second) or set(first)&set(second)
            or not first or not second or min(first+second)<0 or max(first+second)>=len(original)):
        raise ValueError('Regional masks must be nonempty, valid and disjoint')
    if frontal.shape!=original[first].shape or posterior.shape!=original[second].shape:
        raise ValueError('Regional prediction axes differ from input')
    correction=np.zeros_like(original,dtype=float)
    correction[first]=np.asarray(original[first],float)-frontal
    correction[second]=np.asarray(original[second],float)-posterior
    result=(original-correction).astype(np.float32)
    if not np.isfinite(result).all():raise ValueError('Nonfinite regional prediction')
    unchanged=sorted(set(range(len(original)))-set(first+second))
    np.testing.assert_array_equal(result[unchanged],original[unchanged])
    return result


def run(input_root,source_root,output):
    if not Path('/kaggle/input').exists():raise RuntimeError('Numerical work runs on Kaggle')
    source_root,output=Path(source_root),Path(output)
    output.mkdir(parents=True,exist_ok=True)
    config=json.loads((source_root/'configs/frontal_six.json').read_text())
    source_config=json.loads((source_root/'configs/frontal_osf.json').read_text())
    montage=json.loads((source_root/'docs/provenance/klados_montage.json').read_text())
    names=montage['channel_names_in_paper_list_order'];front=montage['frontal_indices']
    back=[i for i,name in enumerate(names) if region(name)=='posterior']
    central=[i for i,name in enumerate(names) if channel_group(name)=='central']
    temporal=[i for i,name in enumerate(names) if channel_group(name)=='temporal']
    if set(front+back+central+temporal)!=set(range(19)):raise ValueError('Incomplete montage assignment')
    verify=verified_table_parent(input_root,'frontal-six-full-001','c19637465e5a5c29991c2a988f25e6527a156f09')
    parents=list(Path(input_root).rglob('snr_target_protocol.json'))
    if len(parents)!=1:raise ValueError('Require unique original full-montage cache')
    cache=parents[0].parent/'test_vmd_cache.npz'
    if sha256(cache)!=config['historical_test_cache_sha256']:raise ValueError('Full scoring cache checksum differs')
    with np.load(cache,allow_pickle=False) as stream:
        arrays={key:stream[key].copy() for key in ['raw','clean','eyes','keys']}
    work=Path(tempfile.mkdtemp(prefix='combined-originals-'))
    root=prepare_sources(input_root,work/'sources',output,source_config['source_supplement_sha256'])
    dirty,_,references,_=klados(root)
    summary=[];channels=[];controls=[];failures=[];timings=[]
    groups={'frontal':front,'posterior':back,'central_unchanged':central,
            'temporal_unchanged':temporal,'whole_cap':list(range(19))}
    for record in sorted(set(int(key[0]) for key in arrays['keys'])):
        directory=output/'klados'/f'record_{record:02d}';directory.mkdir(parents=True,exist_ok=True)
        began=time.perf_counter()
        values=preprocess(dirty[record,:,:2000]);eyes=preprocess(references[record,:,:2000])
        ridge=fit_ridge(values,eyes,.001)
        activity=np.mean((eyes/np.maximum(eyes.std(axis=-1,keepdims=True),1e-12))**2,axis=0)
        lower,upper=np.quantile(activity,[.3,.7])
        try:mwf=fit_mwf(values[back][:,activity>=upper],values[back][:,activity<=lower],rank=1,strength=1.)
        except (ValueError,RuntimeError,np.linalg.LinAlgError) as error:
            mwf=None;failures.append({'record':record,'method':'posterior_mwf_rank1','error':str(error),'fallback':'identity'})
        np.savez_compressed(directory/'ridge_calibration.npz',**vars(ridge))
        if mwf is not None:np.savez_compressed(directory/'mwf_calibration.npz',**vars(mwf))
        save_json(directory/'calibration.json',{'samples':[0,2000],'frontal_ridge_penalty':.001,
            'posterior_ridge_penalty':.001,'mwf_rank':1,'mwf_strength':1.,
            'mwf_masks':'calibration EOG activity bottom/top30%; no clean target used',
            'ridge_parameters_sha256':sha256(directory/'ridge_calibration.npz')})
        for position,key in enumerate(arrays['keys']):
            if int(key[0])!=record:continue
            start=int(key[1]);raw=arrays['raw'][position];target=arrays['clean'][position];eyes=arrays['eyes'][position]
            parent_name=f'windows/record_{record:02d}_start_{start}'
            with np.load(verify(parent_name+'/predictions.npz'),allow_pickle=False) as stream:
                np.testing.assert_array_equal(stream['raw'],raw[front]);np.testing.assert_array_equal(stream['clean'],target[front])
                frontal_vmd=stream['historical_vmd'].copy()
            with np.load(verify(parent_name+'/clean_input_predictions.npz'),allow_pickle=False) as stream:
                frontal_vmd_clean=stream['historical_vmd'].copy()
            frontal={'identity':raw[front],'historical_vmd':frontal_vmd,
                     'calibrated_ridge':(raw-ridge.artifact(raw,eyes))[front]}
            frontal_clean={'identity':target[front],'historical_vmd':frontal_vmd_clean,
                           'calibrated_ridge':(target-ridge.artifact(target,eyes))[front]}
            posterior={'identity':raw[back],'calibrated_ridge':(raw-ridge.artifact(raw,eyes))[back],
                       'mwf_rank1':raw[back]-(mwf.artifact(raw[back]) if mwf is not None else 0)}
            posterior_clean={'identity':target[back],'calibrated_ridge':(target-ridge.artifact(target,eyes))[back],
                             'mwf_rank1':target[back]-(mwf.artifact(target[back]) if mwf is not None else 0)}
            payload={'raw':raw,'clean':target,'heog':eyes[0],'veog':eyes[1],
                     'channel_names':np.asarray(names),'frontal_rows':np.asarray(front),'posterior_rows':np.asarray(back)}
            for frontal_method,posterior_method in itertools.product(frontal,posterior):
                method=frontal_method+'__'+posterior_method
                prediction=compose(raw,frontal[frontal_method],posterior[posterior_method],front,back)
                clean_prediction=compose(target,frontal_clean[frontal_method],posterior_clean[posterior_method],front,back)
                payload[method]=prediction;payload[method+'_clean_control']=clean_prediction
                for group,indices in groups.items():
                    score=paired(prediction[indices],target[indices]);baseline=paired(raw[indices],target[indices])
                    identity={'record':record,'start':start,'method':method,'frontal_method':frontal_method,
                              'posterior_method':posterior_method,'group':group}
                    summary.append({**identity,'pooled_window_snr_db':score['pooled_window_snr_db'],
                        'pooled_snr_improvement_db':score['pooled_window_snr_db']-baseline['pooled_window_snr_db'],
                        **{metric:float(np.mean(score[metric])) for metric in ['mse','rmse','rrmse_t','rrmse_s','pearson_cc']}})
                    details=correlation_rows(raw[indices],prediction[indices],eyes,[names[i] for i in indices],identity)
                    for channel,row in enumerate(details):
                        row.update({metric:float(score[metric][channel]) for metric in ['snr_out_db','mse','rmse','rrmse_t','rrmse_s','pearson_cc']})
                    channels.extend(details)
                    controls.append({**identity,**preservation(clean_prediction[indices],target[indices])})
            np.savez_compressed(directory/f'window_{start}_combined.npz',**payload)
            pd.DataFrame(summary).to_csv(output/'per_window_region.csv',index=False)
            pd.DataFrame(channels).to_csv(output/'per_channel.csv',index=False)
            pd.DataFrame(controls).to_csv(output/'clean_controls.csv',index=False)
        timings.append({'record':record,'runtime_s':time.perf_counter()-began,
                        'cached_frontal_VMD_cost_excluded':True})
        save_json(output/'method_failures.json',failures)
        print('COMBINED_RECORD_SAVED',record,flush=True)
    frame=pd.DataFrame(summary)
    metrics=['pooled_window_snr_db','pooled_snr_improvement_db','mse','rmse','rrmse_t','rrmse_s','pearson_cc']
    per_record=frame.groupby(['method','frontal_method','posterior_method','group','record'],as_index=False)[metrics].mean()
    per_record.to_csv(output/'per_record_region.csv',index=False)
    per_record.groupby(['method','frontal_method','posterior_method','group'],as_index=False)[metrics].mean().to_csv(output/'comparison.csv',index=False)
    pd.DataFrame(timings).to_csv(output/'runtime.csv',index=False)
    save_json(output/'protocol.json',{'frontal_names':[names[i] for i in front],
        'posterior_names':[names[i] for i in back],'unchanged_central':[names[i] for i in central],
        'unchanged_temporal':[names[i] for i in temporal],'row_binding_independently_verified':False,
        'frontal_parent_sha':'c19637465e5a5c29991c2a988f25e6527a156f09','test_cache_sha256':sha256(cache),
        'central_temporal_unchanged_asserted':True,'single_subtraction':True,
        'whole_cap_snr':'10log10(sum(clean**2)/sum((clean-prediction)**2)); pooled per window, equal record weight',
        'regional_db_added':False,'frontal_to_posterior_message_passing':False,
        'SGEYESUB_Klados':'ineligible: six-class calibration labels unavailable',
        'native_OSF_reconstruction_SNR':'unavailable without clean targets',
        'LEMON_used':False,'new_AutoVMD_search':False,'production_qualified':False})
    save_json(output/'completion.json',{'complete':True,'records':len(per_record.record.unique()),
        'combinations':9,'method_failures':len(failures),'production_qualified':False})


def report(output):
    import matplotlib.pyplot as plt
    output=Path(output);table=pd.read_csv(output/'comparison.csv')
    records=pd.read_csv(output/'per_record_region.csv');clean=pd.read_csv(output/'clean_controls.csv')
    clean=clean.groupby(['method','group','record'],as_index=False).mean(numeric_only=True)
    gates=[];uncertainty=[]
    for group in table.group.unique():
        control=records[(records.group==group)&(records.method=='calibrated_ridge__calibrated_ridge')].set_index('record')
        for method in table.method.unique():
            selected=records[(records.group==group)&(records.method==method)].set_index('record')
            keep=clean[(clean.group==group)&(clean.method==method)]
            row={'group':group,'method':method,'worst_record_clean_change':float(keep.relative_change.max()),
                 'worst_record_alpha_db':float(keep.alpha_error_db.max()),'worst_record_beta_db':float(keep.beta_error_db.max()),
                 'worst_record_covariance':float(keep.covariance_relative_error.max()),
                 'worst_record_cc_deterioration_vs_ridge':float((control.pearson_cc-selected.pearson_cc).max())}
            row['paired_preservation_pass']=bool(row['worst_record_clean_change']<=.01 and
                row['worst_record_alpha_db']<=.5 and row['worst_record_beta_db']<=.5 and
                row['worst_record_covariance']<=.02 and row['worst_record_cc_deterioration_vs_ridge']<=.005)
            gates.append(row)
            if group=='whole_cap' and method.startswith('historical_vmd__'):
                frontal_only=records[(records.group==group)&(records.method=='historical_vmd__identity')].set_index('record')
                uncertainty.append({'method':method,'comparator':'historical_vmd__identity',
                    **bootstrap_difference(selected.pooled_window_snr_db,frontal_only.pooled_window_snr_db)})
    pd.DataFrame(gates).to_csv(output/'preservation_gates.csv',index=False)
    pd.DataFrame(uncertainty).to_csv(output/'combined_snr_uncertainty.csv',index=False)
    paths=[];directory=output/'figures';directory.mkdir(exist_ok=True)
    def save(figure,name):
        figure.tight_layout()
        for extension in ['png','pdf']:figure.savefig(directory/(name+'.'+extension),dpi=150,bbox_inches='tight')
        paths.append(str(directory/(name+'.png')));plt.close(figure)
    for metric,label in [('pooled_window_snr_db','Output SNR (dB)'),('rmse','RMSE'),('pearson_cc','CC with clean EEG')]:
        figure,axes=plt.subplots(1,3,figsize=(19,6))
        for axis,group in zip(axes,['frontal','posterior','whole_cap']):
            selected=table[table.group==group];axis.barh(selected.method,selected[metric]);axis.set(title=group,xlabel=label);axis.grid(axis='x',alpha=.2)
        save(figure,metric)
    figure,axis=plt.subplots(figsize=(12,5))
    for method in ['identity__identity','historical_vmd__identity','historical_vmd__calibrated_ridge','historical_vmd__mwf_rank1']:
        selected=records[(records.group=='whole_cap')&(records.method==method)]
        axis.plot(selected.record,selected.pooled_window_snr_db,'o-',label=method)
    axis.set(xlabel='Archived record ID',ylabel='Whole-cap output SNR (dB)',title='Every record retained; central/temporal unchanged');axis.legend();save(figure,'record_snr')
    for metric,label,limit in [('relative_change','Clean-input relative RMS modification (%)',1),
                               ('alpha_error_db','Clean alpha error (dB)',.5),('beta_error_db','Clean beta error (dB)',.5),
                               ('covariance_relative_error','Relative clean covariance error',.02)]:
        figure,axis=plt.subplots(figsize=(12,5));selected=clean[clean.group=='whole_cap'].groupby('method')[metric].max()
        axis.barh(selected.index,selected*(100 if metric=='relative_change' else 1));axis.axvline(limit,color='black',linestyle='--')
        axis.set(xlabel=label,title='Worst-record mean, whole cap; fixed engineering limit');save(figure,metric)
    text='# Combined frontal/posterior Klados experiment\n\n'
    text+='Historical frontal VMD+EOG, calibrated frontal ridge and identity crossed with posterior identity/ridge/rank1 MWF. '
    text+='Only posterior channels receive the new posterior correction. Central C3/C4/Cz and temporal T3/T4/T5/T6 remain unchanged. '
    text+='This is composition, not frontal-to-back message passing. No regional dB values are added.\n\n'
    text+=table[table.group.isin(['frontal','posterior','whole_cap'])].round(6).to_markdown(index=False)+'\n\n'
    text+=pd.DataFrame(gates).round(6).to_markdown(index=False)+'\n\n'
    text+=pd.DataFrame(uncertainty).round(6).to_markdown(index=False)+'\n\n'
    text+='Paired-only, reused development records with unknown participant identities and paper-list row binding unverified. '
    text+='Calibration fits use the independently preprocessed first2000samples, no clean target. MWF uses fixed top/bottom30% calibration EOG-activity masks, not proven artifact-free resting EEG. '
    text+='All actual scoring arrays are the same saved full19-channel cache. Historical frontal predictions and clean-input controls are verified parent artifacts. '
    text+='No VMD is recomputed or applied to posterior channels. Runtime excludes cached frontal work. '
    text+='Native OSF has no reconstruction SNR; native regional qualification remains separate. No production qualification follows from this experiment.\n'
    (output/'RESULTS.md').write_text(text,encoding='utf-8')
    return paths
