"""Frontal-only scoring of full-montage author spatial methods."""
from pathlib import Path
import json
import tempfile
import time
import numpy as np
import pandas as pd
from .data import save_json, prepare_sources, osf_trials, osf_sessions, calibration_split,klados,preprocess,sha256
from .osf_reader import read_osf
from .spatial_author import (prepare_backend, fit_sgeyesub, fit_regica,
                            predict_regica, calibration_sample_labels)
from .pretrained_review import verified_table_parent
from .central import correlation_rows
from .metrics import preservation


def run_paired(root,input_root,source_root,output,backend,failures):
    """Score original seven-row frontal caches with full19-channel REGICA fit."""
    if not Path('/kaggle/input').exists():raise RuntimeError('Numerical work runs on Kaggle')
    from .metrics import paired
    from .methods import fit_ridge
    config=json.loads((source_root/'configs/frontal_six.json').read_text())
    montage=json.loads((source_root/'docs/provenance/klados_montage.json').read_text())
    indices=montage['frontal_indices'];names=montage['frontal_names']
    parents=list(Path(input_root).rglob('snr_target_protocol.json'))
    if len(parents)!=1:raise ValueError('Original19-channel cache parent is not unique')
    cache_path=parents[0].parent/'test_vmd_cache.npz'
    if sha256(cache_path)!=config['historical_test_cache_sha256']:raise ValueError('Paired cache identity differs')
    with np.load(cache_path,allow_pickle=False) as archive:
        arrays={key:archive[key].copy() for key in ['raw','clean','eyes','keys']}
    verify=verified_table_parent(input_root,'frontal-six-full-001','c19637465e5a5c29991c2a988f25e6527a156f09')
    old_metrics=pd.read_csv(verify('per_channel_window.csv'))
    dirty,_,references,_=klados(root)
    rows=[];controls=[]
    for record in sorted(set(int(key[0]) for key in arrays['keys'])):
        directory=output/'klados'/f'record_{record:02d}';directory.mkdir(parents=True,exist_ok=True)
        values=preprocess(dirty[record,:,:2000]);eyes=preprocess(references[record,:,:2000])
        ridge=fit_ridge(values,eyes,.001)
        try:model=fit_regica(values,backend,directory/'regica_fit')
        except (RuntimeError,ValueError,np.linalg.LinAlgError) as error:
            model=None;failures.append({'record':record,'dataset':'Klados','method':'regica',
                'phase':'calibration','error':str(error),'fallback':'identity'})
        for position,key in enumerate(arrays['keys']):
            if int(key[0])!=record:continue
            start=int(key[1]);raw_full=arrays['raw'][position];clean_full=arrays['clean'][position]
            eyes=arrays['eyes'][position]
            old_path=f'windows/record_{record:02d}_start_{start}/predictions.npz'
            with np.load(verify(old_path),allow_pickle=False) as archive:old_raw=archive['raw']
            np.testing.assert_array_equal(raw_full[indices],old_raw)
            stored={'raw':old_raw,'clean':clean_full[indices],'eyes':eyes,
                'channel_names':np.asarray(names),'source_row_indices':np.asarray(indices)}
            for method in ['regica','direct_calibrated_ridge']:
                try:
                    if method=='regica' and model is not None:
                        prediction=predict_regica(raw_full,eyes,model,backend,directory/f'window_{start}')[indices]
                        clean_prediction=predict_regica(clean_full,eyes,model,backend,directory/f'clean_{start}')[indices]
                    elif method=='direct_calibrated_ridge':
                        prediction=(raw_full-ridge.artifact(raw_full,eyes))[indices]
                        clean_prediction=(clean_full-ridge.artifact(clean_full,eyes))[indices]
                    else:prediction=old_raw.copy();clean_prediction=clean_full[indices].copy()
                except (RuntimeError,ValueError,np.linalg.LinAlgError) as error:
                    prediction=old_raw.copy();clean_prediction=clean_full[indices].copy()
                    failures.append({'dataset':'Klados','record':record,'start':start,'method':method,
                        'phase':'inference','error':str(error),'fallback':'identity'})
                scores=paired(prediction,clean_full[indices]);baseline=paired(old_raw,clean_full[indices])
                details=correlation_rows(old_raw,prediction,eyes,names,{'record':record,'start':start,'method':method})
                for channel,row in enumerate(details):
                    row.update({metric:float(scores[metric][channel]) for metric in
                        ['mse','rmse','mae','rrmse_t','rrmse_s','pearson_cc','snr_out_db']})
                    row['snr_improvement_db']=float(scores['snr_out_db'][channel]-baseline['snr_out_db'][channel])
                    row.update(pooled_window_snr_db=scores['pooled_window_snr_db'],
                        pooled_snr_improvement_db=scores['pooled_window_snr_db']-baseline['pooled_window_snr_db'])
                rows.extend(details)
                controls.append({'record':record,'start':start,'method':method,
                                 **preservation(clean_prediction,clean_full[indices])})
                stored[method]=prediction.astype(np.float32)
                stored[method+'_clean_control']=clean_prediction.astype(np.float32)
            np.savez_compressed(directory/f'window_{start}_frontal.npz',**stored)
            pd.DataFrame(rows).to_csv(output/'new_klados_per_channel.csv',index=False)
            pd.DataFrame(controls).to_csv(output/'new_klados_clean_controls.csv',index=False)
            save_json(output/'method_failures.json',failures)
        print('SPATIAL_KLADOS_RECORD_SAVED',record,flush=True)
    pd.concat([old_metrics,pd.DataFrame(rows)],ignore_index=True).to_csv(output/'klados_all_per_channel.csv',index=False)
    frame=pd.DataFrame(rows)
    metrics=['mse','rmse','mae','rrmse_t','rrmse_s','pearson_cc','snr_out_db','snr_improvement_db','pooled_window_snr_db','pooled_snr_improvement_db']
    records=frame.groupby(['method','record'],as_index=False)[metrics].mean()
    records.groupby('method',as_index=False)[metrics].mean().to_csv(output/'new_klados_summary.csv',index=False)
    old_table=pd.read_csv(verify('comparison.csv'))
    pd.concat([old_table,pd.read_csv(output/'new_klados_summary.csv')],ignore_index=True).to_csv(output/'klados_comparison.csv',index=False)
    save_json(output/'paired_protocol.json',{'full_fit_channels':19,'scored_channels':names,
        'same_scoring_inputs_as_parent':True,'original_test_cache_sha256':sha256(cache_path),
        'regica_calibration':'original EEG first2000 independently preprocessed samples',
        'adaptation':'frozen calibration ICA/FD, original SRLS reset for each independent scoring/clean-control window',
        'clean_targets_used_for_fit':False,'SGEYESUB':'not eligible: no verified six-class sample annotations',
        'comparison_limits':'identical scoring arrays; methods retain different training/calibration information access'})


def readiness(input_root, source_root, output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    source_root=Path(source_root)
    if not Path('/kaggle/input').exists():raise RuntimeError('Numerical work runs on Kaggle')
    backend=prepare_backend(source_root,output/'author_sources')
    config=json.loads((source_root/'configs/frontal_osf.json').read_text())
    work=Path(tempfile.mkdtemp(prefix='frontal-spatial-originals-'))
    root=prepare_sources(input_root,work/'sources',output,config['source_supplement_sha256'])
    partition=json.loads((source_root/'docs/provenance/osf_partition.json').read_text())['partitions']
    path=osf_sessions(root,partition,1)[0]
    trials,metadata=osf_trials(path)
    calibration,scoring,policy=calibration_split(trials)
    item=read_osf(path)
    labels=calibration_sample_labels(item,calibration)
    values=np.concatenate([trial['eeg'] for trial in calibration],axis=-1)
    counts={str(label):int(np.count_nonzero(labels==label)) for label in range(7)}
    save_json(output/'native_calibration_eligibility.json',{'record':metadata['record'],
        'EEG_channels':metadata['channel_names'],'full_montage_channels':len(values),
        'sample_class_counts':counts,'policy':policy,'frontal_names':config['common_frontal_names'],
        'calibration_trial_ids':[trial['index'] for trial in calibration],
        'scoring_trial_ids':[trial['index'] for trial in scoring],
        'clean_target_used':False,'sample_labels_filtered':False})
    began=time.perf_counter()
    C=fit_sgeyesub(values,labels,backend,output/'sgeyesub_readiness')
    indices=[metadata['channel_names'].index(name) for name in config['common_frontal_names']]
    trial=scoring[0]
    prediction=C@trial['eeg']
    if not np.isfinite(prediction).all():raise ValueError('Nonfinite author SGEYESUB prediction')
    sge_runtime=time.perf_counter()-began
    # Verify the original ICA+FD+SRLS path on deterministic small numerical data
    # before the expensive 58-channel calibration fit in the full campaign.
    rng=np.random.default_rng(42)
    time_axis=np.arange(2400)/200
    sources=np.stack([np.sin(2*np.pi*time_axis),np.sign(np.sin(6*np.pi*time_axis)),
                      rng.laplace(size=2400)])
    mixture=np.array([[1.,.2,.3],[.4,1.,.2],[.2,.3,1.]])@sources
    model=fit_regica(mixture[:,:1600],backend,output/'regica_readiness_fit')
    clean=predict_regica(mixture[:,1600:],sources[:2,1600:],model,backend,output/'regica_readiness_apply')
    if clean.shape!=mixture[:,1600:].shape or not np.isfinite(clean).all():
        raise ValueError('Author REG-ICA runtime contract failed')
    np.savez_compressed(output/'native_readiness_prediction.npz',raw=trial['eeg'][indices],
        sgeyesub=prediction[indices],heog=trial['evaluation_eyes'][0],veog=trial['evaluation_eyes'][1],
        channel_names=np.asarray(config['common_frontal_names']))
    save_json(output/'readiness.json',{'complete':True,'sgeyesub_full_montage_fit_passed':True,
        'regica_author_routines_fixture_passed':True,'native_record':metadata['record'],
        'sgeyesub_fit_and_apply_s':sge_runtime,'scored_region':'frontal only',
        'no_central_posterior_results':True,'production_qualified':False})
    print('FRONTAL_SPATIAL_READINESS_PASSED',metadata['record'],flush=True)


def run_native(input_root,source_root,output):
    if not Path('/kaggle/input').exists():raise RuntimeError('Numerical work runs on Kaggle')
    output=Path(output);source_root=Path(source_root)
    backend=prepare_backend(source_root,output/'author_sources')
    config=json.loads((source_root/'configs/frontal_osf.json').read_text())
    work=Path(tempfile.mkdtemp(prefix='frontal-spatial-originals-'))
    root=prepare_sources(input_root,work/'sources',output,config['source_supplement_sha256'])
    verify=verified_table_parent(input_root,'frontal-osf-correlations-001','b9df0e65e0b7135aae45f9614439272bda88a8f3')
    old=Path(verify('osf_frontal_before_after_per_channel.csv')).parent
    previous=pd.read_csv(old/'osf_frontal_before_after_per_channel.csv')
    # Existing raw/prediction payloads are verified individually before use.
    partition=json.loads((source_root/'docs/provenance/osf_partition.json').read_text())['partitions']
    rows=[];rest_rows=[];failures=[];timings=[]
    run_paired(root,input_root,source_root,output,backend,failures)
    for path in osf_sessions(root,partition,4):
        trials,metadata=osf_trials(path);calibration,scoring,policy=calibration_split(trials)
        labels=calibration_sample_labels(read_osf(path),calibration)
        values=np.concatenate([trial['eeg'] for trial in calibration],axis=-1)
        directory=output/'osf'/metadata['record'];directory.mkdir(parents=True,exist_ok=True)
        models={}
        for method,function in [('sgeyesub',lambda:fit_sgeyesub(values,labels,backend,directory/'sgeyesub_fit')),
                                ('regica',lambda:fit_regica(values,backend,directory/'regica_fit'))]:
            began=time.perf_counter()
            try:models[method]=function()
            except (ValueError,RuntimeError,np.linalg.LinAlgError) as error:
                models[method]=None
                failures.append({'record':metadata['record'],'method':method,'phase':'calibration',
                    'error':str(error),'fallback':'identity, all cases retained'})
            timings.append({'record':metadata['record'],'method':method,'phase':'fit','runtime_s':time.perf_counter()-began})
            print('SPATIAL_FIT_SAVED',metadata['record'],method,models[method] is not None,flush=True)
        selected=previous[(previous.record==metadata['record'])&(previous.method=='identity')]
        trial_ids=set(int(value) for value in selected.trial)
        names=config['common_frontal_names'];indices=[metadata['channel_names'].index(name) for name in names]
        save_json(directory/'calibration.json',{**metadata,'sample_class_counts':{str(i):int(np.count_nonzero(labels==i)) for i in range(7)},
            'full_montage_fit':True,'scored_channel_names':names,'policy':policy,
            'calibration_trials':[trial['index'] for trial in calibration],'scoring_trials':sorted(trial_ids),
            'REGICA_adaptation':'calibration-frozen original Infomax/FD; original SRLS reset per scoring trial; first2 outputs preserved',
            'no_clean_target_fitting':True})
        for trial in scoring:
            if trial['index'] not in trial_ids:continue
            parent_name=f'osf/{metadata["record"]}/trial_{trial["index"]:03d}_predictions.npz'
            with np.load(verify(parent_name),allow_pickle=False) as archive:
                old_raw=archive['raw'];evaluation=np.stack([archive['heog'],archive['veog']])
            np.testing.assert_allclose(trial['eeg'][indices],old_raw,rtol=0,atol=0,
                err_msg='Native frontal scoring inputs differ from completed parent')
            np.testing.assert_array_equal(trial['evaluation_eyes'],evaluation)
            stored={'raw':old_raw,'heog':evaluation[0],'veog':evaluation[1],'channel_names':np.asarray(names)}
            for method,model in models.items():
                began=time.perf_counter()
                try:
                    if model is None:prediction=old_raw.copy()
                    elif method=='sgeyesub':prediction=(model@trial['eeg'])[indices]
                    else:prediction=predict_regica(trial['eeg'],trial['eyes'],model,backend,
                        directory/f'regica_trial_{trial["index"]:03d}')[indices]
                except (ValueError,RuntimeError,np.linalg.LinAlgError) as error:
                    prediction=old_raw.copy()
                    failures.append({'record':metadata['record'],'trial':trial['index'],'method':method,
                                     'phase':'inference','error':str(error),'fallback':'identity'})
                from .frontal_correlations import CONDITIONS
                identity={'record':metadata['record'],'participant':metadata['participant'],
                    'trial':trial['index'],'condition':CONDITIONS[trial['condition']],'method':method}
                rows.extend(correlation_rows(old_raw,prediction,evaluation,names,identity))
                if trial['condition']==1:rest_rows.append({**identity,**preservation(prediction,old_raw)})
                stored[method]=prediction.astype(np.float32)
                timings.append({**identity,'phase':'inference','runtime_s':time.perf_counter()-began})
            np.savez_compressed(directory/f'trial_{trial["index"]:03d}_frontal.npz',**stored)
            pd.DataFrame(rows).to_csv(output/'new_osf_per_channel.csv',index=False)
            pd.DataFrame(rest_rows).to_csv(output/'new_osf_rest_preservation.csv',index=False)
            save_json(output/'method_failures.json',failures)
            print('SPATIAL_FRONTAL_TRIAL_SAVED',metadata['record'],trial['index'],flush=True)
    frame=pd.concat([previous,pd.DataFrame(rows)],ignore_index=True)
    metric_names=[name for name in frame if '_cc_' in name]
    participants=frame.groupby(['method','participant','condition'],as_index=False)[metric_names].mean()
    participants.to_csv(output/'osf_participant_correlations.csv',index=False)
    participants.groupby(['method','condition'],as_index=False)[metric_names].mean().to_csv(output/'osf_comparison.csv',index=False)
    prior_rest=pd.read_csv(verify('osf_frontal_rest_preservation.csv'))
    rest=pd.concat([prior_rest,pd.DataFrame(rest_rows)],ignore_index=True)
    rest.to_csv(output/'osf_rest_all_trials.csv',index=False)
    participants=rest.groupby(['method','participant'],as_index=False).mean(numeric_only=True)
    participants.groupby('method',as_index=False).mean(numeric_only=True).to_csv(output/'osf_rest_summary.csv',index=False)
    participants.groupby('method',as_index=False).max(numeric_only=True).to_csv(output/'osf_rest_worst_participant.csv',index=False)
    pd.DataFrame(timings).to_csv(output/'runtime.csv',index=False)
    save_json(output/'completion.json',{'complete':True,'scored_regions':['frontal'],
        'full_montage_fit':True,'native_scoring_inputs_exactly_match_parent':True,
        'new_methods':['sgeyesub','regica'],'method_failures':len(failures),
        'klados_SGEYESUB':'not eligible: original six-class sample labels unavailable',
        'central_posterior_scored':False,'LEMON_used':False,'neural_training':False,'production_qualified':False})
