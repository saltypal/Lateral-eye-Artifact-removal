"""Frontal-only scoring of full-montage author spatial methods."""
from pathlib import Path
import json
import tempfile
import time
import shutil
import numpy as np
import pandas as pd
from .data import save_json, prepare_sources, osf_trials, osf_sessions, calibration_split,klados,preprocess,sha256
from .osf_reader import read_osf
from .spatial_author import (prepare_backend, fit_sgeyesub, fit_regica,
                            predict_regica, calibration_sample_labels)
from .pretrained_review import verified_table_parent
from .central import correlation_rows
from .metrics import preservation


def canonical_scoring_input(full,indices,archived):
    """Use exact archived scored rows; reject differences larger than one float32 ULP.

    Different CPU/library execution can round a filtered float32 value to its
    adjacent representation. This is an input-identity check, not a metric or
    scientific acceptance tolerance. All scored model inputs use archived bits.
    """
    current=np.asarray(full[indices],np.float32)
    archived=np.asarray(archived,np.float32)
    if current.shape!=archived.shape or not np.isfinite(current).all() or not np.isfinite(archived).all():
        raise ValueError('Scoring identity shape/finiteness differs')
    np.testing.assert_array_max_ulp(current,archived,maxulp=1)
    result=full.copy();result[indices]=archived
    np.testing.assert_array_equal(result[indices],archived)
    return result,{'different_float32_values':int(np.count_nonzero(current!=archived)),
                   'max_absolute_rounding_difference':float(np.max(np.abs(current.astype(float)-archived))),
                   'maximum_permitted_float32_ulp':1,'actual_scored_frontal_values':'exact archived bytes'}


def restore_failed_parent(input_root,output):
    """Restore declared failed run only; verify every copied file before reuse."""
    matches=[]
    for path in Path(input_root).rglob('artifact_manifest.json'):
        manifest=json.loads(path.read_text())
        if manifest.get('run_id')=='frontal-spatial-full-001':matches.append((path,manifest))
    if not matches:
        save_json(Path(output)/'recovery_parent.json',{'available':False,
            'requested_parent':'frontal-spatial-full-001/e8d0021',
            'action':'fresh recomputation; no cached outputs or fits reused',
            'scientific_acceptance_thresholds_changed':False})
        print('SPATIAL_RECOVERY_UNAVAILABLE_FRESH_RECOMPUTATION',flush=True)
        return
    if len(matches)!=1:raise ValueError('Failed parent must be unique')
    path,manifest=matches[0]
    if manifest['git_sha']!='e8d0021bf1c2e67b28babb370411b1e7590fa868' or manifest['complete']:
        raise ValueError('Unexpected recovery parent identity/state')
    copied=[]
    names={'new_klados_per_channel.csv','new_klados_clean_controls.csv','klados_all_per_channel.csv',
           'new_klados_summary.csv','klados_comparison.csv','paired_protocol.json',
           'new_osf_per_channel.csv','new_osf_rest_preservation.csv','method_failures.json'}
    for entry in manifest['files']:
        name=entry['path']
        if not (name.startswith(('klados/','osf/')) or name in names):continue
        source=path.parent/name;destination=Path(output)/name
        if not destination.resolve().is_relative_to(Path(output).resolve()):raise ValueError('Invalid recovery path')
        if source.stat().st_size!=entry['bytes'] or sha256(source)!=entry['sha256']:
            raise ValueError('Failed parent payload hash differs: '+name)
        destination.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,destination);copied.append(name)
    save_json(Path(output)/'recovery_parent.json',{'run_id':manifest['run_id'],'git_sha':manifest['git_sha'],
        'parent_completed':False,'copied_verified_files':copied,'calibration_model_reuse_requires_input_identity_check':True,
        'repair':'one-float32-ULP preprocessing identity check; actual scored frontal rows use archived bytes',
        'scientific_acceptance_thresholds_changed':False})


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
    if (output/'paired_protocol.json').is_file():
        protocol=json.loads((output/'paired_protocol.json').read_text())
        if protocol['original_test_cache_sha256']!=sha256(cache_path) or protocol['scored_channels']!=names:
            raise ValueError('Paired recovery protocol differs')
        expected={(int(key[0]),int(key[1])) for key in arrays['keys']}
        recovered=pd.read_csv(output/'new_klados_per_channel.csv')
        for method in ['regica','direct_calibrated_ridge']:
            selected=recovered[recovered.method==method]
            if set(zip(selected.record,selected.start))!=expected or len(selected)!=len(expected)*len(names):
                raise ValueError('Paired recovery incomplete')
        print('SPATIAL_KLADOS_RECOVERED',len(expected),'exact parent windows',flush=True)
        return
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
    restore_failed_parent(input_root,output)
    backend=prepare_backend(source_root,output/'author_sources')
    config=json.loads((source_root/'configs/frontal_osf.json').read_text())
    work=Path(tempfile.mkdtemp(prefix='frontal-spatial-originals-'))
    root=prepare_sources(input_root,work/'sources',output,config['source_supplement_sha256'])
    verify=verified_table_parent(input_root,'frontal-osf-correlations-001','b9df0e65e0b7135aae45f9614439272bda88a8f3')
    old=Path(verify('osf_frontal_before_after_per_channel.csv')).parent
    previous=pd.read_csv(old/'osf_frontal_before_after_per_channel.csv')
    # Existing raw/prediction payloads are verified individually before use.
    partition=json.loads((source_root/'docs/provenance/osf_partition.json').read_text())['partitions']
    rows=pd.read_csv(output/'new_osf_per_channel.csv').to_dict('records') if (output/'new_osf_per_channel.csv').exists() else []
    rest_rows=pd.read_csv(output/'new_osf_rest_preservation.csv').to_dict('records') if (output/'new_osf_rest_preservation.csv').exists() else []
    failures=json.loads((output/'method_failures.json').read_text()) if (output/'method_failures.json').exists() else []
    timings=[];rounding=[]
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
            try:
                fit_directory=directory/(method+'_fit')
                if (fit_directory/'model.mat').is_file():
                    from scipy.io import loadmat
                    saved=loadmat(fit_directory/'input.mat')
                    np.testing.assert_array_max_ulp(np.asarray(saved['X'],np.float32),values.astype(np.float32),maxulp=1)
                    if method=='sgeyesub':np.testing.assert_array_equal(saved['labels'].reshape(-1),labels)
                    restored=loadmat(fit_directory/'model.mat')
                    models[method]=restored['C'] if method=='sgeyesub' else restored
                    phase='restore_verified_calibration_model'
                else:models[method]=function();phase='fit'
            except (ValueError,RuntimeError,np.linalg.LinAlgError) as error:
                models[method]=None
                failures.append({'record':metadata['record'],'method':method,'phase':'calibration',
                    'error':str(error),'fallback':'identity, all cases retained'})
                phase='failed_fit'
            timings.append({'record':metadata['record'],'method':method,'phase':phase,'runtime_s':time.perf_counter()-began})
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
            scoring_values,detail=canonical_scoring_input(trial['eeg'],indices,old_raw)
            rounding.append({'record':metadata['record'],'trial':trial['index'],**detail})
            save_json(output/'scoring_input_identity.json',rounding)
            np.testing.assert_array_equal(trial['evaluation_eyes'],evaluation)
            previous_rows=[row for row in rows if row['record']==metadata['record'] and int(row['trial'])==trial['index']]
            if previous_rows:
                if len(previous_rows)!=2*len(names) or {row['method'] for row in previous_rows}!={'sgeyesub','regica'}:
                    raise ValueError('Incomplete recovered trial')
                with np.load(directory/f'trial_{trial["index"]:03d}_frontal.npz',allow_pickle=False) as archive:
                    np.testing.assert_array_equal(archive['raw'],old_raw)
                print('SPATIAL_TRIAL_RECOVERED',metadata['record'],trial['index'],flush=True)
                continue
            stored={'raw':old_raw,'heog':evaluation[0],'veog':evaluation[1],'channel_names':np.asarray(names)}
            for method,model in models.items():
                began=time.perf_counter()
                try:
                    if model is None:prediction=old_raw.copy()
                    elif method=='sgeyesub':prediction=(model@scoring_values)[indices]
                    else:prediction=predict_regica(scoring_values,trial['eyes'],model,backend,
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
