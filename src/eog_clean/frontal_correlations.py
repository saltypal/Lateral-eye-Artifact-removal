"""Explicit paired-clean correlations and frontal-only native OSF transfer.

No neural training here. Original references remain unchanged during evaluation.
Native OSF has no paired clean EEG; resting-input agreement is only a proxy.
"""
import json
from pathlib import Path
import tempfile
import time
import traceback
import numpy as np
import pandas as pd
import torch
from .data import save_json,sha256,prepare_sources,osf_sessions,osf_trials,calibration_split
from .frontal import window_starts,overlap_add
from .frontal_benchmark import METHODS,historical_predict
from .metrics import correlations,preservation
from .methods import fit_mwf,window_regression
from .paper_methods import svmd_fcm,vme_gmetv
from .vmd import rolling_decompose
from .historical_model import EOGContextGainStudent
from .author_models import fetch_author_source,EEGOAR,bandroute_model,bandroute_predict,pack_frontal_montage

CONDITIONS={1:'rest',2:'lateral',3:'vertical',4:'blink'}


def verified_parent(input_root,config):
    matches=[]
    for path in Path(input_root).rglob('artifact_manifest.json'):
        manifest=json.loads(path.read_text())
        if manifest.get('run_id')==config['parent_run_id']:
            matches.append((path,manifest))
    if len(matches)!=1:
        raise ValueError('Need one pinned completed six-method parent')
    path,manifest=matches[0]
    if manifest['git_sha']!=config['parent_git_sha'] or not manifest['complete']:
        raise ValueError('Six-method parent identity/completion differs')
    root=path.parent
    entries={row['path']:row for row in manifest['files']}
    def verify(name):
        target=root/name
        entry=entries[name]
        if target.stat().st_size!=entry['bytes'] or sha256(target)!=entry['sha256']:
            raise ValueError('Parent artifact checksum differs: '+name)
        return target
    for name in ['per_channel_window.csv','comparison.csv','eegoar_best.pt','bandroute_best.pt','data_contract.json']:
        verify(name)
    return root,verify


def _cc(first,second):
    # Independent NumPy recomputation, separate from the shared metric helper.
    return float(np.corrcoef(first,second)[0,1])


def klados_clean_audit(parent,verify,source_root,output):
    labels=json.loads((Path(source_root)/'docs/provenance/klados_montage.json').read_text())
    names=labels['frontal_names']
    published=pd.read_csv(parent/'per_channel_window.csv').set_index(['method','record','start','channel'])
    rows=[]
    for directory in sorted((parent/'windows').iterdir()):
        key=json.loads((directory/'completion.json').read_text())
        path=verify(str((directory/'predictions.npz').relative_to(parent)).replace('\\','/'))
        with np.load(path,allow_pickle=False) as archive:
            predictions={name:archive[name] for name in archive.files}
        raw,target,eyes=predictions['raw'],predictions['clean'],predictions['eyes']
        if raw.shape!=(7,1024) or target.shape!=raw.shape:
            raise ValueError('Frontal NPZ axes differ from paper-listed seven rows')
        destination=output/'klados_labelled'/directory.name
        destination.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(destination/'predictions.npz',**predictions,
            channel_names=np.asarray(names),source_row_indices=np.asarray(labels['frontal_indices']),
            sampling_rate=np.array(200),record=np.array(key['record']),start=np.array(key['start']),
            parent_prediction_sha256=np.array(sha256(path)))
        mode_path=verify(str((directory/'historical_modes.npz').relative_to(parent)).replace('\\','/'))
        with np.load(mode_path,allow_pickle=False) as archive:
            modes={name:archive[name] for name in archive.files}
        np.savez_compressed(destination/'historical_modes.npz',**modes,
            channel_names=np.asarray(names),source_row_indices=np.asarray(labels['frontal_indices']),
            mode_indices=np.array([1,2,3]),sampling_rate=np.array(200))
        for method,prediction in predictions.items():
            if method in ('raw','clean','eyes'):continue
            for channel,name in enumerate(names):
                before,after=_cc(raw[channel],target[channel]),_cc(prediction[channel],target[channel])
                previous=published.loc[(method,key['record'],key['start'],name),'pearson_cc']
                if not np.isclose(after,previous,atol=5e-6,rtol=5e-6,equal_nan=True):
                    raise ValueError('Saved prediction/clean CC differs from original metrics')
                detail={'method':method,'record':key['record'],'start':key['start'],'channel':name,
                        'source_row_index':labels['frontal_indices'][channel],
                        'clean_cc_before':before,'clean_cc_after':after,'clean_cc_change':after-before}
                for axis,label in enumerate(['heog','veog']):
                    a,b=_cc(raw[channel],eyes[axis]),_cc(prediction[channel],eyes[axis])
                    detail.update({label+'_cc_before':a,label+'_cc_after':b,
                                   label+'_abs_cc_before':abs(a),label+'_abs_cc_after':abs(b)})
                rows.append(detail)
    frame=pd.DataFrame(rows)
    frame.to_csv(output/'klados_clean_and_eog_correlations.csv',index=False)
    numeric=[name for name in frame.columns if '_cc_' in name]
    records=frame.groupby(['method','record'],as_index=False)[numeric].mean()
    records.to_csv(output/'klados_correlation_per_record.csv',index=False)
    records.groupby('method',as_index=False)[numeric].mean().to_csv(output/'klados_correlation_summary.csv',index=False)
    save_json(output/'klados_label_binding.json',{**labels,'all_prediction_and_mode_npz_contain_names':True,
              'saved_prediction_cc_independently_recomputed_and_verified':True,
              'clean_truth':'paired Klados Pure_Data-derived archived target'})
    print('KLADOS_CLEAN_CC_AND_NPZ_LABELS_VERIFIED',len(frame),flush=True)


def load_models(input_root,parent,source_root,output,config):
    specs=json.loads((Path(source_root)/'docs/provenance/author_sources.json').read_text())
    temporary=Path(tempfile.mkdtemp(prefix='frontal-transfer-authors-'))
    sources={name:fetch_author_source(spec,temporary/name) for name,spec in specs.items()}
    save_json(output/'author_sources.json',specs)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    names=json.loads((Path(source_root)/'docs/provenance/klados_montage.json').read_text())['frontal_names']
    eegoar=EEGOAR(sources['eegoar'],names,device)
    eegoar.model.load_state_dict(torch.load(parent/'eegoar_best.pt',map_location=device,weights_only=True)['model'])
    eegoar.model.eval();eegoar.finetuned=True
    bandroute,_=bandroute_model(sources['bandroute'],device)
    bandroute.load_state_dict(torch.load(parent/'bandroute_best.pt',map_location=device,weights_only=True)['model'])
    bandroute.eval()
    candidates=list(Path(input_root).rglob('snr_target_protocol.json'))
    if len(candidates)!=1:raise ValueError('Historical checkpoint parent not unique')
    checkpoint=candidates[0].parent/'eog_vmd_context_student/best_safe.pt'
    if sha256(checkpoint)!=config['historical_checkpoint_sha256']:
        raise ValueError('Historical model checksum differs')
    historical=EOGContextGainStudent().eval()
    historical.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True)['model'])
    historical.inference_gate_mode,historical.inference_gate_offset='hard',0.
    save_json(output/'frozen_model_transfer.json',{'neural_training':False,
        'eegoar_checkpoint_sha256':sha256(parent/'eegoar_best.pt'),
        'bandroute_checkpoint_sha256':sha256(parent/'bandroute_best.pt'),
        'historical_checkpoint_sha256':sha256(checkpoint),'eegoar_missing_columns':['FP1','FP2'],
        'eegoar_original_custom_order':names,'missing_context_training_previously_performed':False,
        'author_pretraining_OSF_participant_exposure':'not independently verified; exploratory development validation'})
    return historical,eegoar,bandroute,device


def _window_predictions(raw,eyes,mwf,models,config):
    historical,eegoar,bandroute,device=models
    names=config['common_frontal_names']
    modes,valid,residuals,diagnostics=[],[],[],[]
    for channel,vector in enumerate(raw):
        try:
            vectors,residual,detail=rolling_decompose(vector,3,2000,max_iterations=2000)
            accepted=not detail['hit_iteration_limit']
        except Exception as error:
            vectors,residual,accepted=np.zeros((3,len(vector))),vector.copy(),False
            detail={'error':repr(error),'hit_iteration_limit':False}
        modes.append(vectors);valid.append(accepted);residuals.append(residual)
        association=correlations(vectors,eyes)
        for index,row in enumerate(association):
            diagnostics.append({'method':'historical_vmd','channel':names[channel],'mode':index+1,
                                'heog_cc':float(row[0]),'veog_cc':float(row[1]),
                                'valid':accepted,**detail})
    modes=np.asarray(modes)
    failures=[]
    predictions={'identity':raw.copy()}
    extra_modes={}
    try:predictions['direct_eog_regression']=raw-window_regression(raw,eyes)
    except Exception as error:
        predictions['direct_eog_regression']=raw.copy()
        failures.append({'method':'direct_eog_regression','error':repr(error),'fallback':'identity'})
    for method in METHODS:
        try:
            if method=='historical_vmd':prediction=historical_predict(historical,raw,modes,eyes,np.asarray(valid))
            elif method=='gevd_mwf':prediction=raw-mwf.artifact(raw)
            elif method=='bandroutenet':prediction=bandroute_predict(bandroute,raw,device)
            elif method=='eegoarnet_finetuned':
                padded,present,positions=pack_frontal_montage(raw,names,eegoar.slot_contract['input_channel_names'])
                prediction=eegoar.predict(padded,present=present)[positions]
            else:
                solver=svmd_fcm if method=='svmd_fcm_adaptation' else vme_gmetv
                settings=config['svmd' if method=='svmd_fcm_adaptation' else 'vme_gmetv']
                outputs=[]
                for channel,vector in enumerate(raw):
                    result,detail=solver(vector,config=settings)
                    outputs.append(result)
                    prefix=method+'_channel_'+str(channel)+'_'
                    for key,value in detail.items():
                        if isinstance(value,np.ndarray):extra_modes[prefix+key]=value
                    association=correlations(detail['modes'],eyes)
                    for index,row in enumerate(association):
                        diagnostics.append({'method':method,'channel':names[channel],'mode':index+1,
                            'heog_cc':float(row[0]),'veog_cc':float(row[1]),
                            'center_hz':float(detail['centers_hz'][index]),
                            'mode_cap_reached':detail.get('mode_cap_reached',False),
                            'valid':True})
                prediction=np.stack(outputs)
            if prediction.shape!=raw.shape or not np.isfinite(prediction).all():
                raise ValueError('Invalid native frontal output')
            predictions[method]=prediction
        except Exception as error:
            predictions[method]=raw.copy()
            failures.append({'method':method,'error':repr(error),'traceback':traceback.format_exc(),
                             'fallback':'identity'})
    return predictions,{'modes':modes,'residual':np.asarray(residuals),'valid':np.asarray(valid),
                        **extra_modes},diagnostics,failures


def run_native(root,output,source_root,config,models):
    partition=json.loads((Path(source_root)/config['confirmation_manifest']).read_text())['partitions']
    rows,rest_rows,failures,mode_rows,timings=[],[],[],[],[]
    selected=osf_sessions(root,partition,config['osf_max_sessions'])
    if not selected:raise ValueError('No eligible OSF development sessions')
    for path in selected:
        trials,metadata=osf_trials(path)
        names=[name.upper() for name in metadata['channel_names']]
        requested=config['common_frontal_names']
        indices=[names.index(name.upper()) for name in requested]
        if any(metadata['regions'][index]!='frontal' for index in indices):
            raise ValueError('Nonfrontal channel reached native frontal experiment')
        calibration,scoring,policy=calibration_split(trials)
        ocular=np.concatenate([trial['eeg'][indices] for trial in calibration if trial['condition']!=1],axis=-1)
        rest=np.concatenate([trial['eeg'][indices] for trial in calibration if trial['condition']==1],axis=-1)
        mwf=fit_mwf(ocular,rest,config['mwf_rank'],config['mwf_strength'])
        directory=output/'osf'/metadata['record'];directory.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(directory/'mwf_calibration.npz',operator=mwf.operator,
                            mean=mwf.mean,eigenvalues=mwf.eigenvalues)
        save_json(directory/'source_and_calibration.json',{**metadata,'selected_frontal_names':requested,
            'selected_source_rows':indices,'calibration_policy':policy,
            'calibration_trial_ids':[trial['index'] for trial in calibration],
            'scoring_trial_ids':[trial['index'] for trial in scoring],
            'shared_frontal_comparison':True,'paired_clean_target_available':False,
            'mwf_annotations':'native ocular/rest conditions in unscored calibration only'})
        counts={label:0 for label in CONDITIONS}
        for trial in scoring:
            condition=trial['condition']
            if counts[condition]>=config['max_trials_per_condition']:continue
            counts[condition]+=1
            began=time.perf_counter()
            raw,eyes,evaluation=trial['eeg'][indices],trial['eyes'],trial['evaluation_eyes']
            starts=window_starts(raw.shape[-1],config['window'],config['hop'])
            window_outputs={method:[] for method in ['identity','direct_eog_regression',*METHODS]}
            for start in starts:
                window=slice(start,start+config['window'])
                predictions,modes,diagnostics,errors=_window_predictions(raw[:,window],eyes[:,window],mwf,models,config)
                for method,prediction in predictions.items():window_outputs[method].append(prediction)
                for detail in diagnostics:mode_rows.append({'record':metadata['record'],'trial':trial['index'],
                    'start':start,'condition':CONDITIONS[condition],**detail})
                for error in errors:failures.append({'record':metadata['record'],'trial':trial['index'],'start':start,**error})
                np.savez_compressed(directory/f'trial_{trial["index"]:03d}_start_{start}_modes.npz',
                    **modes,channel_names=np.asarray(requested),source_row_indices=np.asarray(indices),
                    heog=evaluation[0,window],veog=evaluation[1,window])
            outputs={'raw':raw,'heog':evaluation[0],'veog':evaluation[1],
                     'channel_names':np.asarray(requested),'source_row_indices':np.asarray(indices)}
            before=correlations(raw,evaluation)
            for method,windows in window_outputs.items():
                prediction=overlap_add(np.stack(windows),starts,raw.shape[-1])
                outputs[method]=prediction.astype(np.float32)
                after=correlations(prediction,evaluation)
                for channel,name in enumerate(requested):
                    detail={'method':method,'record':metadata['record'],'participant':metadata['participant'],
                        'trial':trial['index'],'condition':CONDITIONS[condition],'channel':name}
                    for axis,label in enumerate(['heog','veog']):
                        eligible=bool(np.std(evaluation[axis])>1e-12)
                        a,b=(float(before[channel,axis]),float(after[channel,axis])) if eligible else (float('nan'),float('nan'))
                        detail.update({label+'_cc_before':a,label+'_cc_after':b,
                                       label+'_abs_cc_before':abs(a),label+'_abs_cc_after':abs(b),
                                       label+'_reference_nonconstant':eligible})
                    rows.append(detail)
                if condition==1:
                    rest_rows.append({'method':method,'record':metadata['record'],
                        'participant':metadata['participant'],'trial':trial['index'],
                        'resting_input_cc':float(np.mean([_cc(a,b) for a,b in zip(prediction,raw)])),
                        **preservation(prediction,raw)})
            np.savez_compressed(directory/f'trial_{trial["index"]:03d}_predictions.npz',**outputs)
            timings.append({'record':metadata['record'],'trial':trial['index'],
                            'condition':CONDITIONS[condition],'all_methods_runtime_s':time.perf_counter()-began})
            pd.DataFrame(rows).to_csv(output/'osf_frontal_before_after_per_channel.csv',index=False)
            pd.DataFrame(rest_rows).to_csv(output/'osf_frontal_rest_preservation.csv',index=False)
            pd.DataFrame(timings).to_csv(output/'osf_runtime.csv',index=False)
            pd.DataFrame(mode_rows).to_csv(output/'osf_frontal_mode_eog_correlations.csv',index=False)
            save_json(output/'osf_method_failures.json',failures)
            print('OSF_FRONTAL_TRIAL_SAVED',metadata['record'],trial['index'],CONDITIONS[condition],flush=True)
    frame=pd.DataFrame(rows)
    metric_names=[name for name in frame if '_cc_' in name]
    participants=frame.groupby(['method','participant','condition'],as_index=False)[metric_names].mean()
    participants.to_csv(output/'osf_frontal_participant_correlations.csv',index=False)
    participants.groupby(['method','condition'],as_index=False)[metric_names].mean().to_csv(
        output/'osf_frontal_correlation_summary.csv',index=False)
    save_json(output/'native_completion.json',{'complete':True,'sessions':len(selected),
        'participants':sorted(frame.participant.unique().tolist()),'trials':len(timings),
        'processed_channel_names':config['common_frontal_names'],'regions_processed':['frontal'],
        'paired_clean_target':False,'reconstruction_snr_reported':False,'failed_method_windows':len(failures)})


def run(input_root,output,source_root,config):
    if not Path('/kaggle/input').exists():raise RuntimeError('Numerical work is Kaggle only')
    output=Path(output)
    parent,verify=verified_parent(input_root,config)
    klados_clean_audit(parent,verify,source_root,output)
    models=load_models(input_root,parent,source_root,output,config)
    work=Path(tempfile.mkdtemp(prefix='native-frontal-originals-'))
    root=prepare_sources(input_root,work/'originals',output,config['source_supplement_sha256'])
    run_native(root,output,source_root,config,models)
