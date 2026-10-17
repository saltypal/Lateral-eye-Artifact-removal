"""Build one Markdown comparison report from verified saved metrics on Kaggle."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .data import save_json,sha256
from .pretrained_review import verified_table_parent

PARENTS={
    'frontal':('frontal-spatial-full-002','d87b2e43116c9a7d58dd3adf755ab8df6a0698e1'),
    'middle':('central-temporal-001','20584ab33028b43fada3912960e926c7948f70cc'),
    'combined':('frontal-posterior-combined-001','f94436ab167d3019aacb4ceeadb64775b42d10c0'),
    'historical':('klados-osf-pilot-002','da7e2778fcc0187565c23064829bf67357584ea7')}

LABELS={'identity':'Unprocessed / pass-through','historical_vmd':'Historical VMD + EOG model',
    'direct_eog_regression':'Window-fitted EOG ridge','direct_calibrated_ridge':'Calibration-frozen EOG ridge',
    'calibrated_ridge':'Calibration-frozen EOG ridge','regica':'REG-ICA adaptation',
    'sgeyesub':'SGEYESUB (full-montage fit)','bandroutenet':'BandRouteNet adaptation',
    'eegoarnet_finetuned':'EEGOAR-Net custom-montage fine-tuning','gevd_mwf':'GEVD-MWF frontal comparator',
    'svmd_fcm_adaptation':'SVMD-FCM adaptation','vme_gmetv_adaptation':'VME-GMETV adaptation',
    'resampling_identity':'Resampling-only control','ridge_025':'EOG ridge, 25% correction',
    'ridge_100':'EOG ridge, 100% correction','mwf_025':'Rank-1 MWF, 25% correction',
    'mwf_100':'Rank-1 MWF, 100% correction','mwf_rank1':'Rank-1 MWF'}
QUALITY=['pooled_window_snr_db','pooled_snr_improvement_db','mse','rmse','rrmse_t','rrmse_s','pearson_cc']
CC=[axis+suffix for axis in ['heog','veog'] for suffix in
    ['_cc_before','_cc_after','_abs_cc_before','_abs_cc_after']]
HEADERS={'method':'Method','group':'Region','record':'Record','condition':'Condition',
    'frontal_method':'Frontal','posterior_method':'Posterior','pooled_window_snr_db':'Output SNR dB',
    'pooled_snr_improvement_db':'SNR gain dB','mse':'MSE','rmse':'RMSE','rrmse_t':'RRMSE time',
    'rrmse_s':'RRMSE spectrum','pearson_cc':'Clean EEG Pearson CC',
    **{axis+'_abs_cc_'+time:axis.upper()+' mean |r| '+time for axis in ['heog','veog'] for time in ['before','after']},
    **{axis+'_cc_'+time:axis.upper()+' signed r '+time for axis in ['heog','veog'] for time in ['before','after']}}


def frontal_correlations(frame):
    """Unify older signed-after columns and new before/after columns by identity."""
    frame=frame.copy();keys=['record','start','channel']
    baseline=frame[frame.method=='identity'].set_index(keys)
    if baseline.index.has_duplicates:raise ValueError('Duplicate identity correlation fixture')
    for axis in ['heog','veog']:
        after=axis+'_cc_after';old=axis+'_cc'
        frame[after]=frame[after].combine_first(frame[old])
        lookup=baseline[old].reindex(pd.MultiIndex.from_frame(frame[keys])).to_numpy()
        available=frame[axis+'_cc_before'].notna()
        np.testing.assert_allclose(frame.loc[available,axis+'_cc_before'],lookup[available],rtol=0,atol=1e-10)
        frame[axis+'_cc_before']=lookup
        for time in ['before','after']:frame[axis+'_abs_cc_'+time]=frame[axis+'_cc_'+time].abs()
    if frame[CC].isna().any().any():raise ValueError('Incomplete paired frontal EOG evidence')
    return frame


def correlation_rollup(frame,keys):
    """Equal channel, window, then record weight; absolute values precede means."""
    if frame[['record','start','channel',*keys]].isna().any().any():raise ValueError('Missing grouping identity')
    by_window=frame.groupby([*keys,'record','start'],as_index=False)[CC].mean()
    by_record=by_window.groupby([*keys,'record'],as_index=False)[CC].mean()
    return by_record.groupby(keys,as_index=False)[CC].mean()


def build(input_root,output):
    if not Path('/kaggle/input').exists():raise RuntimeError('All metric aggregation runs on Kaggle')
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    audit=[];tables=[];sections=[]
    def read(parent,name,json_file=False):
        run_id,source=PARENTS[parent]
        path=verified_table_parent(input_root,run_id,source)(name)
        audit.append({'run_id':run_id,'git_sha':source,'file':name,'sha256':sha256(path)})
        return json.loads(path.read_text()) if json_file else pd.read_csv(path)
    def emit(title,frame,note=''):
        sections.append('## '+title+'\n\n')
        if note:sections.append(note+'\n\n')
        table=frame.copy()
        for column in ['method','frontal_method','posterior_method']:
            if column in table:
                table[column]=table[column].map(lambda value: LABELS.get(value,str(value).replace('__',' / ')))
        table=table.rename(columns=HEADERS)
        sections.append(table.to_markdown(index=False,floatfmt='.6f',missingval='N/A').replace('nan','N/A')+'\n\n')
        tables.append({'title':title,'rows':len(frame),'columns':len(frame.columns)})
    absolute=['heog_abs_cc_before','heog_abs_cc_after','veog_abs_cc_before','veog_abs_cc_after']
    signed=['heog_cc_before','heog_cc_after','veog_cc_before','veog_cc_after']
    emit('1. Dataset and protocol scope',pd.DataFrame([
        {'Comparison':'Frontal paired','Dataset':'Klados','Extent':'8 reused records; 24 windows; 7 frontal rows','Protocol':'Archived scoring arrays; full-montage REG-ICA fit; other method access disclosed'},
        {'Comparison':'Central / temporal paired','Dataset':'Klados','Extent':'8 reused records; 24 windows per group','Protocol':'Independent calibration/scoring filtering; not identical to frontal caches'},
        {'Comparison':'Posterior / combined paired','Dataset':'Klados','Extent':'8 reused records; 24 windows; 19-channel targets','Protocol':'Same archived scoring arrays as frontal; central/temporal unchanged'},
        {'Comparison':'Native frontal','Dataset':'OSF','Extent':'4 participants; 32 trials; 5 exact common frontal names','Protocol':'Full 58-channel SG/REG fits; frontal outputs only'},
        {'Comparison':'Native central / temporal','Dataset':'OSF','Extent':'4 participants; 32 trials per group','Protocol':'Central FC/C/CP; temporal FT/T/TP separate region-only fits'},
        {'Comparison':'Native posterior / combined','Dataset':'OSF','Extent':'Not run in the current accepted regional stage','Protocol':'No native reconstruction SNR without clean targets'}]))
    front=read('frontal','klados_comparison.csv')
    emit('2. Klados frontal reconstruction',front[['method',*QUALITY]],
         'Primary output SNR pools channel energies within each window, then averages windows and records in dB. SGEYESUB is not eligible for a paired Klados score because its six-class calibration labels are unavailable.')
    frontal_cc=correlation_rollup(frontal_correlations(read('frontal','klados_all_per_channel.csv')),['method'])
    frontal_cc.to_csv(output/'frontal_paired_eog_correlations.csv',index=False)
    emit('3. Klados frontal HEOG / VEOG correlation',frontal_cc[['method',*absolute]],
         'Mean absolute Pearson correlations; lower association alone does not establish neural preservation.')
    middle=read('middle','klados_summary.csv');middle_channels=read('middle','klados_per_channel.csv')
    middle_cc=correlation_rollup(middle_channels,['group','method'])
    for group,number in [('central',4),('temporal',6)]:
        emit(f'{number}. Klados {group} reconstruction',middle[middle.group==group][['method',*QUALITY]],
            'This earlier region-only protocol independently filtered calibration/scoring segments; do not rank its dB against the archived frontal/combined protocol as a matched comparison. 25% and 100% are correction strengths of the same fitted algorithm, not different algorithms.')
        emit(f'{number+1}. Klados {group} HEOG / VEOG correlation',middle_cc[middle_cc.group==group][['method',*absolute]])
    combined=read('combined','comparison.csv');combined_channels=read('combined','per_channel.csv')
    combined_cc=correlation_rollup(combined_channels,['group','method'])
    combined_cc.to_csv(output/'combined_paired_eog_correlations.csv',index=False)
    posterior=combined[(combined.group=='posterior')&(combined.frontal_method=='identity')]
    emit('8. Klados posterior reconstruction',posterior[['posterior_method',*QUALITY]],
         'Posterior rows P3/P4/O1/O2/Pz; values are independent of the frontal choice because masks are disjoint. MWF is the instantaneous rank-1 adaptation, not a new exact author reproduction.')
    chosen=combined_cc[(combined_cc.group=='posterior')&combined_cc.method.str.startswith('identity__')].copy()
    chosen['posterior_method']=chosen.method.str.split('__').str[1]
    emit('9. Klados posterior HEOG / VEOG correlation',chosen[['posterior_method',*absolute]])
    whole=combined[combined.group=='whole_cap']
    emit('10. Klados combined full 19-channel reconstruction',whole[['frontal_method','posterior_method',*QUALITY]],
         'Whole-cap SNR is computed from the combined prediction and clean target. Regional dB values are never added. Central C3/C4/Cz and temporal T3/T4/T5/T6 stay exactly unprocessed. This is composition, not frontal-to-back message passing.')
    whole_cc=whole[['method','frontal_method','posterior_method']].merge(combined_cc[combined_cc.group=='whole_cap'],on='method',validate='one_to_one')
    emit('11. Klados combined whole-cap HEOG / VEOG correlation',whole_cc[['frontal_method','posterior_method',*absolute]])
    native=read('frontal','osf_comparison.csv')
    for offset,condition in enumerate(['blink','lateral','vertical','rest'],12):
        emit(f'{offset}. Native OSF frontal: {condition}',native[native.condition==condition][['method',*absolute]],
             'Participant-balanced mean absolute correlation with unchanged original references. No native reconstruction SNR or correlation with unavailable clean EEG.')
    native_middle=read('middle','osf_summary.csv')
    number=16
    for group in ['central','temporal']:
        for condition in ['blink','lateral','vertical','rest']:
            selected=native_middle[(native_middle.group==group)&(native_middle.condition==condition)]
            emit(f'{number}. Native OSF {group}: {condition}',selected[['method',*absolute]])
            number+=1
    rest=read('frontal','osf_rest_summary.csv');worst=read('frontal','osf_rest_worst_participant.csv')
    preservation=rest[['method','relative_change','modification_rmse','alpha_error_db','beta_error_db','covariance_relative_error']].copy()
    preservation['Mean rest change %']=100*preservation.pop('relative_change')
    preservation=preservation.merge(worst[['method','relative_change']].rename(columns={'relative_change':'Worst participant rest change %'}),on='method',validate='one_to_one')
    preservation['Worst participant rest change %']*=100
    emit('24. Native OSF frontal resting preservation',preservation,
         'Modification RMSE uses the resting input as reference, not a known clean target. Covariance is relative error; alpha/beta are mean absolute band-power errors in dB. Signal modification is not a neural-loss percentage.')
    rest_middle=read('middle','osf_rest_summary.csv');worst_middle=read('middle','osf_rest_worst_participant.csv')
    rest_middle=rest_middle[['group','method','relative_change','modification_rmse','alpha_error_db','beta_error_db','covariance_relative_error']].copy()
    rest_middle['Mean rest change %']=100*rest_middle.pop('relative_change')
    rest_middle=rest_middle.merge(worst_middle[['group','method','relative_change']].rename(columns={'relative_change':'Worst participant rest change %'}),on=['group','method'],validate='one_to_one')
    rest_middle['Worst participant rest change %']*=100
    emit('25. Native OSF central / temporal resting preservation',rest_middle)
    emit('26. Frontal paired and native preservation review',read('frontal','preservation_review.csv'),
         'These are project engineering gates, separate from paper metrics. Missing paired SGEYESUB evidence is unavailable, not a zero score.')
    emit('27. Central / temporal engineering gates',read('middle','preservation_gates.csv'))
    gate=read('combined','preservation_gates.csv')
    emit('28. Combined whole-cap engineering gates',gate[gate.group=='whole_cap'])
    emit('29. Paired record-cluster whole-cap SNR differences',read('combined','combined_snr_uncertainty.csv'),
         'Comparator is the same frontal VMD with posterior pass-through. Eight record clusters; unknown participant identities. Intervals are exploratory. Positive differences would favor the added posterior method; neither addition improves this measured mean.')
    emit('30. Frontal paired / native method-difference uncertainty',read('frontal','paired_cluster_uncertainty.csv'),
         'Saved paired differences: eight Klados record clusters or four OSF participants. Signs must be interpreted per metric: higher SNR/clean CC, lower RMSE/EOG association. These are development results, not external superiority tests.')
    emit('31. Signed paired EOG correlations: frontal',frontal_cc[['method',*signed]],
         'Signed means can cancel across channels; use the absolute-correlation tables for suppression comparisons.')
    emit('32. Signed paired EOG correlations: central / temporal',middle_cc[['group','method',*signed]])
    emit('33. Signed paired EOG correlations: posterior',chosen[['posterior_method',*signed]])
    for parent,name,title,filters in [
        ('frontal','klados_per_record.csv','34. Frontal output SNR by record',None),
        ('combined','per_record_region.csv','35. Combined whole-cap output SNR by record',{'group':'whole_cap'})]:
        records=read(parent,name)
        if filters:
            for key,value in filters.items():records=records[records[key]==value]
        if records.duplicated(['record','method']).any():raise ValueError('Ambiguous record-level SNR rows')
        pivot=records.pivot(index='record',columns='method',values='pooled_window_snr_db').reset_index()
        emit(title,pivot)
    historical=read('historical','historical_replay.json',True)
    emit('36. Historical all-channel replay: separate reference',pd.DataFrame([{
        'System':'Frozen historical VMD + EOG neural model, all 19 rows',
        'Output SNR dB':historical['measured_record_macro_snr_db'],
        'Records':historical['records'],'Windows':historical['windows'],
        'Runtime EOG required':historical['runtime_eog_required'],
        'Frontal-only result':historical['verified_frontal_claim']}]),
         'This 17.9272 dB replay corrected all 19 rows. It is not the new combination with central/temporal pass-through, not VMD alone, and not fresh confirmation.')
    emit('37. Historical all-channel replay by record',read('historical','historical_record_metrics.csv').drop(columns=['start']))
    emit('38. Eligibility and interpretation',pd.DataFrame([
        {'Item':'SGEYESUB paired Klados','Status':'Not run / ineligible','Reason':'No verified six-class calibration annotations'},
        {'Item':'Native OSF reconstruction SNR / clean EEG CC','Status':'Unavailable','Reason':'No paired clean EEG target'},
        {'Item':'Current native posterior / combined evaluation','Status':'Not run','Reason':'Current combined stage evaluated paired Klados only'},
        {'Item':'SGEYESUB / REG-ICA central or posterior outputs','Status':'Not scored in the accepted spatial run','Reason':'Full-montage fitting supplied frontal outputs only; no invented regional scores'},
        {'Item':'EEG-WNet / Diff-ADN','Status':'Not run','Reason':'No verified compatible pretrained source/checkpoint pair in recorded audit'},
        {'Item':'REG-ICA solver','Status':'One capped fit retained','Reason':'Klados record 14 reached 512 steps; not asserted converged'},
        {'Item':'Frontal REG-ICA failed-version score','Status':'Superseded; preserved separately','Reason':'5.597328 dB in incomplete version2; accepted full run is 5.665687 dB'},
        {'Item':'AutoVMD / retired LEMON mixtures','Status':'Historical work; excluded from active ranking','Reason':'Retired protocols are not comparable to current accepted Klados/OSF tables'},
        {'Item':'Production qualification','Status':'Not achieved','Reason':'Preservation/evidence gates fail; no forced winner'}]))
    emit('39. Exact source runs',pd.DataFrame([{'Evidence':key,'Run':value[0],'Git SHA':value[1]} for key,value in PARENTS.items()]))
    intro='''# All results — direct comparison tables

Generated from hash-verified accepted Kaggle artifacts. This document covers
the current frontal, central, temporal, posterior and combined Klados/OSF
comparisons. Historical all-channel replay is separated. Failed/smoke and
retired protocols are not mixed into the active rankings.

- **Clean EEG CC:** ordinary signed Pearson correlation with paired clean EEG; higher is better.
- **HEOG/VEOG mean |r|:** absolute correlation calculated per channel before averaging; lower association alone does not prove neural preservation.
- **Output SNR:** 10 log10(sum(clean²) / sum((clean − prediction)²)), pooled over the declared region within a window, then averaged within record and across records in dB.
- **SNR gain:** output SNR minus the matching unprocessed input SNR, with the same aggregation. Regional dB values are never added.
- **RMSE/MSE:** retained source amplitude units / squared units; no independently verified microvolt conversion is asserted. RMSE is averaged per channel/window/record, not obtained by taking the root of the displayed mean MSE.
- **RRMSE:** temporal and spectral relative RMSE. Mean SNR, mean MSE and mean CC can rank differently; inspect record-level outcomes.
- Klados electrode names use the requested paper-list order; row binding and participant identities remain unverified. Eight reused records are not fresh confirmation.
- Native OSF uses named channels and four development participants. No native reconstruction SNR or clean EEG CC is available.
- All new aggregation in this report ran on Kaggle; no predictions, denoising parameters, targets or acceptance thresholds changed.

'''
    (output/'ALL_RESULTS_TABLES.md').write_text(intro+''.join(sections),encoding='utf-8')
    save_json(output/'table_inventory.json',tables)
    save_json(output/'table_sources.json',audit)
    save_json(output/'completion.json',{'complete':True,'tables':len(tables),
        'source_payloads_verified':len(audit),'waveform_processing':False,'predictions_changed':False})
    print('ALL_RESULTS_TABLES_COMPLETE',len(tables),flush=True)
