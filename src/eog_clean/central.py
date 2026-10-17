"""Separate FC/C/CP and temporal calibration experiments; no frontal correction."""
import json
from pathlib import Path
import re
import tempfile
import time
import numpy as np
import pandas as pd
from .data import (save_json, prepare_sources, klados, preprocess, osf_trials,
                   osf_sessions, calibration_split)
from .methods import fit_ridge, fit_mwf
from .metrics import paired, correlations, preservation
from .pretrained_review import verified_table_parent

CONDITIONS = {1:'rest', 2:'lateral', 3:'vertical', 4:'blink'}
RECIPES = [('identity', None, 0.), ('ridge_025', 'ridge', .25),
           ('ridge_100', 'ridge', 1.), ('mwf_025', 'mwf', .25), ('mwf_100', 'mwf', 1.)]


def channel_group(name):
    name = name.strip().upper()
    if re.fullmatch(r'(?:FC|C|CP)(?:\d+|Z)', name):
        return 'central'
    if re.fullmatch(r'(?:FT|T|TP)(?:\d+|Z)', name):
        return 'temporal'
    return None


def group_indices(names, group):
    if len({name.upper() for name in names}) != len(names):
        raise ValueError('Duplicate electrode labels')
    return [index for index, name in enumerate(names) if channel_group(name) == group]


def fit_group(values, eyes, ocular, rest, failures, identity):
    fits = {}
    for method, function in [('ridge', lambda: fit_ridge(values, eyes, .001)),
                             ('mwf', lambda: fit_mwf(ocular, rest, rank=1, strength=1.))]:
        try:
            fits[method] = function()
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as error:
            fits[method] = None
            failures.append({'identity':identity, 'method':method,
                             'error':type(error).__name__+': '+str(error), 'fallback':'identity'})
    return fits


def predictions(raw, eyes, fits):
    artifacts = {name: model.artifact(raw, eyes) if model is not None else np.zeros_like(raw)
                 for name, model in fits.items()}
    return {name: raw if method is None else raw-strength*artifacts[method]
            for name, method, strength in RECIPES}


def correlation_rows(raw, cleaned, eyes, names, identity):
    before, after = correlations(raw, eyes), correlations(cleaned, eyes)
    rows = []
    for index, name in enumerate(names):
        row = {**identity, 'channel':name}
        for axis, label in enumerate(['heog', 'veog']):
            valid = bool(np.std(eyes[axis]) > 1e-12 and np.std(cleaned[index]) > 1e-12
                         and np.std(raw[index]) > 1e-12)
            a, b = (float(before[index, axis]), float(after[index, axis])) if valid else (np.nan, np.nan)
            row.update({label+'_cc_before':a, label+'_cc_after':b,
                        label+'_abs_cc_before':abs(a), label+'_abs_cc_after':abs(b)})
        rows.append(row)
    return rows


def save_fits(directory, fits):
    for name, model in fits.items():
        if model is not None:
            np.savez_compressed(directory/(name+'_calibration.npz'), **{
                key:np.asarray(value) for key, value in vars(model).items()})


def run_klados(root, output, names, records, failures):
    dirty, clean, eyes, source = klados(root)
    save_json(output/'klados_sources.json', {**source, 'label_policy':'paper-list row order, unverified binding',
        'paired_preprocessing':'200Hz .5–40Hz independently before/after 2000-sample calibration boundary',
        'comparison_to_frontal':'different calibration/filter-boundary protocol; not a matched superiority comparison'})
    rows, controls, timing = [], [], []
    for record in records:
        for group in ['central', 'temporal']:
            indices = group_indices(names, group)
            labels = [names[index] for index in indices]
            if not indices:
                raise ValueError('Missing required Klados group')
            directory = output/'klados'/f'record_{record:02d}'/group
            directory.mkdir(parents=True, exist_ok=True)
            began = time.perf_counter()
            calibration = preprocess(dirty[record, indices, :2000])
            cal_eyes = preprocess(eyes[record, :, :2000])
            activity = np.mean((cal_eyes/np.maximum(cal_eyes.std(axis=-1, keepdims=True), 1e-12))**2, axis=0)
            lower, upper = np.quantile(activity, [.3, .7])
            fits = fit_group(calibration, cal_eyes, calibration[:, activity >= upper],
                             calibration[:, activity <= lower], failures, f'Klados:{record}:{group}')
            save_fits(directory, fits)
            raw_all = preprocess(dirty[record, indices, 2000:])
            clean_all = preprocess(clean[record, indices, 2000:])
            eyes_all = preprocess(eyes[record, :, 2000:])
            save_json(directory/'calibration.json', {'source_row_indices':indices,'channel_names':labels,
                'fit_samples':[0,2000], 'scoring_starts':[2000,3024,4048],
                'clean_target_used_for_fitting':False, 'mwf_masks':'calibration EOG activity bottom/top30%'})
            for start in [2000, 3024, 4048]:
                segment = slice(start-2000, start-2000+1024)
                raw, target, reference = raw_all[:,segment], clean_all[:,segment], eyes_all[:,segment]
                if raw.shape[-1] != 1024:
                    raise ValueError('Paired scoring interval exceeds source')
                outputs = predictions(raw, reference, fits)
                clean_outputs = predictions(target, reference, fits)
                baseline = paired(raw, target)
                stored = {'raw':raw,'clean':target,'eyes':reference,'channel_names':np.asarray(labels),
                          'source_row_indices':np.asarray(indices)}
                for method, prediction in outputs.items():
                    scores = paired(prediction, target)
                    details = correlation_rows(raw, prediction, reference, labels,
                        {'record':record,'start':start,'group':group,'method':method})
                    for channel, row in enumerate(details):
                        row.update({metric:float(scores[metric][channel]) for metric in
                            ['mse','rmse','rrmse_t','rrmse_s','pearson_cc','snr_out_db']})
                        row.update(pooled_window_snr_db=scores['pooled_window_snr_db'],
                            pooled_snr_improvement_db=scores['pooled_window_snr_db']-baseline['pooled_window_snr_db'])
                    rows.extend(details)
                    controls.append({'record':record,'start':start,'group':group,'method':method,
                                     **preservation(clean_outputs[method], target)})
                    stored[method] = prediction.astype(np.float32)
                    stored[method+'_clean_control'] = clean_outputs[method].astype(np.float32)
                np.savez_compressed(directory/f'window_{start}.npz', **stored)
            timing.append({'dataset':'Klados','record':record,'group':group,'runtime_s':time.perf_counter()-began})
            save_json(directory/'completion.json', {'complete':True,'record':record,'group':group})
            pd.DataFrame(rows).to_csv(output/'klados_per_channel.csv',index=False)
            pd.DataFrame(controls).to_csv(output/'klados_clean_preservation.csv',index=False)
            print('CENTRAL_KLADOS_SAVED',record,group,flush=True)
    frame = pd.DataFrame(rows)
    record_table = frame.groupby(['group','method','record'],as_index=False).mean(numeric_only=True)
    record_table.groupby(['group','method'],as_index=False).mean(numeric_only=True).to_csv(output/'klados_summary.csv',index=False)
    pd.DataFrame(timing).to_csv(output/'klados_runtime.csv',index=False)


def run_native(root, output, partition, failures):
    rows, rest_rows, timings, mappings = [], [], [], []
    for path in osf_sessions(root, partition, 4):
        trials, metadata = osf_trials(path)
        calibration, scoring, policy = calibration_split(trials)
        for group in ['central', 'temporal']:
            indices = group_indices(metadata['channel_names'], group)
            if not indices:
                raise ValueError('Required native group absent')
            labels = [metadata['channel_names'][index] for index in indices]
            values = np.concatenate([trial['eeg'][indices] for trial in calibration],axis=-1)
            eyes = np.concatenate([trial['eyes'] for trial in calibration],axis=-1)
            ocular = np.concatenate([trial['eeg'][indices] for trial in calibration if trial['condition'] != 1],axis=-1)
            rest = np.concatenate([trial['eeg'][indices] for trial in calibration if trial['condition'] == 1],axis=-1)
            fits = fit_group(values, eyes, ocular, rest, failures, metadata['record']+':'+group)
            directory = output/'osf'/metadata['record']/group
            directory.mkdir(parents=True,exist_ok=True)
            save_fits(directory,fits)
            mapping = {**metadata,'group':group,'selected_channel_names':labels,'source_row_indices':indices,
                'calibration_policy':policy,'calibration_trial_ids':[trial['index'] for trial in calibration],
                'scoring_trial_ids':[trial['index'] for trial in scoring]}
            save_json(directory/'source_and_calibration.json',mapping)
            mappings.append({'record':metadata['record'],'group':group,'names':labels})
            counts = {condition:0 for condition in CONDITIONS}
            for trial in scoring:
                condition = trial['condition']
                if counts[condition] >= 2:
                    continue
                counts[condition] += 1
                began = time.perf_counter()
                raw, reference = trial['eeg'][indices], trial['evaluation_eyes']
                outputs = predictions(raw, trial['eyes'], fits)
                stored = {'raw':raw,'heog':reference[0],'veog':reference[1],
                          'channel_names':np.asarray(labels),'source_row_indices':np.asarray(indices)}
                for method, prediction in outputs.items():
                    rows.extend(correlation_rows(raw,prediction,reference,labels,
                        {'record':metadata['record'],'participant':metadata['participant'],'trial':trial['index'],
                         'condition':CONDITIONS[condition],'group':group,'method':method}))
                    stored[method] = prediction.astype(np.float32)
                    if condition == 1:
                        rest_rows.append({'participant':metadata['participant'],'record':metadata['record'],
                            'trial':trial['index'],'group':group,'method':method,**preservation(prediction,raw)})
                np.savez_compressed(directory/f'trial_{trial["index"]:03d}.npz',**stored)
                timings.append({'record':metadata['record'],'trial':trial['index'],'group':group,
                                'condition':CONDITIONS[condition],'runtime_s':time.perf_counter()-began})
                pd.DataFrame(rows).to_csv(output/'osf_per_channel.csv',index=False)
                pd.DataFrame(rest_rows).to_csv(output/'osf_rest_preservation.csv',index=False)
            save_json(directory/'completion.json',{'complete':True,'group':group,'counts':counts})
            print('CENTRAL_OSF_SAVED',metadata['record'],group,flush=True)
    if not rows:
        raise ValueError('No native scoring trials')
    frame = pd.DataFrame(rows)
    numeric = [name for name in frame if '_cc_' in name]
    participants = frame.groupby(['group','method','condition','participant'],as_index=False)[numeric].mean()
    participants.groupby(['group','method','condition'],as_index=False)[numeric].mean().to_csv(output/'osf_summary.csv',index=False)
    rest = pd.DataFrame(rest_rows)
    participants = rest.groupby(['group','method','participant'],as_index=False).mean(numeric_only=True)
    participants.groupby(['group','method'],as_index=False).mean(numeric_only=True).to_csv(output/'osf_rest_summary.csv',index=False)
    participants.groupby(['group','method'],as_index=False).max(numeric_only=True).to_csv(output/'osf_rest_worst_participant.csv',index=False)
    pd.DataFrame(timings).to_csv(output/'osf_runtime.csv',index=False)
    save_json(output/'channel_mappings.json',mappings)


def run(input_root, source_root, output):
    if not Path('/kaggle/input').exists():
        raise RuntimeError('Numerical work is Kaggle only')
    source_root, output = Path(source_root), Path(output)
    config = json.loads((source_root/'configs/central.json').read_text())
    verify = verified_table_parent(input_root,'frontal-six-full-001','c19637465e5a5c29991c2a988f25e6527a156f09')
    contract = json.loads(verify('data_contract.json').read_text())
    records = next(item['records'] for item in contract['files'] if item['partition']=='test')
    montage = json.loads((source_root/'docs/provenance/klados_montage.json').read_text())
    partition = json.loads((source_root/'docs/provenance/osf_partition.json').read_text())['partitions']
    work = Path(tempfile.mkdtemp(prefix='central-originals-'))
    root = prepare_sources(input_root,work/'sources',output,config['source_supplement_sha256'])
    failures = []
    run_klados(root,output,montage['channel_names_in_paper_list_order'],records,failures)
    run_native(root,output,partition,failures)
    save_json(output/'calibration_failures.json',failures)
    save_json(output/'completion.json',{'complete':True,'groups':['central','temporal'],
        'central_definition':'FC/C/CP','temporal_definition':'FT/T/TP, separate fits and results',
        'klados_records':records,'recipes':RECIPES,'ridge_penalty':.001,'mwf_rank':1,
        'calibration_failures':len(failures),'paired_clean_used_for_fitting':False,
        'VMD_used':False,'LEMON_used':False,'frontal_context_used':False,
        'confirmation_opened':False,'production_qualified':False})
    print('CENTRAL_EXPERIMENT_COMPLETE',flush=True)


def report(output):
    import matplotlib.pyplot as plt
    output = Path(output)
    paired_table = pd.read_csv(output/'klados_summary.csv')
    native = pd.read_csv(output/'osf_summary.csv')
    controls = pd.read_csv(output/'klados_clean_preservation.csv')
    rest = pd.read_csv(output/'osf_rest_worst_participant.csv')
    scores = pd.read_csv(output/'klados_per_channel.csv')
    record_scores = scores.groupby(['group','method','record'],as_index=False).mean(numeric_only=True)
    gates = []
    for group in ['central','temporal']:
        comparator = record_scores[(record_scores.group==group)&(record_scores.method=='ridge_100')].set_index('record')
        for method, _, _ in RECIPES:
            clean = controls[(controls.group==group)&(controls.method==method)].groupby('record').mean(numeric_only=True)
            native_worst = rest[(rest.group==group)&(rest.method==method)].iloc[0]
            selected = record_scores[(record_scores.group==group)&(record_scores.method==method)].set_index('record')
            row = {'group':group,'method':method,
                'worst_record_clean_change':float(clean.relative_change.max()),
                'worst_record_clean_alpha_db':float(clean.alpha_error_db.max()),
                'worst_record_clean_beta_db':float(clean.beta_error_db.max()),
                'worst_record_clean_covariance':float(clean.covariance_relative_error.max()),
                'worst_record_cc_deterioration_vs_ridge100':float((comparator.pearson_cc-selected.pearson_cc).max()),
                'native_worst_participant_rest_change':float(native_worst.relative_change),
                'native_worst_participant_alpha_db':float(native_worst.alpha_error_db),
                'native_worst_participant_beta_db':float(native_worst.beta_error_db),
                'native_worst_participant_covariance':float(native_worst.covariance_relative_error)}
            row['measured_preservation_gates_pass'] = bool(
                row['worst_record_clean_change']<=.01 and row['worst_record_clean_alpha_db']<=.5
                and row['worst_record_clean_beta_db']<=.5 and row['worst_record_clean_covariance']<=.02
                and row['worst_record_cc_deterioration_vs_ridge100']<=.005
                and row['native_worst_participant_rest_change']<=.01
                and row['native_worst_participant_alpha_db']<=.5 and row['native_worst_participant_beta_db']<=.5
                and row['native_worst_participant_covariance']<=.02)
            gates.append(row)
    gate_table = pd.DataFrame(gates)
    gate_table.to_csv(output/'preservation_gates.csv',index=False)
    directory = output/'figures';directory.mkdir(exist_ok=True)
    paths = []
    def save(figure,name):
        figure.tight_layout()
        for extension in ['png','pdf']:
            figure.savefig(directory/(name+'.'+extension),dpi=160,bbox_inches='tight')
        paths.append(str(directory/(name+'.png')));plt.close(figure)
    for number, (metric,title) in enumerate([
        ('pooled_window_snr_db','Paired output SNR (dB)'),('rmse','Paired RMSE (source amplitude units)'),
        ('pearson_cc','Pearson correlation with paired clean EEG')],1):
        figure,axes = plt.subplots(1,2,figsize=(13,4))
        for axis,group in zip(axes,['central','temporal']):
            table = paired_table[paired_table.group==group]
            axis.barh(table.method,table[metric]);axis.set_title(group);axis.set_xlabel(title)
            axis.grid(axis='x',alpha=.2)
        figure.suptitle('Klados: independently calibrated groups, equal record weight')
        save(figure,f'{number:02d}_paired_{metric}')
    for number,condition in enumerate(['blink','lateral','vertical'],4):
        figure,axes = plt.subplots(2,2,figsize=(13,8))
        for row,group in enumerate(['central','temporal']):
            table = native[(native.group==group)&(native.condition==condition)]
            for axis,reference in zip(axes[row],['heog','veog']):
                axis.barh(table.method,table[reference+'_abs_cc_after'])
                axis.axvline(float(table[reference+'_abs_cc_before'].iloc[0]),linestyle='--',color='black',label='Before')
                axis.set(title=group+' / '+reference.upper(),xlabel='Mean absolute Pearson CC',xlim=(0,1))
                axis.legend();axis.grid(axis='x',alpha=.2)
        figure.suptitle('Native OSF '+condition+': unchanged references, equal participant weight')
        save(figure,f'{number:02d}_native_{condition}')
    for number,(frame,caption) in enumerate([(controls,'Klados clean-input modification'),
                                           (pd.read_csv(output/'osf_rest_preservation.csv'),'Native OSF resting-input modification')],7):
        unit='record' if number==7 else 'participant'
        summary=frame.groupby(['group','method',unit],as_index=False).mean(numeric_only=True)
        summary=summary.groupby(['group','method'],as_index=False).max(numeric_only=True)
        figure,axes=plt.subplots(2,2,figsize=(13,8))
        for row,group in enumerate(['central','temporal']):
            table=summary[summary.group==group]
            axes[row,0].barh(table.method,100*table.relative_change)
            axes[row,0].axvline(1,color='black',linestyle='--')
            axes[row,0].set(title=group,xlabel='Worst '+unit+' mean relative RMS change (%)')
            axes[row,1].barh(table.method,table.covariance_relative_error)
            axes[row,1].axvline(.02,color='black',linestyle='--')
            axes[row,1].set(title=group,xlabel='Worst '+unit+' mean relative covariance error')
        figure.suptitle(caption+'; dashed lines are fixed engineering thresholds')
        save(figure,f'{number:02d}_preservation')
    text='# Central and separately fitted temporal comparison\n\n'
    text+='Central = FC/C/CP. Temporal = FT/T/TP. No frontal/posterior correction, VMD, neural training or frontal communication. '
    text+='Ridge penalty0.001, MWF rank1; strengths0.25 and1 are fixed ablations, not a parameter search. '
    text+='Fit on calibration EEG/EOG only; no clean targets fit a correction.\n\n'
    text+=paired_table.round(6).to_markdown(index=False)+'\n\n'
    text+=native.round(6).to_markdown(index=False)+'\n\n'
    text+=gate_table.round(6).to_markdown(index=False)+'\n\n'
    text+='Klados uses paper-list row binding as an explicit assumption and reused test records. '
    text+='Its independently filtered calibration/scoring protocol differs from archived frontal caches; '
    text+='do not compare regional SNRs as matched improvements or add dB. Native rest is unpaired and no native reconstruction SNR is reported. '
    text+='Identity passing preservation alone does not establish artifact suppression. Any future integration requires separate review.\n'
    (output/'RESULTS.md').write_text(text,encoding='utf-8')
    return paths
