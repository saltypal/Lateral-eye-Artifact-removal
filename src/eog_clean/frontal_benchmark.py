"""One matched seven-channel Klados comparison, with per-record recovery."""
import json
from pathlib import Path
import random
import time
import traceback
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from scipy.signal import resample_poly
from .data import save_json, sha256
from .historical_model import EOGContextGainStudent
from .metrics import paired, preservation, correlations
from .methods import fit_mwf, window_regression
from .paper_methods import svmd_fcm, vme_gmetv
from .author_models import (fetch_author_source, EEGOAR, bandroute_model,
                            bandroute_predict)


METHODS = ['historical_vmd', 'svmd_fcm_adaptation', 'vme_gmetv_adaptation',
           'eegoarnet_finetuned', 'gevd_mwf', 'bandroutenet']


def load_caches(input_root, output, source_root, config):
    """Verify immutable parent and split original record IDs before any training."""
    candidates = list(Path(input_root).rglob('snr_target_protocol.json'))
    if len(candidates) != 1:
        raise ValueError('Require exactly one pinned historical context parent')
    parent = candidates[0].parent
    montage = json.loads((Path(source_root)/'docs/provenance/klados_montage.json').read_text())
    channels = montage['frontal_indices']
    caches, identities, files = {}, {}, []
    for partition in ('train', 'val', 'test'):
        path = parent/f'{partition}_vmd_cache.npz'
        digest = sha256(path)
        if digest != config['historical_cache_sha256'][partition]:
            raise ValueError('Historical '+partition+' cache checksum differs')
        if partition == 'test' and digest != config['historical_test_cache_sha256']:
            raise ValueError('Historical test cache checksum differs')
        with np.load(path, allow_pickle=False) as archive:
            cache = {name:archive[name].copy() for name in
                     ('raw','clean','eyes','modes','clean_modes','valid','clean_valid','keys')}
        if cache['raw'].shape != cache['clean'].shape or cache['raw'].shape[1:] != (19,1024):
            raise ValueError('Klados cache axes or window length differ')
        if cache['eyes'].shape != (len(cache['raw']),2,1024):
            raise ValueError('Expected aligned HEOG/VEOG axes')
        for name in ('raw','clean','modes','clean_modes','valid','clean_valid'):
            cache[name] = cache[name][:, channels]
        if not all(np.isfinite(cache[name]).all() for name in ('raw','clean','eyes','modes','clean_modes')):
            raise ValueError('Nonfinite cached EEG or references')
        if cache['modes'].shape != (len(cache['raw']),7,3,1024):
            raise ValueError('Historical fixed K=3 mode axes differ')
        identities[partition] = sorted(set(int(key[0]) for key in cache['keys']))
        caches[partition] = cache
        files.append({'partition':partition,'sha256':digest,'bytes':path.stat().st_size,
                      'records':identities[partition], 'windows':len(cache['raw']),
                      'invalid_dirty_decompositions':int(np.count_nonzero(~cache['valid'].astype(bool))),
                      'invalid_clean_decompositions':int(np.count_nonzero(~cache['clean_valid'].astype(bool)))})
    for a,b in [('train','val'),('train','test'),('val','test')]:
        if set(identities[a]) & set(identities[b]):
            raise ValueError('Record leakage between partitions')
    checkpoint = parent/'eog_vmd_context_student/best_safe.pt'
    if sha256(checkpoint) != config['historical_checkpoint_sha256']:
        raise ValueError('Frozen model checkpoint differs')
    save_json(output/'data_contract.json', {'files':files, 'montage':montage,
        'record_partitions_disjoint':True, 'participant_partitions_verifiable':False,
        'record_id_definition':'0-based local export record index, not publisher file number',
        'historical_evaluation_development_exposed':True,
        'cache_preprocessing':'historical 200 Hz 0.5–40 Hz; no refiltering',
        'condition_labels_available':False, 'LEMON_used':False})
    return caches, checkpoint, montage


def historical_predict(model, raw, modes, eyes, valid):
    with torch.inference_mode():
        value = torch.from_numpy(np.asarray(raw, np.float32))[None]
        result = model(value, torch.from_numpy(np.asarray(modes, np.float32))[None],
            torch.from_numpy(np.asarray(eyes, np.float32))[None], torch.ones(value.shape[:2]),
            torch.full(value.shape[:2], 3, dtype=torch.long))
        artifact = result['artifact'][0].numpy()*np.asarray(valid)[:,None]
    return raw-artifact


def training_blocks(cache):
    # Scale each entire input channel/window; use that scale for its target.
    raw, clean = cache['raw'], cache['clean']
    scale = np.maximum(np.sqrt(np.mean(raw**2, axis=-1, keepdims=True)), 1e-8)
    return ((raw/scale).reshape(-1,1,512).astype(np.float32),
            (clean/scale).reshape(-1,1,512).astype(np.float32))


def train_paired(model, caches, output, config, device, name='bandroute', adapter=None):
    random.seed(config['seed'])
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config['seed'])
    make_blocks = training_blocks if adapter is None else adapter.training_blocks
    train_x, train_y = make_blocks(caches['train'])
    valid_x, valid_y = make_blocks(caches['val'])
    def predict(inputs):
        return model(inputs)['y_hat'] if adapter is None else adapter.forward_training(inputs)
    generator = torch.Generator().manual_seed(config['seed'])
    loader = DataLoader(TensorDataset(torch.from_numpy(train_x),torch.from_numpy(train_y)),
                        batch_size=config['bandroute_batch_size'], shuffle=True,
                        generator=generator, num_workers=0)
    validation = DataLoader(TensorDataset(torch.from_numpy(valid_x),torch.from_numpy(valid_y)),
                            batch_size=config['bandroute_batch_size'], shuffle=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'],
                                 weight_decay=config['weight_decay'])
    best, history = float('inf'), []
    began = time.perf_counter()
    for epoch in range(config['bandroute_epochs']):
        model.train()
        total, count = 0., 0
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = predict(inputs)
            loss = torch.mean((prediction-targets)**2)
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite '+name+' training loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            total += float(loss.detach())*len(inputs)
            count += len(inputs)
        model.eval()
        error, valid_count = 0., 0
        with torch.inference_mode():
            for inputs, targets in validation:
                inputs, targets = inputs.to(device), targets.to(device)
                loss = torch.mean((predict(inputs)-targets)**2)
                error += float(loss)*len(inputs)
                valid_count += len(inputs)
        value = error/valid_count
        history.append({'epoch':epoch+1,'train_mse':total/count,'validation_mse':value,
                        'elapsed_s':time.perf_counter()-began})
        state = {'model':model.state_dict(),'optimizer':optimizer.state_dict(),
                 'epoch':epoch+1,'validation_mse':value,'config':config,
                 'rng_state':torch.get_rng_state(),'shuffle_rng_state':generator.get_state()}
        torch.save(state, output/f'{name}_last.pt')
        if value < best:
            best = value
            torch.save(state, output/f'{name}_best.pt')
        pd.DataFrame(history).to_csv(output/f'{name}_training.csv', index=False)
        print(name.upper()+'_EPOCH_SAVED', epoch+1, 'train',total/count,'validation',value,
              'elapsed_s', round(time.perf_counter()-began,1), flush=True)
    checkpoint = torch.load(output/f'{name}_best.pt', map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    save_json(output/f'{name}_selection.json', {'best_epoch':checkpoint['epoch'],
        'validation_mse':checkpoint['validation_mse'],'selection_uses_test':False,
        'epochs_completed':len(history),'parameters':sum(p.numel() for p in model.parameters()),
        'source':'pinned author BRAANetV2; fs200 and explicit bands adapted' if adapter is None else
                 'pinned author EEGOARNET; pretrained initialization, custom seven+absent montage, fs128',
        'batch_size_adaptation':8, 'clean_target_normalization':'input RMS only'})


def mode_evidence(modes, eyes, raw, fs, method, window, names):
    rows = []
    for channel,name in enumerate(names):
        for index, mode in enumerate(modes[channel]):
            spectrum = np.abs(np.fft.rfft(mode))**2
            freq = np.fft.rfftfreq(len(mode), 1/fs)
            center = float(freq@spectrum/max(spectrum.sum(), 1e-30))
            associations = correlations(mode[None], eyes)[0]
            rows.append({'method':method,'window':window,'channel':name,'mode':index+1,
                         'center_hz':center,'heog_cc':associations[0],'veog_cc':associations[1],
                         'energy_fraction':float(np.sum(mode**2)/max(np.sum(raw[channel]**2),1e-30))})
    return rows


def run(input_root, output, source_root, config):
    if not Path('/kaggle/input').exists():
        raise RuntimeError('All numerical work must execute on Kaggle')
    output, source_root = Path(output), Path(source_root)
    caches, checkpoint, montage = load_caches(input_root, output, source_root, config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not torch.cuda.is_available() and not config.get('smoke'):
        raise RuntimeError('Full author-model training requires the requested Kaggle GPU')
    torch.manual_seed(config['seed'])
    sources = json.loads((source_root/'docs/provenance/author_sources.json').read_text())
    import tempfile
    author_root = Path(tempfile.mkdtemp(prefix='pinned-eog-authors-'))
    author_paths = {name:fetch_author_source(spec, author_root/name) for name,spec in sources.items()}
    save_json(output/'author_sources.json', sources)
    print('AUTHOR_SOURCES_VERIFIED', flush=True)
    historical = EOGContextGainStudent().eval()
    historical.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True)['model'])
    historical.inference_gate_mode, historical.inference_gate_offset = 'hard', 0.
    eegoar = EEGOAR(author_paths['eegoar'], montage['frontal_names'], device)
    save_json(output/'eegoar_channel_contract.json',eegoar.slot_contract)
    bandroute, _ = bandroute_model(author_paths['bandroute'], device)
    print('AUTHOR_MODELS_LOADED', 'eegoar_parameters',eegoar.parameters,
          'bandroute_parameters',sum(p.numel() for p in bandroute.parameters()), flush=True)
    if config.get('smoke'):
        for cache in caches.values():
            for name in cache:
                cache[name] = cache[name][:1]
    train_paired(eegoar.model, caches, output, config, device,name='eegoar',adapter=eegoar)
    eegoar.finetuned = True
    train_paired(bandroute, caches, output, config, device)
    training = caches['train']
    values = training['raw'].transpose(1,0,2).reshape(7,-1)
    eyes = training['eyes'].transpose(1,0,2).reshape(2,-1)
    normalized = eyes/np.maximum(eyes.std(axis=-1,keepdims=True),1e-12)
    activity = np.mean(normalized**2,axis=0)
    lower, upper = np.quantile(activity,[.3,.7])
    mwf = fit_mwf(values[:,activity>=upper],values[:,activity<=lower],
                  config['mwf_rank'],config['mwf_strength'])
    np.savez_compressed(output/'mwf_calibration.npz', operator=mwf.operator,mean=mwf.mean,
                        eigenvalues=mwf.eigenvalues)
    save_json(output/'mwf_calibration.json', {'records':'training only',
        'masks':'top/bottom 30% joint standardized EOG activity',
        'paired_target_used':False,'rank':config['mwf_rank'],'strength':config['mwf_strength'],
        'adaptation':'instantaneous regularized GEVD–MWF; no lag embedding'})
    rows, clean_rows, failures, times, evidence = [], [], [], [], []
    test = caches['test']
    for window,(record,start) in enumerate(test['keys']):
        raw, target, eyes = test['raw'][window],test['clean'][window],test['eyes'][window]
        directory = output/'windows'/f'record_{int(record):02d}_start_{int(start)}'
        directory.mkdir(parents=True,exist_ok=True)
        predictions = {'raw':raw,'clean':target,'eyes':eyes}
        np.savez_compressed(directory/'historical_modes.npz',modes=test['modes'][window],
            residual=raw-test['modes'][window].sum(axis=1),valid=test['valid'][window],
            clean_modes=test['clean_modes'][window],clean_valid=test['clean_valid'][window],
            clean_residual=target-test['clean_modes'][window].sum(axis=1))
        clean_predictions = {}
        for method in ['identity','direct_eog_regression','resampling_identity',*METHODS]:
            began = time.perf_counter()
            def predict(values, clean_input=False):
                if method == 'identity': return values.copy()
                if method == 'direct_eog_regression': return values-window_regression(values,eyes)
                if method == 'resampling_identity':
                    return resample_poly(resample_poly(values,16,25,axis=-1),25,16,axis=-1)[:,:values.shape[-1]]
                if method == 'gevd_mwf':return values-mwf.artifact(values)
                if method == 'eegoarnet_finetuned':return eegoar.predict(values)
                if method == 'bandroutenet':return bandroute_predict(bandroute,values,device)
                if method == 'historical_vmd':
                    modes = test['clean_modes' if clean_input else 'modes'][window]
                    valid = test['clean_valid' if clean_input else 'valid'][window]
                    return historical_predict(historical,values,modes,eyes,valid)
                solver = svmd_fcm if method == 'svmd_fcm_adaptation' else vme_gmetv
                settings = config['svmd' if method == 'svmd_fcm_adaptation' else 'vme_gmetv']
                result, diagnostics = [], []
                for channel,vector in enumerate(values):
                    cleaned, detail = solver(vector,fs=config['sampling_rate'],config=settings)
                    result.append(cleaned)
                    modes = detail.pop('modes')
                    residual = detail.pop('residual')
                    arrays = {name:detail.pop(name) for name in list(detail) if isinstance(detail[name],np.ndarray)}
                    np.savez_compressed(directory/f'{method}_{"clean" if clean_input else "dirty"}_{channel}.npz',
                                        modes=modes,residual=residual,**arrays)
                    diagnostics.append({'channel':montage['frontal_names'][channel], **detail})
                    if not clean_input:
                        evidence.extend(mode_evidence([modes],eyes,values[channel:channel+1],200,
                            method,window,[montage['frontal_names'][channel]]))
                save_json(directory/f'{method}_{"clean" if clean_input else "dirty"}_diagnostics.json', diagnostics)
                return np.asarray(result)
            try:
                prediction = predict(raw)
                clean_prediction = predict(target,True)
                if prediction.shape != raw.shape or not np.isfinite(prediction).all() or not np.isfinite(clean_prediction).all():
                    raise ValueError('Invalid prediction axes or nonfinite values')
                status = 'ok'
            except Exception as error:
                failures.append({'window':window,'record':int(record),'method':method,
                    'error':repr(error),'traceback':traceback.format_exc(),'fallback':'identity'})
                prediction, clean_prediction, status = raw.copy(),target.copy(),'failed_identity_fallback'
                print('METHOD_FAILED',method,window,type(error).__name__,str(error),flush=True)
            predictions[method] = prediction.astype(np.float32)
            clean_predictions[method] = clean_prediction.astype(np.float32)
            scores, baseline = paired(prediction,target),paired(raw,target)
            associations = correlations(prediction,eyes)
            for channel,name in enumerate(montage['frontal_names']):
                rows.append({'method':method,'record':int(record),'start':int(start),'channel':name,
                    'status':status,'snr_out_db':scores['snr_out_db'][channel],
                    'snr_improvement_db':scores['snr_out_db'][channel]-baseline['snr_out_db'][channel],
                    'pooled_window_snr_db':scores['pooled_window_snr_db'],
                    'pooled_snr_improvement_db':scores['pooled_window_snr_db']-baseline['pooled_window_snr_db'],
                    **{name:float(scores[name][channel]) for name in ['mse','rmse','mae','rrmse_t','rrmse_s','pearson_cc']},
                    'heog_cc':associations[channel,0],'veog_cc':associations[channel,1]})
            clean_rows.append({'method':method,'record':int(record),'start':int(start),
                               'status':status,**preservation(clean_prediction,target)})
            times.append({'window':window,'method':method,
                          'dirty_and_clean_runtime_s':time.perf_counter()-began,
                          'historical_cached_vmd_cost_excluded':method=='historical_vmd',
                          'inference_device':'cpu' if method in ('historical_vmd','svmd_fcm_adaptation',
                              'vme_gmetv_adaptation','gevd_mwf','direct_eog_regression','identity',
                              'resampling_identity') else str(device)})
            pd.DataFrame(rows).to_csv(output/'per_channel_window.csv',index=False)
            pd.DataFrame(clean_rows).to_csv(output/'clean_preservation.csv',index=False)
            pd.DataFrame(times).to_csv(output/'runtime.csv',index=False)
            save_json(output/'failures.json',failures)
            print('METHOD_SAVED',window,method,'snr',scores['pooled_window_snr_db'],status,flush=True)
        evidence.extend(mode_evidence(test['modes'][window],eyes,raw,200,'historical_vmd',window,montage['frontal_names']))
        np.savez_compressed(directory/'predictions.npz',**predictions)
        np.savez_compressed(directory/'clean_input_predictions.npz',**clean_predictions)
        pd.DataFrame(evidence).to_csv(output/'mode_reference_correlations.csv',index=False)
        save_json(directory/'completion.json',{'complete':True,'record':int(record),'start':int(start)})
    frame = pd.DataFrame(rows)
    # First collapse duplicated pooled window fields across channels, then
    # windows within each record, then records equally. No addition of dB.
    metric_names=['snr_out_db','snr_improvement_db','pooled_window_snr_db',
                  'pooled_snr_improvement_db','mse','rmse','mae','rrmse_t','rrmse_s',
                  'pearson_cc','heog_cc','veog_cc']
    records = frame.groupby(['method','record'],as_index=False)[metric_names].mean()
    records.to_csv(output/'per_record.csv',index=False)
    records.groupby('method',as_index=False)[metric_names].mean().to_csv(output/'comparison.csv',index=False)
    save_json(output/'numerical_completion.json',{'complete':True,'methods':METHODS,
        'test_windows':len(test['raw']),'records':len(set(test['keys'][:,0].tolist())),
        'failed_method_windows':len(failures),'smoke':config.get('smoke',False),
        'production_qualified':False,'regional_mapping_assumption':montage})
