"""Evidence-based frontal closeout when requested pretrained releases are unverified.

This module does not implement substitute architectures or fabricate predictions.
It verifies completed parents and makes release eligibility visible beside them.
"""
import json
import hashlib
from pathlib import Path
import shutil

from .data import save_json, sha256


def discovery_identity(path):
    """Hash JSON content independently of Git's Windows/Linux newline conversion."""
    value = json.loads(Path(path).read_text(encoding='utf-8'))
    canonical = json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True)
    return hashlib.sha256(canonical.encode('utf-8')).hexdigest()


def verified_table_parent(input_root, run_id, source_sha):
    matches = []
    for path in Path(input_root).rglob('artifact_manifest.json'):
        manifest = json.loads(path.read_text())
        if manifest.get('run_id') == run_id:
            matches.append((path, manifest))
    if len(matches) != 1:
        raise ValueError('Require one exact parent: ' + run_id)
    path, manifest = matches[0]
    if not manifest.get('complete') or manifest.get('git_sha') != source_sha:
        raise ValueError('Parent completion or source SHA differs: ' + run_id)
    entries = {entry['path']: entry for entry in manifest['files']}

    def verify(name):
        target = path.parent / name
        entry = entries[name]
        if target.stat().st_size != entry['bytes'] or sha256(target) != entry['sha256']:
            raise ValueError('Parent payload differs: ' + name)
        return target

    return verify


def validate_blocked_candidates(review):
    """Never treat absent checkpoint identity or search absence as successful inference."""
    if review.get('training_allowed') is not False:
        raise ValueError('This extension is pretrained-only')
    candidates = review['candidates']
    if {entry['model'] for entry in candidates} != {'EEG-WNet', 'Diff-ADN'}:
        raise ValueError('Unexpected replacement model or missing candidate')
    for entry in candidates:
        if entry['eligible'] or entry['status'] != 'not run—pretrained reproduction unavailable':
            raise ValueError('A newly eligible release needs a verified inference adapter before execution')
        if entry['checkpoint_sha256'] is not None or not entry['reason']:
            raise ValueError('Blocked candidate must have no invented checkpoint and an explicit reason')
    return candidates


def run(input_root, source_root, output):
    if not Path('/kaggle/input').exists():
        raise RuntimeError('Numerical reporting executes on Kaggle only')
    import matplotlib.pyplot as plt
    import pandas as pd
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    source_root = Path(source_root)
    review = json.loads((source_root / 'configs/frontal_pretrained.json').read_text())
    candidates = validate_blocked_candidates(review)
    discovery_path = source_root / 'docs/provenance/frontal_pretrained_discovery.json'
    if review.get('discovery_hash_encoding') != 'canonical-json-sort-keys-ascii-compact':
        raise ValueError('Unknown discovery identity encoding')
    if discovery_identity(discovery_path) != review['discovery_sha256']:
        raise ValueError('Release discovery snapshot changed without review')
    shutil.copy2(discovery_path, output / discovery_path.name)
    save_json(output / 'pretrained_eligibility.json', review)
    pd.DataFrame(candidates).to_csv(output / 'pretrained_eligibility.csv', index=False)
    for parent in review['parents']:
        verify = verified_table_parent(input_root, parent['run_id'], parent['git_sha'])
        for name in parent['tables']:
            shutil.copy2(verify(name), output / name)
    paired = pd.read_csv(output / 'comparison.csv')
    native = pd.read_csv(output / 'native_frontal_before_after_summary.csv')
    rest = pd.read_csv(output / 'native_rest_participant_summary.csv')
    contract = json.loads((output / 'data_contract.json').read_text())
    if contract['montage']['frontal_names'] != ['FP1', 'FP2', 'F3', 'F4', 'F7', 'F8', 'Fz']:
        raise ValueError('Frontal label contract differs')
    if not contract['record_partitions_disjoint'] or contract['LEMON_used']:
        raise ValueError('Dataset boundary differs')
    reference_audit = json.loads((output / 'native_reference_audit.json').read_text())
    if not reference_audit['before_after_prediction_table_matches_saved_signals']:
        raise ValueError('Native reference audit failed')
    figures = output / 'figures'
    figures.mkdir(exist_ok=True)
    paths = []

    def save(figure, name):
        for suffix in ['png', 'pdf']:
            figure.savefig(figures / (name + '.' + suffix), dpi=160, bbox_inches='tight')
        paths.append(str(figures / (name + '.png')))
        plt.close(figure)

    table = paired.set_index('method').sort_values('pooled_window_snr_db')
    figure, axis = plt.subplots(figsize=(11, 6))
    axis.barh(table.index, table.pooled_window_snr_db, color='#2563a8')
    axis.set(xlabel='Record-balanced pooled-window output SNR (dB)',
             title='Completed Klados frontal results — carried forward, no new model inference')
    axis.grid(axis='x', alpha=.2)
    save(figure, '01_existing_frontal_paired_results')
    lateral = native[native.condition == 'lateral'].set_index('method')
    blink = native[native.condition == 'blink'].set_index('method')
    resting = rest.set_index('method')
    figure, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    # The resampling-only control exists in paired Klados, not in native OSF.
    methods = [method for method in table.index if method != 'resampling_identity']
    if set(methods) != set(lateral.index) or set(methods) != set(blink.index):
        raise ValueError('Native method coverage differs from completed paired methods')
    values = [blink.loc[methods, 'veog_abs_cc_after'], lateral.loc[methods, 'heog_abs_cc_after'],
              100 * resting.loc[methods, 'participant_mean_relative_change']]
    for axis, values, title in zip(axes, values,
        ['Blink: mean absolute VEOG CC', 'Lateral: mean absolute HEOG CC', 'Rest: relative RMS modification (%)']):
        axis.barh(methods, values, color='#2563a8')
        axis.set_title(title)
        axis.grid(axis='x', alpha=.2)
    axes[0].axvline(float(blink.loc['identity', 'veog_abs_cc_before']), color='black', linestyle='--', label='Before')
    axes[1].axvline(float(lateral.loc['identity', 'heog_abs_cc_before']), color='black', linestyle='--', label='Before')
    axes[2].axvline(1, color='black', linestyle='--', label='1% engineering threshold')
    for axis in axes:
        axis.legend(fontsize=8)
    figure.suptitle('Existing native OSF frontal transfer: suppression and preservation together')
    figure.tight_layout()
    save(figure, '02_existing_native_tradeoffs')
    save_json(output / 'completion.json', {'complete': True, 'new_models_run': [],
        'new_predictions_created': False, 'training_performed': False,
        'blocked_candidates': [entry['model'] for entry in candidates],
        'regions_processed': ['frontal'], 'parent_results_recomputed': False,
        'native_preservation_resolved': False, 'production_qualified': False,
        'parents': review['parents'], 'discovery_sha256': review['discovery_sha256']})
    text = '# Frontal pretrained extension closeout\n\n'
    text += 'Both requested models are not run: no compatible public pretrained release was verified. '
    text += 'Search failures do not prove that weights do not exist. No new denoising, training, '
    text += 'architecture substitute or regional integration occurred.\n\n'
    text += pd.DataFrame(candidates)[['model', 'status', 'reason']].to_markdown(index=False)
    text += '\n\nExisting paired Klados and native OSF results are copied from exact verified parents. '
    text += 'They are not results from EEG-WNet or Diff-ADN. Native resting preservation remains failed. '
    text += 'No native reconstruction SNR is claimed. The frontal experiment campaign is closed with limitations; '
    text += 'a deployment-qualified frontal system has not been achieved.\n'
    (output / 'RESULTS.md').write_text(text, encoding='utf-8')
    print('PRETRAINED_REVIEW_COMPLETE; NEW_MODELS_RUN=0; BLOCKED=2', flush=True)
    return paths
