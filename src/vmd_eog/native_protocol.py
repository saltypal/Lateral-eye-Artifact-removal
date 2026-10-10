"""Complete-trial native evaluation; adapted chance analysis is labelled."""
from fractions import Fraction
import json
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
from scipy.signal import resample_poly

from .io import atomic_json, unpack_source, write_jsonl, sha256_file
from .paper_metrics import pearson, ratio, resting_spectrum, EEGOAR_BANDS
from .paper_report import comparison_family


CONDITIONS = {1: "rest", 2: "lateral", 3: "vertical", 4: "blink"}
METHODS = ("identity", "direct", "regional_mwf", "regional_ica")


class ConditionMoments:
    """Merge trial moments to match Pearson on concatenated conditions.

    Merge centered moments rather than subtracting large raw sums. Trial
    offsets contribute to the concatenated correlation, as in author code.
    No filtering, spatial fitting or Welch segment crosses a trial join.
    """
    def __init__(self, channels):
        self.count = 0
        self.mean_x = np.zeros(channels)
        self.mean_y = np.zeros(channels)
        self.xx = np.zeros(channels)
        self.yy = np.zeros(channels)
        self.xy = np.zeros(channels)
        self.squared_error = np.zeros(channels)

    def update(self, first, second):
        first, second = np.asarray(first, float), np.asarray(second, float)
        if first.ndim != 2 or first.shape != second.shape or first.shape[0] != len(self.mean_x):
            raise ValueError("Moment inputs must have matching channel/sample axes")
        if first.shape[-1] < 2 or not np.isfinite(first).all() or not np.isfinite(second).all():
            raise ValueError("Moment inputs must contain finite samples")
        size = first.shape[-1]
        mean_x, mean_y = first.mean(-1), second.mean(-1)
        x, y = first-mean_x[:, None], second-mean_y[:, None]
        delta_x, delta_y = mean_x-self.mean_x, mean_y-self.mean_y
        total = self.count+size
        cross_weight = self.count*size/total
        self.xx += (x*x).sum(-1)+delta_x**2*cross_weight
        self.yy += (y*y).sum(-1)+delta_y**2*cross_weight
        self.xy += (x*y).sum(-1)+delta_x*delta_y*cross_weight
        self.mean_x += delta_x*size/total
        self.mean_y += delta_y*size/total
        self.squared_error += ((first-second)**2).sum(-1)
        self.count = total

    def correlation(self):
        return ratio(self.xy, np.sqrt(self.xx*self.yy))

    def rmse(self):
        if not self.count:
            return np.full_like(self.mean_x, np.nan)
        return np.sqrt(self.squared_error/self.count)


def multi_overlap_add(eeg, references, estimator, methods=METHODS, window=1024, hop=512):
    """Estimate all methods once per aligned window and subtract once."""
    eeg, references = np.asarray(eeg), np.asarray(references)
    if eeg.ndim != 2 or references.shape != (2, eeg.shape[-1]) or not 0 < hop <= window:
        raise ValueError("Need aligned EEG/HEOG/VEOG and a valid window/hop")
    if eeg.shape[-1] < 2 or not np.isfinite(eeg).all() or not np.isfinite(references).all():
        raise ValueError("Invalid trial signal")
    pad = window//2
    padding = ((0, 0), (pad, pad+window))
    padded = np.pad(eeg, padding, mode="reflect")
    ref = np.pad(references, padding, mode="reflect")
    artifacts = {name: np.zeros_like(padded, dtype=float) for name in methods}
    denominator = np.zeros(padded.shape[-1])
    weight = np.hanning(window+2)[1:-1]
    for start in range(0, padded.shape[-1]-window+1, hop):
        estimates = estimator(padded[:, start:start+window], ref[:, start:start+window])
        if set(estimates) != set(methods):
            raise ValueError("Estimator returned an incomplete comparison family")
        for name in methods:
            artifact = np.asarray(estimates[name])
            if artifact.shape != (eeg.shape[0], window) or not np.isfinite(artifact).all():
                raise ValueError("Invalid artifact estimate")
            artifacts[name][:, start:start+window] += artifact*weight
        denominator[start:start+window] += weight
    valid = slice(pad, pad+eeg.shape[-1])
    if (denominator[valid] <= 0).any():
        raise RuntimeError("Uncovered original samples")
    return {name: eeg-value[:, valid]/denominator[valid] for name, value in artifacts.items()}


def eight_second_bank(trials, fs=200):
    """Non-overlapping contiguous banks; short trials never join to make a bank."""
    size = int(8*fs)
    windows, identities = [], []
    for trial_id, values in trials:
        values = np.asarray(values)
        if values.ndim != 2 or not np.isfinite(values).all():
            raise ValueError("Invalid bank trial")
        for start in range(0, values.shape[-1]-size+1, size):
            windows.append(values[:, start:start+size].copy())
            identities.append({"trial_id": int(trial_id), "start": start, "samples": size})
    return windows, identities


def evaluation_references(item, raw_indices, fs=200):
    """Preserve source LPF derivatives; raw fallback is an explicit adaptation."""
    normalized = [name.upper().replace("_", "").replace("-", "") for name in item["names"]]
    matches = [[i for i, name in enumerate(normalized) if name == wanted]
               for wanted in ("HEOGLPF", "VEOGLPF")]
    if all(len(values) == 1 for values in matches):
        indices, status = [values[0] for values in matches], "publisher HEOG_lpf/VEOG_lpf"
    else:
        indices, status = raw_indices, "raw HEOG/VEOG fallback; not author LPF replication"
    values = np.asarray(item["data"][:, indices], float)
    if item["fs"] != fs:
        factor = Fraction(fs/float(item["fs"])).limit_denominator(10000)
        values = resample_poly(values, factor.numerator, factor.denominator, axis=-1)
    return values, {"status": status, "channels": [item["names"][i] for i in indices],
                    "processing": "resampling only; no additional derivative/filter invented"}


def chance_bootstrap(records, output, seed=42, repetitions=5000, participants_per_draw=5):
    """EEGOAR-inspired whole-montage null, not the unavailable four-region replica."""
    generator = np.random.default_rng(seed)
    thresholds = []
    draws = []
    pair_cache = {}
    for condition in ("lateral", "vertical", "blink"):
        eligible = {}
        for index, record in enumerate(records):
            if record["rest"] and record[condition]:
                eligible.setdefault(record["participant"], []).append(index)
        participants = sorted(eligible)
        if len(participants) < participants_per_draw:
            thresholds.append({"condition": condition, "status": "unavailable: fewer than five eligible participants",
                               "participants": len(participants)})
            continue
        distribution = np.empty(repetitions)
        for iteration in range(repetitions):
            selected = generator.choice(participants, participants_per_draw, replace=False)
            values = []
            for participant in selected:
                index = int(generator.choice(eligible[str(participant)]))
                record = records[index]
                rest_index = int(generator.integers(len(record["rest"])))
                eye_index = int(generator.integers(len(record[condition])))
                key = (index, condition, rest_index, eye_index)
                if key not in pair_cache:
                    rest = record["rest"][rest_index]
                    ocular = record[condition][eye_index]
                    ref_index = 0 if condition == "lateral" else 1
                    reference = np.broadcast_to(ocular[ref_index], rest.shape)
                    pair_cache[key] = float(np.abs(pearson(rest, reference)).mean())
                values.append(pair_cache[key])
                draws.append({"condition": condition, "iteration": iteration, "participant": str(participant),
                              "record_id": record["record_id"], "rest_bank": rest_index, "ocular_bank": eye_index})
            distribution[iteration] = np.mean(values)
        finite = np.isfinite(distribution).all()
        np.save(output/("chance_distribution_"+condition+".npy"), distribution)
        thresholds.append({"condition": condition, "status": "computed adapted null" if finite else "unavailable: undefined correlation",
                           "participants": len(participants), "repetitions": repetitions,
                           "p95": float(np.quantile(distribution, .95)) if finite else None})
    pd.DataFrame(draws).to_csv(output/"chance_draws.csv.gz", index=False)
    atomic_json(output/"native_chance_summary.json", {
        "conditions": thresholds, "participants_per_draw": participants_per_draw, "seed": seed,
        "window_seconds": 8, "aggregation": "mean absolute channel Pearson -> mean of five distinct participants",
        "selection": "participant without replacement; session then eligible non-overlapping bank uniformly",
        "exact_paper_replication": False, "equivalence_claim": False,
        "limitations": "Four-region assignment and exact five-person aggregation unavailable; development-only null."})
    return thresholds


def run_native_protocol(input_root, output, config, profile):
    """Run original-source pilot/full evaluation using immutable search recipes."""
    from .artifacts import verify_parent
    from .corpus import unique_parent
    from .data import eog_axes
    from .experiments import corpus_parent, grouped_selection, recipe_only, frontal_artifact, posterior_artifact
    from .osf_reader import read_osf
    from .signal import preprocess

    parent, rows = corpus_parent(input_root)
    vmd = unique_parent(input_root, "vmd_summary.json").parent
    posterior = unique_parent(input_root, "posterior_summary.json").parent
    verify_parent(vmd, ("vmd_search.csv", "vmd_grouped_candidates.csv.gz"))
    verify_parent(posterior, ("posterior_search.csv", "posterior_scores.csv"))
    folds = {r["recipient"]: r["partition"]["fold"] for r in rows if r["target_kind"] == "controlled_recipient_reference"}
    front = grouped_selection(pd.read_csv(vmd/"vmd_search.csv"), pd.read_csv(vmd/"vmd_grouped_candidates.csv.gz"), folds, config["preservation"])
    post = grouped_selection(pd.read_csv(posterior/"posterior_search.csv"), pd.read_csv(posterior/"posterior_scores.csv"), folds, config["preservation"])
    atomic_json(output/"native_recipe_selections.json", {"frontal": front, "posterior": post})
    source = unpack_source(input_root, tempfile.mkdtemp(prefix="eog-native-"), output)
    ledger = sorted((parent/"calibration").glob("*/record.json"))
    ledger = [path for path in ledger if json.loads(path.read_text())["dataset"] == "osf"]
    if profile == "pilot":
        ledger = ledger[:2]
    if not ledger:
        raise ValueError("No development OSF recordings in the frozen corpus")
    write_jsonl(output/"native_source_ledger.jsonl", [json.loads(path.read_text()) for path in ledger])
    metrics, spectra, failures, bank_records = [], [], [], []
    predictions = output/"native_trial_predictions"
    predictions.mkdir()
    fits = {}
    trial_count = 0
    for record_path in ledger:
        record = json.loads(record_path.read_text())
        if record["partition"]["role"] != "development":
            raise ValueError("Reserved native recording encountered")
        path = (source/record["path"]).resolve()
        if not path.is_relative_to(source.resolve()) or sha256_file(path) != record["source_sha256"]:
            raise ValueError("Native source differs from frozen ledger")
        item = read_osf(path)
        raw_refs = eog_axes(item)
        eval_refs, ref_identity = evaluation_references(item, raw_refs, config["fs"])
        verify_parent(parent, (str((record_path.parent/"calibration.npz").relative_to(parent)), str(record_path.relative_to(parent))))
        calibration = dict(np.load(record_path.parent/"calibration.npz", allow_pickle=False))
        names, regions = calibration["channel_names"], calibration["regions"]
        actual_names = np.asarray([item["names"][i] for i in item["eeg_indices"]])
        if not np.array_equal(names, actual_names):
            raise ValueError("Full-trial montage differs from calibration")
        fold = str(record["partition"]["fold"])
        direct = recipe_only(front[fold]["direct"], "frontal") if "direct" in front[fold] else None
        vmd_recipe = recipe_only(front[fold]["vmd_projected"], "frontal") if "vmd_projected" in front[fold] else None
        post_recipes = {method: recipe_only(post[fold][method], "posterior") if method in post[fold] else None
                        for method in ("mwf", "ica")}
        frontal_ids = np.flatnonzero(regions == 0)
        common_ids = np.flatnonzero(~np.isin(regions, [0, 1]))
        moments = {}
        banks = {"participant": record["participant"], "record_id": record["record_id"],
                 "reference_identity": ref_identity, "bank_identity": {}, **{condition: [] for condition in CONDITIONS.values()}}
        for trial_id in record["scoring_trial_ids"]:
            condition = CONDITIONS.get(item["trial_labels"][trial_id])
            if condition is None:
                failures.append({"record_id": record["record_id"], "trial_id": trial_id, "failure": "unknown condition", "status": "ineligible"})
                continue
            values = preprocess(item["data"][trial_id, item["eeg_indices"]+raw_refs], item["fs"])
            eeg, refs = values[:-2], values[-2:]
            scoring_ref = eval_refs[trial_id]
            if scoring_ref.shape[-1] != eeg.shape[-1]:
                raise ValueError("Evaluation reference sample alignment differs")

            def estimate(window_eeg, window_refs):
                data = {**calibration, "eeg": window_eeg, "references": window_refs}
                direct_art, direct_diag = frontal_artifact(data, direct, config)
                front_art, front_diag = frontal_artifact(data, vmd_recipe, config, frontal_ids)
                common = front_art.copy()
                common[common_ids] = direct_art[common_ids]
                corrections = {"identity": np.zeros_like(window_eeg), "direct": direct_art}
                for method, recipe in post_recipes.items():
                    artifact, diagnostic = posterior_artifact(data, calibration, recipe, config, fits, record["record_id"])
                    corrections["regional_"+method] = common+artifact
                    for issue in front_diag+[diagnostic]:
                        if issue.get("failure"):
                            failures.append({"record_id": record["record_id"], "trial_id": trial_id,
                                             "method": "regional_"+method, **issue})
                for issue in direct_diag:
                    if issue.get("failure"):
                        failures.append({"record_id": record["record_id"], "trial_id": trial_id, "method": "direct", **issue})
                return corrections

            cleaned = multi_overlap_add(eeg, refs, estimate, window=config["window"], hop=config["hop"])
            np.savez_compressed(predictions/f"{record['record_id']}-trial-{trial_id:04d}.npz",
                                original=eeg, evaluation_references=scoring_ref, inference_references=refs,
                                channel_names=names, regions=regions, **cleaned)
            for method, result in cleaned.items():
                key = (condition, method)
                moments.setdefault(key, ConditionMoments(len(names)))
                second = eeg if condition == "rest" else np.broadcast_to(scoring_ref[0 if condition == "lateral" else 1], eeg.shape)
                moments[key].update(result, second)
                if condition == "rest":
                    spectrum = resting_spectrum(result, eeg, config["fs"], normalization_band=(1., 40.))
                    segments = 1+(eeg.shape[-1]-2*config["fs"])//config["fs"]
                    spectra.append({"record_id": record["record_id"], "method": method,
                                    "segments": segments, "spectrum": spectrum})
            bank, identities = eight_second_bank([(trial_id, eeg if condition == "rest" else scoring_ref)], config["fs"])
            banks[condition].extend(bank)
            banks["bank_identity"].setdefault(condition, []).extend(identities)
            trial_count += 1
            print("NATIVE_TRIAL", record["record_id"], trial_id, condition, eeg.shape[-1], flush=True)
        bank_records.append(banks)
        for (condition, method), summary in moments.items():
            before = moments[(condition, "identity")]
            for channel, name in enumerate(names):
                metrics.append({"source": record["participant"], "record_id": record["record_id"], "dataset": "osf",
                    "condition": condition, "method": method, "channel": str(name), "samples": summary.count,
                    "rest_rmse": float(summary.rmse()[channel]) if condition == "rest" else np.nan,
                    "eeg_eog_signed_r_after": float(summary.correlation()[channel]) if condition != "rest" else np.nan,
                    "eeg_eog_abs_r_after": float(abs(summary.correlation()[channel])) if condition != "rest" else np.nan,
                    "eeg_eog_abs_r_before": float(abs(before.correlation()[channel])) if condition != "rest" else np.nan,
                    "reference_status": ref_identity["status"], "scoring_unit": "complete condition concatenation",
                    "units": "original EEG source array units; microvolt scale unverified"})
        pd.DataFrame(metrics).to_csv(output/"native_condition_channels.csv", index=False)
        write_jsonl(output/"native_failures.jsonl", failures)
    bank_metadata = [{key: value for key, value in record.items() if key not in CONDITIONS.values()} for record in bank_records]
    atomic_json(output/"native_bank_inventory.json", bank_metadata)
    chance_bootstrap(bank_records, output, config["seed"])
    write_native_reports(pd.DataFrame(metrics), spectra, output)
    atomic_json(output/"native_protocol_summary.json", {
        "profile": profile, "recordings": len(ledger), "scoring_trials": trial_count, "failure_count": len(failures),
        "reserved_confirmation_opened": False, "native_reconstruction_snr_claim": False,
        "condition_concatenation": True, "trial_join_filtering": False,
        "reference_statuses": sorted({record["reference_identity"]["status"] for record in bank_records}),
        "paper_four_region_replication": False, "chance_analysis": "declared whole-montage adaptation",
        "model_authorization": False})


def write_native_reports(frame, spectra, output):
    strict_mean = lambda values: np.asarray(values, float).mean()
    fields = ["rest_rmse", "eeg_eog_abs_r_after", "eeg_eog_abs_r_before"]
    records = frame.groupby(["source", "record_id", "condition", "method"])[fields].agg(strict_mean).reset_index()
    participants = records.groupby(["source", "condition", "method"])[fields].agg(strict_mean).reset_index()
    participants["dataset"] = "osf"
    participants.to_csv(output/"native_condition_participants.csv", index=False)
    tests = []
    for condition, values in participants.groupby("condition"):
        metric = "rest_rmse" if condition == "rest" else "eeg_eog_abs_r_after"
        tests.append(comparison_family(values, "source", ["dataset", "condition"], [metric]))
    tests = pd.concat(tests, ignore_index=True) if tests else pd.DataFrame()
    if len(tests):
        tests["comparison_family_size"] = len(tests)
        if "p_two_sided" in tests:
            tests["p_bonferroni"] = (tests.p_two_sided*len(tests)).clip(upper=1.)
    tests.to_csv(output/"native_condition_permutations.csv", index=False)
    directory = output/"native_rest_spectra"
    directory.mkdir()
    for record_id, method in sorted({(entry["record_id"], entry["method"]) for entry in spectra}):
        entries = [entry for entry in spectra if (entry["record_id"], entry["method"]) == (record_id, method)]
        weights = np.asarray([entry["segments"] for entry in entries], float)
        frequencies = entries[0]["spectrum"]["frequencies_hz"]
        before = np.average([entry["spectrum"]["psd_before"] for entry in entries], axis=0, weights=weights)
        after = np.average([entry["spectrum"]["psd_after"] for entry in entries], axis=0, weights=weights)
        powers = {}
        total = (frequencies >= 1) & (frequencies < 40)
        for band, (lo, hi) in EEGOAR_BANDS.items():
            mask = (frequencies >= lo) & (frequencies < hi)
            powers[band+"_relative_before"] = ratio(before[:, mask].sum(-1), before[:, total].sum(-1))
            powers[band+"_relative_after"] = ratio(after[:, mask].sum(-1), after[:, total].sum(-1))
        np.savez_compressed(directory/f"{record_id}-{method}.npz", frequencies_hz=frequencies,
                            psd_before=before, psd_after=after, absolute_psd_change=np.abs(before-after), **powers)
