"""Fixed-recipe frontal research with source-excluded selection and saved shards.

Imported caches are passive artifacts. This module never imports AutoVMD,
invokes a selector, or decomposes a posterior/shared/unknown channel.
"""
import itertools
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import welch
from scipy.stats import skew, kurtosis
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .artifacts import verify_parent
from .contracts import canonical_hash
from .experiments import corpus_parent
from .frontal_methods import (FixedRecipe, frontal_indices, decompose_frontal,
    estimate_components, prepare_sobi, author_selector_adaptation)
from .io import atomic_json, read_jsonl, sha256_file, write_jsonl
from .metrics import paired
from .paper_evaluation import save_protocol
from .paper_metrics import paired_channels
from .posterior_context import select_examples
from .reference import correlations, project
from .frontal_review import strict_mean
from .sobi import approximate_entropy


def finite_value(value):
    """Undefined metrics remain null with coverage counts, never fabricated zeros."""
    return float(value) if np.isfinite(value) else None


def detector_features(values, fs, calibration_scale):
    """Declared subset of the 2024 feature families, at project window length.

    RBF SVM C=1 and gamma=scale; no mRMR or unspecified nonlinear features.
    Consequently this is explicitly a fixed-parameter adaptation.
    """
    values = np.asarray(values, dtype=float)
    if not np.isfinite(calibration_scale) or calibration_scale <= 0:
        raise ValueError("Detector needs positive unscored same-unit calibration SD")
    centered = values - values.mean()
    scale = np.sqrt(np.mean(centered ** 2))
    unit = centered / scale if scale > 0 else np.zeros_like(centered)
    first, second = np.diff(unit), np.diff(unit, n=2)
    variance = np.var(unit)
    first_variance = np.var(first)
    mobility = np.sqrt(first_variance / variance) if variance > 0 else 0.
    complexity = np.sqrt(np.var(second) / first_variance) / mobility if first_variance > 0 and mobility > 0 else 0.
    frequencies, power = welch(unit, fs=fs, nperseg=min(len(unit), 400), noverlap=min(200, len(unit)//2))
    total = power.sum()
    bands = [power[(frequencies >= lo) & (frequencies < hi)].sum() / total if total > 0 else 0.
             for lo, hi in ((.5, 4), (4, 8), (8, 13), (13, 30), (30, 40))]
    asymmetry = float(skew(unit)) if scale > 0 else 0.
    peakedness = float(kurtosis(unit, fisher=False)) if scale > 0 else 0.
    return np.asarray([np.log(max(scale / calibration_scale, 1e-30)), asymmetry, peakedness,
                       variance, mobility, complexity, *bands])


def fit_detectors(parent, rows, config):
    features, labels, folds, source_rows = [], [], [], []
    for row in rows:
        with np.load(parent / row["calibration"], allow_pickle=False) as calibration:
            calibration_eeg = calibration["eeg"].copy()
        with np.load(parent / row["array_path"], allow_pickle=False) as data:
            if calibration_eeg.shape[0] != data["eeg"].shape[0]:
                raise ValueError("Detector calibration channel axis differs")
            ids = frontal_indices(data["eeg"], data["channel_names"], data["regions"], data["mask"])
            for index in ids:
                features.append(detector_features(data["eeg"][index], config["fs"], calibration_eeg[index].std()))
                labels.append(int(row["condition"] != "clean"))
                folds.append(row["partition"]["fold"])
                source_rows.append(row)
    features, labels, folds = np.asarray(features), np.asarray(labels), np.asarray(folds)
    detectors, evidence = {}, []
    for fold in sorted(set(folds)):
        train = folds != fold
        train_rows = [row for row, use in zip(source_rows, train) if use]
        heldout = [row for row, use in zip(source_rows, train) if not use]
        for identity in ("recipient", "donor"):
            if {row[identity] for row in train_rows} & {row[identity] for row in heldout}:
                raise ValueError("SVM fitting leaks " + identity)
        if set(labels[train]) != {0, 1}:
            raise ValueError("Frontal detector training needs clean and ocular conditions")
        classifier = make_pipeline(StandardScaler(), SVC(C=1., gamma="scale", class_weight="balanced"))
        classifier.fit(features[train], labels[train])
        detectors[int(fold)] = classifier
        evidence.append({"fold": int(fold), "training_recipients": sorted({r["recipient"] for r in train_rows}),
            "training_donors": sorted({r["donor"] for r in train_rows}),
            "heldout_recipients": sorted({r["recipient"] for r in heldout}),
            "heldout_donors": sorted({r["donor"] for r in heldout}),
            "features": "log input/calibration SD ratio, skewness, kurtosis, Hjorth activity/mobility/complexity, five relative PSD bins",
            "amplitude_scaling": "unscored same-channel calibration SD; no scoring target scaling",
            "labels": "declared clean versus ocular mixture condition, not target reconstruction values",
            "mrmr": False, "C": 1., "gamma": "scale", "window_samples": config["window"],
            "paper_replication": False})
    return detectors, evidence


def uncertainty_report(oof, output, config, repetitions=2000):
    """Paired recipient/donor multiplier bootstrap; project statistical analysis.

    Source means are condition/input balanced first. Independent Poisson(1)
    recipient and donor weights acknowledge both reused source populations;
    this is an explicitly labelled approximation, not a paper-native test.
    """
    dirty = oof[oof.condition != "clean"]
    columns = ["example_id", "channel", "snr_energy_db"]
    comparator = dirty[dirty.evaluated_family == "direct"][columns]
    comparator = comparator.rename(columns={"snr_energy_db": "comparator_snr"})
    paired_scores = dirty.merge(comparator, on=["example_id", "channel"], validate="many_to_one")
    paired_scores["snr_delta"] = paired_scores.snr_energy_db - paired_scores.comparator_snr
    units = paired_scores.groupby(["evaluated_family", "recipient", "donor", "condition", "input_snr_db"])[
        ["snr_energy_db", "snr_delta"]].agg(strict_mean).reset_index()
    recipients, donors = sorted(units.recipient.unique()), sorted(units.donor.unique())
    recipient_map = {value: index for index, value in enumerate(recipients)}
    donor_map = {value: index for index, value in enumerate(donors)}
    generator = np.random.default_rng(config["seed"])
    recipient_weights = generator.poisson(1., size=(repetitions, len(recipients)))
    donor_weights = generator.poisson(1., size=(repetitions, len(donors)))
    reports = []
    for family, group in units.groupby("evaluated_family"):
        rindex = np.asarray([recipient_map[value] for value in group.recipient])
        dindex = np.asarray([donor_map[value] for value in group.donor])
        weights = recipient_weights[:, rindex] * donor_weights[:, dindex]
        # Each recipient contributes equal weight even if its eligible source
        # combinations differ. Missing bootstrap mass remains an invalid draw.
        counts = group.groupby("recipient").size()
        weights = weights / np.asarray([counts[value] for value in group.recipient])[None]
        mass = weights.sum(axis=1)
        usable = mass > 0
        for metric in ("snr_energy_db", "snr_delta"):
            values = group[metric].to_numpy()
            if not np.isfinite(values).all():
                reports.append({"family": family, "metric": metric, "status": "undefined metric coverage"})
                continue
            boot = (weights[usable] @ values) / mass[usable]
            point = float(group.groupby("recipient")[metric].mean().mean())
            reports.append({"family": family, "metric": metric, "mean": point,
                "ci95_lower": float(np.quantile(boot, .025)), "ci95_upper": float(np.quantile(boot, .975)),
                "valid_draws": int(usable.sum()), "requested_draws": repetitions,
                "recipients": len(recipients), "donors": len(donors),
                "method": "recipient/donor Poisson multiplier percentile interval; project approximation",
                "scope": "development family-wise OOF; no final winner qualification"})
    atomic_json(output / "frontal_uncertainty.json", reports)


def validate_cached_reconstruction(modes, residual, reference):
    """Check archived float32 vectors using float64 accumulation.

    Summation rounding near a zero crossing must not masquerade as a damaged
    cache. The reference and tolerance are unchanged; vectors are never edited.
    """
    reconstructed = np.asarray(modes).sum(axis=0, dtype=np.float64)
    reconstructed = reconstructed + np.asarray(residual, dtype=np.float64)
    if not np.allclose(reconstructed, reference, rtol=1e-5, atol=1e-11):
        raise ValueError("Cached modes plus residual fail reconstruction")


class ArchivedModes:
    """Read only two checksum-verified settings from previously completed caches."""
    def __init__(self, input_root, corpus, config):
        self.corpus = corpus
        self.config = config
        self.index = {}
        for path in Path(input_root).rglob("mode_cache_index.jsonl"):
            verify_parent(path.parent, ("mode_cache_index.jsonl",))
            for item in read_jsonl(path):
                recipe = item["recipe"]
                if (recipe["K"], recipe["alpha"]) not in ((3, 2000), (4, 2000)):
                    continue
                expected = FixedRecipe(recipe["K"], recipe["alpha"], config["fs"])
                if recipe != asdict(expected) or item["recipe_hash"] != expected.identity() or item["scope"] != "frontal":
                    raise ValueError("Archived solver/scope identity differs from fixed policy")
                if item["corpus_manifest_sha256"] != sha256_file(corpus / "corpus_manifest.jsonl"):
                    raise ValueError("Cache corpus identity differs")
                key = (item["example_id"], expected.K)
                if key in self.index:
                    raise ValueError("Ambiguous archived fixed recipe cache")
                self.index[key] = (path.parent, item)

    def load(self, row, data, recipe):
        key = (row["example_id"], recipe.K)
        if key not in self.index:
            if self.config["require_archived_modes"]:
                raise ValueError("Required fixed-recipe cache entry missing: " + str(key))
            return decompose_frontal(data["eeg"], data["channel_names"], data["regions"], recipe, self.config, data["mask"])
        parent, item = self.index[key]
        if item["source_sha256"] != sha256_file(self.corpus / row["array_path"]):
            raise ValueError("Cached waveform identity differs")
        verify_parent(parent, (item["path"],))
        if sha256_file(parent / item["path"]) != item["sha256"]:
            raise ValueError("Cache index payload digest differs")
        ids = frontal_indices(data["eeg"], data["channel_names"], data["regions"], data["mask"])
        with np.load(parent / item["path"], allow_pickle=False) as saved:
            components, valid = saved["components"], saved["valid"]
            if components.shape[0] != len(data["eeg"]) or components.shape[-1] != self.config["window"]:
                raise ValueError("Cached mode channel/time axes differ")
            unsupported = np.ones(len(components), dtype=bool)
            unsupported[ids] = False
            if np.any(valid[unsupported]) or np.any(components[unsupported] != 0):
                raise ValueError("Cache contains VMD on nonfrontal channels")
            infos = {row["channel"]: row for row in item["diagnostics"]}
            result = {}
            for index in ids:
                info = infos.get(int(index))
                if info is None:
                    raise ValueError("Missing frontal convergence diagnostics")
                modes = components[index, :recipe.K].copy()
                residual = components[index, recipe.K].copy()
                if info.get("failure"):
                    modes[:] = 0
                    residual = data["eeg"][index].copy()
                validate_cached_reconstruction(modes, residual, data["eeg"][index])
                result[int(index)] = (modes, residual, info)
        return result


def candidates(config, profile):
    grid = config["classical_grid"]
    thresholds = [.4, .6] if profile == "pilot" else grid["thresholds"]
    components = [.2, .4] if profile == "pilot" else grid["thresholds"]
    strengths = [.5, 1.] if profile == "pilot" else grid["strengths"]
    lags = [[0]] if profile == "pilot" else grid["lag_banks"]
    penalties = [.01] if profile == "pilot" else grid["ridge_penalties"]
    result = [{"family": "identity", "kind": "identity", "K": None, "alpha": None,
               "trigger": 0., "component_threshold": 0., "strength": 0., "lags": [0], "penalty": .01}]
    for bank, penalty, trigger, strength in itertools.product(lags, penalties, thresholds, strengths):
        result.append({"family": "direct", "kind": "projected", "K": None, "alpha": None,
            "trigger": trigger, "component_threshold": 0., "strength": strength, "lags": bank, "penalty": penalty})
    # Each correction mechanism has the same bounded budget. Expensive entropy
    # selectors have one declared recipe; no invented author optimizer runs.
    for K in (3, 4):
        for family, kind in (("vmd_whole", "whole"), ("vmd_projected", "projected"),
                             ("vmd_interval", "interval_projected"), ("sobi_projected", "projected")):
            for trigger, component, strength in itertools.product(thresholds, components, strengths):
                result.append({"family": family, "kind": kind, "K": K, "alpha": 2000,
                    "trigger": trigger, "component_threshold": component, "strength": strength,
                    "lags": [0], "penalty": .01, "sobi_bank": 0})
        for family in ("svm_sobi_apen_fixed_adaptation", "author2022_selector_adaptation"):
            # SVM has one fully predeclared correction recipe. Its detector
            # OOF predictions cannot be reused for nested parameter tuning:
            # another fold's fitted detector may have seen the outer bucket.
            if family == "svm_sobi_apen_fixed_adaptation" and K != 3:
                continue
            result.append({"family": family, "kind": "apen_whole", "K": K, "alpha": 2000,
                "trigger": .4, "component_threshold": .4, "strength": 1.,
                "lags": [0], "penalty": .01, "sobi_bank": 0})
    if profile != "pilot":
        # Bounded lag/ridge ablation, not a Cartesian multiplication of every
        # threshold/strength setting. Lateral mixtures contain lagged fields;
        # leaving every mode arm instantaneous would handicap the comparison.
        for K, (family, kind), bank, penalty in itertools.product((3, 4),
                (("vmd_whole", "whole"), ("vmd_projected", "projected"),
                 ("vmd_interval", "interval_projected"), ("sobi_projected", "projected")),
                grid["lag_banks"], grid["ridge_penalties"]):
            if kind == "whole":
                continue
            result.append({"family": family, "kind": kind, "K": K, "alpha": 2000,
                "trigger": .4, "component_threshold": .2, "strength": 1.,
                "lags": bank, "penalty": penalty, "sobi_bank": 0})
    for spec in result:
        spec["comparison_arm"] = (spec["family"] if spec["K"] is None else
                                  f"{spec['family']}_K{spec['K']}")
    return {canonical_hash(spec): spec for spec in result}


def summary(scores, specs, limits):
    table = []
    for key, group in scores.groupby("candidate_hash"):
        dirty, clean = group[group.condition != "clean"], group[group.condition == "clean"]
        accuracy_columns = ["snr_energy_db", "rrmse_time", "pearson_cc", "mse", "snr_improvement_db"]
        preservation_columns = ["clean_change", "alpha_db", "beta_db", "covariance_error"]
        complete_accuracy = bool(len(dirty) and np.isfinite(dirty[accuracy_columns].to_numpy()).all())
        complete_preservation = bool(len(clean) and np.isfinite(clean[preservation_columns].to_numpy()).all())
        balanced = dirty.groupby(["recipient", "condition", "input_snr_db"])[
            ["snr_energy_db", "rrmse_time", "pearson_cc", "mse", "snr_improvement_db"]].mean().groupby("recipient").mean()
        clean_balanced = clean.groupby("recipient")[["clean_change", "alpha_db", "beta_db", "covariance_error"]].mean()
        gates = {"clean_change": bool(len(clean_balanced) and clean_balanced.clean_change.max() <= limits["clean_change"]),
                 "alpha": bool(len(clean_balanced) and clean_balanced.alpha_db.max() <= limits["alpha_db"]),
                 "beta": bool(len(clean_balanced) and clean_balanced.beta_db.max() <= limits["beta_db"]),
                 "covariance": bool(len(clean_balanced) and clean_balanced.covariance_error.max() <= limits["covariance"])}
        table.append({"candidate_hash": key, **specs[key], **balanced.mean().to_dict(),
            "worst_recipient_clean_change": float(clean_balanced.clean_change.max()),
            "partial_preservation_pass": complete_preservation and all(gates.values()),
            "selection_eligible": complete_accuracy and complete_preservation and all(gates.values()),
            "accuracy_coverage_complete": complete_accuracy,
            "preservation_coverage_complete": complete_preservation, "recipients": len(balanced),
            "failed_channel_examples": int(group.failure.notna().sum()),
            **{"gate_" + name: value for name, value in gates.items()}})
    return pd.DataFrame(table)


def grouped_selection(scores, specs, output, config):
    chosen, heldout_scores = [], []
    identity = next(key for key, spec in specs.items() if spec["family"] == "identity")
    for fold in sorted(scores.fold.unique()):
        train, heldout = scores[scores.fold != fold], scores[scores.fold == fold]
        for name in ("recipient", "donor"):
            if set(train[name]) & set(heldout[name]):
                raise ValueError("Frontal grouped selection leaks " + name)
        train_summary = summary(train, specs, config["preservation"])
        for family in sorted(set(train_summary.comparison_arm)):
            arm = train_summary[train_summary.comparison_arm == family]
            if family.startswith("svm_sobi_apen_fixed_adaptation"):
                # Other outer-fold detector predictions may come from fits
                # that saw this fold. Even preservation-triggered fallback
                # would be a leaked selection, despite having only one recipe.
                if len(arm) != 1:
                    raise ValueError("SVM must have one predeclared recipe")
                key = arm.iloc[0].candidate_hash
                policy = "predeclared fixed recipe; no selection or fallback using crossfit predictions"
            else:
                feasible = arm[arm.selection_eligible]
                key = identity if feasible.empty else feasible.sort_values(
                    ["snr_energy_db", "candidate_hash"], ascending=[False, True]).iloc[0].candidate_hash
                policy = "correction settings selected on other source buckets; K fixed within arm"
            evaluated = heldout[heldout.candidate_hash == key].copy()
            evaluated["evaluated_family"] = family
            evaluated["selection_fold"] = fold
            heldout_scores.append(evaluated)
            chosen.append({"fold": int(fold), "family": family, "selected": specs[key],
                "candidate_hash": key, "fallback_identity": key == identity and family != "identity",
                "selection_policy": policy,
                "scope": "family-wise source-excluded selection; final family winner not qualified"})
    write_jsonl(output / "frontal_fold_selections.jsonl", chosen)
    result = pd.concat(heldout_scores, ignore_index=True)
    result.to_csv(output / "frontal_oof_scores.csv", index=False)
    # Per-family OOF comparison includes deterioration versus its independently
    # training-selected direct comparator; do not select a winner on heldout.
    direct = result[result.evaluated_family == "direct"][["example_id", "channel", "pearson_cc"]]
    direct = direct.rename(columns={"pearson_cc": "direct_cc"})
    result = result.merge(direct, on=["example_id", "channel"], validate="many_to_one")
    result["cc_deterioration"] = result.direct_cc - result.pearson_cc
    result.to_csv(output / "frontal_oof_scores.csv", index=False)
    dirty = result[result.condition != "clean"]
    balanced = dirty.groupby(["evaluated_family", "recipient", "condition", "input_snr_db"])[
        ["snr_energy_db", "rrmse_time", "pearson_cc", "mse", "snr_improvement_db", "cc_deterioration"]].agg(strict_mean)
    recipients = balanced.groupby(["evaluated_family", "recipient"]).agg(strict_mean)
    recipients.reset_index().to_csv(output / "frontal_oof_recipients.csv", index=False)
    recipients.groupby("evaluated_family").agg(strict_mean).to_csv(output / "frontal_oof_means.csv")
    return result


def run_fixture(output, config):
    times = np.arange(config["window"]) / config["fs"]
    frontal = np.sin(2 * np.pi * 3 * times) + .4 * np.sin(2 * np.pi * 10 * times) + .2 * np.sin(2 * np.pi * 23 * times)
    values = np.stack([frontal, .5 * frontal, .3 * frontal])
    arrays = decompose_frontal(values, ["F3", "P3", "E1"], [0, 1, 3], FixedRecipe(3), config)
    if list(arrays) != [0]:
        raise RuntimeError("Frontal numerical fixture touched nonfrontal channels")
    modes, residual, info = arrays[0]
    if info.get("failure") or info["hit_iteration_limit"]:
        raise RuntimeError("Fixture decomposition did not converge")
    np.testing.assert_allclose(modes.sum(0) + residual, values[0], rtol=1e-5, atol=1e-6)
    atomic_json(output / "frontal_fixture_summary.json", {"passed": True,
        "configuration_hash": canonical_hash(config), "scope": "verified_frontal_only",
        "decomposed_channel_names": ["F3"], "autovmd_executed": False,
        "scientific_accuracy_qualified": False, "reserved_confirmation_opened": False})


def run_comparison(input_root, output, config, profile, experiment):
    if experiment.get("purpose", "fixed_frontal_replication") != "fixed_frontal_replication":
        raise ValueError("Undeclared frontal experiment purpose")
    parent, corpus_rows = corpus_parent(input_root)
    subset = experiment.get("selection_subset")
    if subset not in (None, "source-balanced pilot recipients"):
        raise ValueError("Undeclared correction-selection subset")
    rows = select_examples(corpus_rows, "pilot" if subset else profile)
    specs = candidates(config, profile)
    atomic_json(output / "frontal_candidates.json", specs)
    write_jsonl(output / "frontal_selection_manifest.jsonl", rows)
    save_protocol(output)
    cache = ArchivedModes(input_root, parent, config)
    detectors, detector_evidence = fit_detectors(parent, rows, config)
    write_jsonl(output / "frontal_detector_folds.jsonl", detector_evidence)
    import joblib
    detector_dir = output / "detectors"
    detector_dir.mkdir(exist_ok=True)
    for fold, detector in detectors.items():
        joblib.dump(detector, detector_dir / f"fold-{fold}.joblib")
    shards = output / "frontal_shards"
    shards.mkdir(exist_ok=True)
    all_scores, failures, diagnostics = [], [], []
    started = time.perf_counter()
    for number, row in enumerate(rows):
        print("FRONTAL_BEGIN", number + 1, len(rows), row["example_id"], flush=True)
        data = dict(np.load(parent / row["array_path"], allow_pickle=False))
        calibration = dict(np.load(parent / row["calibration"], allow_pickle=False))
        ids = frontal_indices(data["eeg"], data["channel_names"], data["regions"], data["mask"])
        if not len(ids):
            raise ValueError("Eligible controlled example has no verified frontal channels")
        references = data["references"]
        bundles = {K: cache.load(row, data, FixedRecipe(K)) for K in (3, 4)}
        original, target = data["eeg"][ids], data["paired_reference"][ids]
        association = np.max(np.abs(correlations(original, references, 20)), axis=1)
        channel_state = {}
        for local, index in enumerate(ids):
            predicted = bool(detectors[row["partition"]["fold"]].predict(
                detector_features(original[local], config["fs"], calibration["eeg"][index].std())[None])[0])
            channel_state[int(index)] = {"svm": predicted, "fits": {}, "projections": {}, "estimates": {}}
        example_scores, predictions = [], {}
        for key, spec in specs.items():
            artifacts = np.zeros_like(original, dtype=float)
            candidate_failures = [None] * len(ids)
            for local, index in enumerate(ids):
                state = channel_state[int(index)]
                family = spec["family"]
                trigger = state["svm"] if family == "svm_sobi_apen_fixed_adaptation" else association[local] >= spec["trigger"]
                if family == "identity" or not trigger:
                    continue
                if family == "direct":
                    projection_key = (tuple(spec["lags"]), spec["penalty"])
                    if projection_key not in state["projections"]:
                        state["projections"][projection_key] = project(original[local], references, spec["lags"], spec["penalty"])[0][0]
                    artifacts[local] = spec["strength"] * state["projections"][projection_key]
                    continue
                modes, residual, info = bundles[spec["K"]][int(index)]
                if info.get("failure") or info.get("hit_iteration_limit"):
                    candidate_failures[local] = info.get("failure") or "VMD iteration limit"
                    continue
                components, mixing, entropy = modes, None, None
                fit_key = spec["K"]
                try:
                    if "sobi" in family or family == "author2022_selector_adaptation":
                        if fit_key not in state["fits"]:
                            state["fits"][fit_key] = prepare_sobi(modes, 0)
                        expert = state["fits"][fit_key]
                        components, mixing = expert.sources, expert.mixing
                        if family == "svm_sobi_apen_fixed_adaptation":
                            entropy_key = (fit_key, "apen")
                            if entropy_key not in state["estimates"]:
                                state["estimates"][entropy_key] = np.asarray([approximate_entropy(source) for source in components])
                            entropy = state["estimates"][entropy_key]
                    if family == "author2022_selector_adaptation":
                        estimate_key = (fit_key, "author2022")
                        if estimate_key not in state["estimates"]:
                            selected_h, evidence_h = author_selector_adaptation(components, references[0])
                            selected_v, evidence_v = author_selector_adaptation(components, references[1])
                            selected = selected_h | selected_v
                            state["estimates"][estimate_key] = (mixing[:, selected] @ components[selected]).sum(0)
                            diagnostics.append({"example_id": row["example_id"], "channel": str(data["channel_names"][index]),
                                "K": fit_key, "family": family, "HEOG": evidence_h, "VEOG": evidence_v,
                                "adaptations": ["joint observed HEOG/VEOG", "outer EOG gate", "independent SOBI and entropy", "fixed project recipe"]})
                        artifacts[local] = state["estimates"][estimate_key]
                    else:
                        estimate_key = (spec["K"], family, spec["component_threshold"],
                                        tuple(spec["lags"]), spec["penalty"])
                        if estimate_key not in state["estimates"]:
                            unit_strength = {**spec, "strength": 1.}
                            estimate, detail = estimate_components(components, references, unit_strength,
                                calibration["references"], mixing=mixing, entropy=entropy, fs=config["fs"])
                            state["estimates"][estimate_key] = estimate
                            diagnostics.append({"example_id": row["example_id"], "channel": str(data["channel_names"][index]),
                                "K": spec["K"], "family": family, "component_threshold": spec["component_threshold"], **detail})
                        artifacts[local] = spec["strength"] * state["estimates"][estimate_key]
                except (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError) as error:
                    candidate_failures[local] = str(error)
            cleaned = original - artifacts
            paper = paired_channels(cleaned, target, original)
            preservation = paired(cleaned, target, config["fs"])
            for local, index in enumerate(ids):
                failed = candidate_failures[local]
                score = {"example_id": row["example_id"], "recipient": row["recipient"], "donor": row["donor"],
                    "fold": row["partition"]["fold"], "condition": row["condition"], "input_snr_db": row["input_snr_db"],
                    "channel": str(data["channel_names"][index]), "candidate_hash": key, "family": spec["family"],
                    "failure": failed, "clean_change": float(paper["rrmse_time"][local]),
                    "alpha_db": preservation["alpha_db"], "beta_db": preservation["beta_db"],
                    "covariance_error": preservation["covariance"],
                    **{name: finite_value(values[local]) for name, values in paper.items()}}
                example_scores.append(score)
                if failed:
                    failures.append({"example_id": row["example_id"], "channel": score["channel"], "candidate": key,
                                     "error": failed, "status": "scored_passthrough"})
            # Save every pilot prediction; full jobs retain all tabulated scores
            # and selected predictions in a follow-up evaluation, not huge grids.
            if profile == "pilot":
                predictions[key] = cleaned.astype(np.float32)
        frame = pd.DataFrame(example_scores)
        frame.to_csv(shards / (row["example_id"] + ".csv"), index=False)
        if profile == "pilot":
            np.savez_compressed(shards / (row["example_id"] + ".npz"), original=original, target=target,
                references=references, channel_names=data["channel_names"][ids], **predictions)
        all_scores.extend(example_scores)
        write_jsonl(output / "frontal_failures.jsonl", failures)
        atomic_json(output / "frontal_progress.json", {"completed": number + 1, "total": len(rows),
            "last_example": row["example_id"], "runtime_s": time.perf_counter() - started})
        print("FRONTAL_SAVED", number + 1, len(rows), flush=True)
    scores = pd.DataFrame(all_scores)
    scores.to_csv(output / "frontal_channels.csv", index=False)
    search = summary(scores, specs, config["preservation"])
    search.to_csv(output / "frontal_search.csv", index=False)
    oof = grouped_selection(scores, specs, output, config)
    uncertainty_report(oof, output, config)
    write_jsonl(output / "frontal_component_diagnostics.jsonl", diagnostics)
    mode_records = []
    for row in rows:
        for K in (3, 4):
            _, item = cache.index[(row["example_id"], K)]
            mode_records.extend({"example_id": row["example_id"], "K": K, "alpha": 2000, **info} for info in item["diagnostics"])
    write_jsonl(output / "frontal_mode_diagnostics.jsonl", mode_records)
    plot_results(oof, output)
    atomic_json(output / "frontal_summary.json", {"examples": len(rows), "candidates": len(specs),
        "recipients": int(scores.recipient.nunique()), "frontal_channels": sorted(scores.channel.unique()),
        "failure_rows": len(failures), "profile": profile, "runtime_s": time.perf_counter() - started,
        "selection_subset": subset, "evaluation_complete": subset is None and profile == "full",
        "autovmd_executed": False, "fixed_recipes": config["fixed_recipes"], "scope": "verified_frontal_only",
        "teacher_qualified": False, "regional_integration_authorized": False, "model_training_authorized": False,
        "native_validation_complete": False, "reserved_confirmation_opened": False,
        "paper_replication": False, "status": "development comparison; scientific qualification remains pending",
        "adaptations": "fixed project VMD; independent SOBI/entropy; observed joint EOG; 5.12-second project windows; subset SVM features without mRMR"})


def plot_results(scores, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    dirty = scores[scores.condition != "clean"]
    table = dirty.groupby(["evaluated_family", "recipient", "condition", "input_snr_db"]).snr_energy_db.mean()
    table = table.groupby(["evaluated_family", "condition"]).mean().unstack()
    figure, axes = plt.subplots(1, 2, figsize=(16, 7))
    table.plot.barh(ax=axes[0])
    axes[0].set_xlabel("Controlled output SNR (dB)")
    axes[0].set_title("Source-excluded frontal development comparison")
    axes[0].axvline(15, color="black", linestyle="--", label="15 dB target")
    clean = scores[scores.condition == "clean"].groupby(["evaluated_family", "recipient"]).clean_change.mean().groupby("evaluated_family").max()
    clean.plot.barh(ax=axes[1], color="steelblue")
    axes[1].axvline(.01, color="black", linestyle="--")
    axes[1].set_xlabel("Worst-recipient mean clean relative modification")
    axes[1].set_title("Engineering preservation; native validation pending")
    figure.tight_layout()
    figure.savefig(output / "frontal_comparison.png", dpi=150)
    plt.close(figure)
