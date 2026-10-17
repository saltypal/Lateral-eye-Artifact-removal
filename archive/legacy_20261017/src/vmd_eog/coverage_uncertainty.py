"""Paired source-aware uncertainty summaries for regional coverage metrics.

This is a project-level multiplier bootstrap, not a paper-native test. It first
averages channels within each example, balances condition/input strata within
recipient, then gives recipients equal weight. Recipient and donor Poisson
multipliers are applied to the same paired candidate-minus-comparator rows.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .posterior_context import PAPER_KEYS


DEFAULT_COMPARATOR = "frozen_regional"
DEFAULT_CANDIDATES = ("central_frontal_vmd_context",)
DEFAULT_SCOPES = ("whole", "frontal", "posterior", "central_shared")
METRIC_DIRECTIONS = {
    "mse": "lower_is_better",
    "rrmse_time": "lower_is_better",
    "pearson_cc": "higher_is_better",
    "snr_energy_db": "higher_is_better",
    "snr_improvement_db": "higher_is_better",
}
KEY_COLUMNS = ("example_id", "arm", "scope", "channel")
PAIR_COLUMNS = ("example_id", "scope", "channel")
META_COLUMNS = ("recipient", "donor", "condition", "input_snr_db")


def _json_value(value):
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _validate_scores(scores):
    missing = set(KEY_COLUMNS + META_COLUMNS + ("failure",)) - set(scores.columns)
    missing |= set(PAPER_KEYS) - set(scores.columns)
    if missing:
        raise ValueError("Coverage score table is missing columns: " + ", ".join(sorted(missing)))
    if scores.empty:
        raise ValueError("Coverage score table is empty")
    failures = scores.loc[scores.failure.notna(), "failure"]
    if not failures.map(lambda value: isinstance(value, str) and bool(value.strip())).all():
        raise ValueError("Failure contract requires null for success or a nonempty failure reason")
    if scores[list(KEY_COLUMNS + META_COLUMNS[:-1])].isna().any().any():
        raise ValueError("Coverage keys and source metadata must be defined")
    if scores.loc[scores.condition != "clean", "input_snr_db"].isna().any():
        raise ValueError("Dirty-condition input SNR strata must be defined")
    if scores.duplicated(list(KEY_COLUMNS)).any():
        raise ValueError("Coverage example/arm/scope/channel keys must be unique")
    unexpected = set(scores.scope.unique()) - set(DEFAULT_SCOPES)
    if unexpected:
        raise ValueError("Unexpected coverage scopes: " + ", ".join(sorted(map(str, unexpected))))


def _strict_pair(scores, comparator, candidate, scope, metrics):
    selected = scores[(scores.scope == scope) & scores.arm.isin([comparator, candidate]) &
                      (scores.condition != "clean")].copy()
    left = selected[selected.arm == comparator].copy()
    right = selected[selected.arm == candidate].copy()
    if left.empty or right.empty:
        raise ValueError(f"Missing rows for {candidate} vs {comparator} in scope {scope}")
    if left.duplicated(list(PAIR_COLUMNS)).any() or right.duplicated(list(PAIR_COLUMNS)).any():
        raise ValueError("Paired comparison keys are not unique")
    left_index = pd.MultiIndex.from_frame(left[list(PAIR_COLUMNS)]).sort_values()
    right_index = pd.MultiIndex.from_frame(right[list(PAIR_COLUMNS)]).sort_values()
    if not left_index.equals(right_index):
        raise ValueError(f"Candidate/comparator rows differ for {candidate} vs {comparator}, {scope}")
    left = left.set_index(list(PAIR_COLUMNS)).sort_index()
    right = right.set_index(list(PAIR_COLUMNS)).sort_index()
    left = left.loc[right.index]
    for column in META_COLUMNS:
        if not left[column].equals(right[column]):
            raise ValueError(f"Paired source metadata differs for {column}")
    rows = right[list(META_COLUMNS) + ["failure"] + list(metrics)].copy()
    rows = rows.rename(columns={"failure": "candidate_failure"})
    rows["comparator_failure"] = left["failure"]
    rows["candidate"] = candidate
    rows["comparator"] = comparator
    for metric in metrics:
        rows[f"candidate_{metric}"] = right[metric]
        rows[f"comparator_{metric}"] = left[metric]
        rows[f"delta_{metric}"] = right[metric].to_numpy(dtype=float) - left[metric].to_numpy(dtype=float)
    rows = rows.reset_index()
    return rows


def _status(rows, metric, recipients, donors, expected_strata):
    if len(recipients) < 2 or len(donors) < 2:
        return "unavailable_insufficient_sources", "At least two recipients and two donors are required"
    values = rows[f"delta_{metric}"].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        return "unavailable_nonfinite_metric", "Nonfinite metrics, including perfect-reconstruction infinity, are retained and suppress the interval"
    seen = set(zip(rows.recipient, rows.condition, rows.input_snr_db))
    required = {(recipient, condition, input_snr) for recipient in recipients
                for condition, input_snr in expected_strata}
    if seen != required:
        return "unavailable_incomplete_strata", "Each recipient must have every condition/input stratum"
    return "available", ""


def _units(rows, metric):
    # Channel means are computed first inside each example. Each source unit is
    # one donor/example within its recipient/condition/input cell.
    example_keys = ["example_id", "recipient", "donor", "condition", "input_snr_db"]
    examples = rows.groupby(example_keys, dropna=False, sort=True)[f"delta_{metric}"].agg("mean").reset_index()
    return examples.rename(columns={f"delta_{metric}": "value"})


def _base_weights(units, recipients, expected_strata):
    """Equalize recipients and condition/input cells, then donor units per cell."""
    counts = units.groupby(["recipient", "condition", "input_snr_db"], dropna=False).size()
    return np.asarray([
        1.0 / (len(recipients) * len(expected_strata) *
               int(counts.loc[(row.recipient, row.condition, row.input_snr_db)]))
        for row in units.itertuples(index=False)
    ], dtype=float)


def _bootstrap(units, recipients, donors, base_weights, repetitions, seed):
    """Vectorized crossed source multipliers on prebalanced paired source units."""
    generator = np.random.default_rng(seed)
    recipient_weights = generator.poisson(1.0, size=(repetitions, len(recipients)))
    donor_weights = generator.poisson(1.0, size=(repetitions, len(donors)))
    recipient_index = {value: index for index, value in enumerate(recipients)}
    donor_index = {value: index for index, value in enumerate(donors)}
    values = units.value.to_numpy(dtype=float)
    rindex = np.asarray([recipient_index[value] for value in units.recipient])
    dindex = np.asarray([donor_index[value] for value in units.donor])
    weights = (recipient_weights[:, rindex] * donor_weights[:, dindex] *
               base_weights[None, :])
    mass = weights.sum(axis=1)
    usable = mass > 0
    boot = np.full(repetitions, np.nan, dtype=float)
    boot[usable] = (weights[usable] @ values) / mass[usable]
    return boot[np.isfinite(boot)]


def uncertainty_report(scores, output, config=None, *, comparators=(DEFAULT_COMPARATOR,),
                       candidates=DEFAULT_CANDIDATES, repetitions=2000, seed=42, scopes=DEFAULT_SCOPES):
    """Write paired source-aware intervals and row-level paired differences.

    ``scores`` is the coverage channel score DataFrame. The default comparison
    is ``central_frontal_vmd_context`` against ``frozen_regional``. Outputs are
    ``coverage_paired_differences.csv``, ``coverage_uncertainty_intervals.csv``,
    ``coverage_uncertainty_channel_strata.csv`` and
    ``coverage_uncertainty_method.json`` under ``output``.
    """
    del config  # Kept as an integration-friendly call signature; all policy is explicit here.
    _validate_scores(scores)
    scopes = tuple(scopes)
    if not scopes or len(set(scopes)) != len(scopes) or not set(scopes).issubset(DEFAULT_SCOPES):
        raise ValueError("Uncertainty scopes must be a nonempty unique subset of declared scopes")
    if repetitions < 1 or seed != 42:
        raise ValueError("Coverage uncertainty uses deterministic seed 42 and positive repetitions")
    metrics = tuple(metric for metric in PAPER_KEYS if metric in scores.columns)
    if not metrics:
        raise ValueError("No supported paper metrics are present")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    all_pairs = []
    interval_rows = []
    for comparator in comparators:
        for candidate in candidates:
            if candidate == comparator:
                raise ValueError("Candidate and comparator must differ")
            for scope in scopes:
                pair = _strict_pair(scores, comparator, candidate, scope, metrics)
                all_pairs.append(pair)
                recipients = sorted(pair.recipient.unique(), key=str)
                donors = sorted(pair.donor.unique(), key=str)
                expected_strata = sorted(set(zip(pair.condition, pair.input_snr_db)),
                                          key=lambda item: (str(item[0]), str(item[1])))
                for metric in metrics:
                    status, reason = _status(pair, metric, recipients, donors, expected_strata)
                    row = {"scope": scope, "candidate": candidate, "comparator": comparator,
                           "metric": metric, "direction": METRIC_DIRECTIONS[metric],
                           "status": status, "reason": reason,
                           "recipients": len(recipients), "donors": len(donors),
                           "candidate_failure_rows": int(pair.candidate_failure.notna().sum()),
                           "comparator_failure_rows": int(pair.comparator_failure.notna().sum()),
                           "requested_draws": repetitions, "seed": seed,
                           "method": "paired recipient/donor Poisson multiplier percentile interval; project approximation"}
                    if status == "available":
                        units = _units(pair, metric)
                        base_weights = _base_weights(units, recipients, expected_strata)
                        point = float(base_weights @ units.value.to_numpy(dtype=float))
                        boot = _bootstrap(units, recipients, donors, base_weights,
                                          repetitions, seed)
                        if len(boot):
                            row.update(mean_delta=point,
                                       ci95_lower=float(np.quantile(boot, .025)),
                                       ci95_upper=float(np.quantile(boot, .975)),
                                       valid_draws=int(len(boot)),
                                       valid_mass_draws=int(len(boot)),
                                       favors_candidate=bool(point > 0 if METRIC_DIRECTIONS[metric] == "higher_is_better" else point < 0))
                        else:
                            row.update(status="unavailable_no_valid_draws", reason="Every multiplier draw had zero source mass or incomplete strata",
                                       mean_delta=point, ci95_lower=None, ci95_upper=None,
                                       valid_draws=0, valid_mass_draws=0, favors_candidate=None)
                    else:
                        row.update(mean_delta=None, ci95_lower=None, ci95_upper=None,
                                   valid_draws=0, valid_mass_draws=0, favors_candidate=None)
                    interval_rows.append(row)

    paired = pd.concat(all_pairs, ignore_index=True)
    paired.to_csv(output / "coverage_paired_differences.csv", index=False)
    intervals = pd.DataFrame(interval_rows)
    intervals.to_csv(output / "coverage_uncertainty_intervals.csv", index=False)
    channel_keys = ["scope", "candidate", "comparator", "recipient", "donor",
                    "condition", "input_snr_db", "channel"]
    diagnostic = paired.groupby(channel_keys, dropna=False, sort=True)[
        [f"delta_{metric}" for metric in metrics]].agg("mean").reset_index()
    diagnostic.to_csv(output / "coverage_uncertainty_channel_strata.csv", index=False)
    method = {
        "method": "paired recipient/donor Poisson multiplier percentile intervals; project approximation",
        "seed": seed,
        "requested_repetitions": repetitions,
        "comparators": list(comparators),
        "candidates": list(candidates),
        "scopes": list(scopes),
        "metrics": list(metrics),
        "metric_directions": METRIC_DIRECTIONS,
        "unavailable_requested_metrics": ["rrmse_spectrum"] if "rrmse_spectrum" not in scores.columns else [],
        "channel_aggregation": "mean paired channel differences within example first",
        "condition_input_aggregation": "equal condition/input strata within recipient",
        "recipient_aggregation": "equal recipient weight",
        "source_weights": "independent Poisson(1) recipient and donor multipliers on paired source units; base weights equalize recipients, condition/input cells within recipient, and donor/example units within each cell",
        "failure_semantics": "candidate and comparator failure flags are counted and retained; finite scored values remain in the all-cohort interval, while nonfinite metric values suppress the interval",
        "strict_rules": ["unique example/arm/scope/channel keys", "identical candidate/comparator key sets",
                         "source metadata must match", "failure rows retained and counted; finite values remain in the all-cohort interval",
                         "nonfinite metrics retained and suppress intervals", "at least two recipients and donors",
                         "all recipients require identical condition/input strata"],
        "population_scope": "as recorded by corpus metadata; no participant-heldout claim added",
    }
    (output / "coverage_uncertainty_method.json").write_text(
        json.dumps(_json_value(method), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return intervals
