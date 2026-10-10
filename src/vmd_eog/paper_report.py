"""Paper metric aggregation and predeclared paired comparison families."""
import numpy as np
import pandas as pd
from .io import atomic_json
from .paper_metrics import paired_permutation


PAIRED_METRICS = ("rrmse_time", "mse", "rmse", "pearson_cc", "snr_energy_db")
NATIVE_METRICS = ("rest_rmse", "eeg_eog_abs_r_after")


def example_means(frame, identities, metrics):
    """Channel-first means; an undefined channel invalidates that metric unit.

    This deliberately avoids pandas' default silent missing-value exclusion.
    Infinite SNR remains infinite; it is never capped or converted into a
    finite invented score. Undefined counts remain in the eligibility table.
    """
    grouped = frame.groupby(identities, dropna=False, sort=True)
    means = grouped[list(metrics)].agg(lambda values: np.asarray(values, float).mean())
    counts = grouped[list(metrics)].agg(lambda values: int((~np.isfinite(np.asarray(values, float))).sum()))
    means = means.reset_index()
    counts = counts.add_suffix("_nonfinite_channels").reset_index()
    return means.merge(counts, on=identities)


def comparison_family(frame, source, strata, metrics):
    """Whole-montage participant summaries; identity/direct are fixed comparators."""
    comparisons = []
    for keys, group in frame.groupby(strata, dropna=False, sort=True):
        if not isinstance(keys, tuple): keys = (keys,)
        identity = dict(zip(strata, keys))
        methods = sorted(group.method.unique())
        for metric in metrics:
            pivot = group.pivot(index=source, columns="method", values=metric)
            for comparator in ("identity", "direct"):
                if comparator not in methods: continue
                for method in methods:
                    if method == comparator: continue
                    comparisons.append((identity, metric, method, comparator, pivot))
    records = []
    family_size = len(comparisons)
    for index, (identity, metric, method, comparator, pivot) in enumerate(comparisons):
        first, second = pivot[method].to_numpy(), pivot[comparator].to_numpy()
        eligible = np.isfinite(first) & np.isfinite(second)
        record = {**identity, "metric": metric, "method": method, "comparator": comparator,
                  "participant_count": len(first), "finite_matched_participants": int(eligible.sum()),
                  "comparison_family_size": family_size}
        if not eligible.all() or len(first) < 2:
            record["status"] = "unavailable: incomplete or nonfinite participant pairs"
        else:
            record.update(paired_permutation(first, second, comparisons=family_size,
                                             seed=42+index, repetitions=10000))
            record["status"] = "computed"
        records.append(record)
    return pd.DataFrame(records)


def write_reports(output):
    """Summaries are adaptations; keep the original per-channel paper tables."""
    summaries = {}
    paired_path = output/"paper_paired_channels.csv"
    if paired_path.exists():
        frame = pd.read_csv(paired_path)
        keys = ["dataset", "recipient", "condition", "input_snr_db", "method", "example_id"]
        examples = example_means(frame, keys, PAIRED_METRICS)
        examples.to_csv(output/"paper_paired_example_means.csv", index=False)
        source_keys = keys[:-1]
        sources = examples.groupby(source_keys, dropna=False)[list(PAIRED_METRICS)].agg(
            lambda values: np.asarray(values, float).mean()).reset_index()
        sources.to_csv(output/"paper_paired_source_means.csv", index=False)
        # Klados has unknown participant identities: no invented participant test.
        controlled = sources[sources.dataset.isin(["controlled_LEMON_OSF", "synthetic_fixture"])]
        dirty = controlled[controlled.condition != "clean"]
        tests = comparison_family(dirty, "recipient", ["dataset", "condition", "input_snr_db"], PAIRED_METRICS)
        tests.to_csv(output/"paper_paired_permutation_tests.csv", index=False)
        summaries["paired"] = {"channel_rows": len(frame), "example_rows": len(examples),
            "source_rows": len(sources), "planned_comparisons": len(tests),
            "aggregation": "channel arithmetic mean -> example mean -> recipient-condition-input-level mean",
            "legacy_klados_participant_tests": "not performed: participant mapping unknown"}
    native_path = output/"paper_native_channels.csv"
    if native_path.exists():
        frame = pd.read_csv(native_path)
        keys = ["dataset", "source", "condition", "method", "example_id"]
        examples = example_means(frame, keys, NATIVE_METRICS)
        examples.to_csv(output/"paper_native_example_means.csv", index=False)
        sources = examples.groupby(keys[:-1], dropna=False)[list(NATIVE_METRICS)].agg(
            lambda values: np.asarray(values, float).mean()).reset_index()
        sources.to_csv(output/"paper_native_source_means.csv", index=False)
        planned = []
        for condition, group in sources.groupby("condition"):
            metric = "rest_rmse" if condition == "rest" else "eeg_eog_abs_r_after"
            planned.append(comparison_family(group, "source", ["dataset", "condition"], [metric]))
        # Bonferroni must cover all native conditions together, not separate
        # favorable families per condition. Recompute adjusted probabilities
        # from the saved unadjusted p-values using the combined family size.
        tests = pd.concat(planned, ignore_index=True) if planned else pd.DataFrame()
        if len(tests):
            tests["comparison_family_size"] = len(tests)
            if "p_two_sided" in tests:
                tests["p_bonferroni"] = (tests.p_two_sided*len(tests)).clip(upper=1.)
        tests.to_csv(output/"paper_native_permutation_tests.csv", index=False)
        summaries["native"] = {"channel_rows": len(frame), "source_rows": len(sources),
            "planned_comparisons": len(tests), "chance_level_established": False,
            "aggregation": "channel mean -> cached-window mean -> participant-condition mean",
            "whole_condition_concatenation_replication": False}
    atomic_json(output/"paper_metric_report_summary.json", {
        "results": summaries, "scope": "source-mapped metrics with declared adapted aggregation",
        "paper_region_assignment_verified": False,
        "reason": "EEGOAR supplementary section S.II channel assignments unavailable; retain per-channel values",
        "native_chance_level_established": False, "model_authorization": False,
    })
