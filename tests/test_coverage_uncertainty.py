import numpy as np
import pandas as pd
import pytest

from vmd_eog.coverage_uncertainty import uncertainty_report


SCOPES = ("whole", "frontal", "posterior", "central_shared")


def score_fixture():
    rows = []
    for recipient_index in range(2):
        recipient = f"r{recipient_index}"
        for condition_index, condition in enumerate(("blink", "saccade")):
            for input_index, input_snr in enumerate((-5.0, 0.0)):
                donor = f"d{(recipient_index + condition_index + input_index) % 2}"
                example_id = f"{recipient}-{donor}-{condition}-{input_snr}"
                for scope in SCOPES:
                    for arm in ("frozen_regional", "central_frontal_vmd_context"):
                        is_candidate = arm == "central_frontal_vmd_context"
                        rows.append({
                                "example_id": example_id,
                                "arm": arm,
                                "scope": scope,
                                "channel": "C1",
                                "recipient": recipient,
                                "donor": donor,
                                "condition": condition,
                                "input_snr_db": input_snr,
                                "failure": None,
                                "mse": 0.2 - (0.02 if is_candidate else 0.0),
                                "rrmse_time": 0.4 - (0.02 if is_candidate else 0.0),
                                "pearson_cc": 0.8 + (0.01 if is_candidate else 0.0),
                                "snr_energy_db": 10.0 + (2.0 if is_candidate else 0.0),
                                "snr_improvement_db": 1.0 + (2.0 if is_candidate else 0.0),
                            })
        clean_example = f"{recipient}-clean"
        for scope in SCOPES:
            for arm in ("frozen_regional", "central_frontal_vmd_context"):
                rows.append({
                    "example_id": clean_example,
                    "arm": arm,
                    "scope": scope,
                    "channel": "C1",
                    "recipient": recipient,
                    "donor": f"d{recipient_index}",
                    "condition": "clean",
                    "input_snr_db": np.nan,
                    "failure": None,
                    "mse": np.nan,
                    "rrmse_time": np.nan,
                    "pearson_cc": np.nan,
                    "snr_energy_db": np.nan,
                    "snr_improvement_db": np.nan,
                })
    return pd.DataFrame(rows)


def test_explicit_frontal_scope_retains_paired_analysis_without_fabricated_regions(tmp_path):
    scores = score_fixture()
    frontal = scores[scores.scope == "frontal"].copy()
    report = uncertainty_report(frontal, tmp_path, scopes=("frontal",), repetitions=20)
    assert set(report.scope) == {"frontal"}
    assert report[report.metric == "snr_energy_db"].iloc[0].mean_delta == pytest.approx(2.)
    with pytest.raises(ValueError, match="scopes"):
        uncertainty_report(frontal, tmp_path / "bad", scopes=("frontal", "frontal"), repetitions=20)


def test_paired_source_bootstrap_has_expected_direction_and_is_reproducible(tmp_path):
    scores = score_fixture()
    first = uncertainty_report(scores, tmp_path / "first", repetitions=80)
    second = uncertainty_report(scores, tmp_path / "second", repetitions=80)
    snr = first[(first.scope == "whole") & (first.metric == "snr_energy_db")].iloc[0]
    assert snr.status == "available"
    assert snr.mean_delta == pytest.approx(2.0)
    assert snr.favors_candidate
    pd.testing.assert_frame_equal(first, second)
    assert (tmp_path / "first" / "coverage_paired_differences.csv").is_file()
    assert (tmp_path / "first" / "coverage_uncertainty_channel_strata.csv").is_file()
    assert (tmp_path / "first" / "coverage_uncertainty_method.json").is_file()


def test_unique_arm_scope_channel_example_key_is_required(tmp_path):
    scores = score_fixture()
    scores = pd.concat([scores, scores.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="unique"):
        uncertainty_report(scores, tmp_path, repetitions=10)


def test_candidate_and_comparator_must_have_identical_pair_rows(tmp_path):
    scores = score_fixture()
    drop = ((scores.arm == "central_frontal_vmd_context") &
            (scores.scope == "central_shared")).idxmax()
    scores = scores.drop(index=drop)
    with pytest.raises(ValueError, match="rows differ"):
        uncertainty_report(scores, tmp_path, repetitions=10)


def test_failure_rows_are_retained_counted_and_finite_values_stay_in_interval(tmp_path):
    scores = score_fixture()
    failure_row = ((scores.arm == "central_frontal_vmd_context") &
                   (scores.scope == "whole")).idxmax()
    scores.loc[failure_row, "failure"] = "fit failed"
    intervals = uncertainty_report(scores, tmp_path, repetitions=20)
    row = intervals[(intervals.scope == "whole") &
                    (intervals.metric == "snr_energy_db")].iloc[0]
    assert row.status == "available"
    assert row.candidate_failure_rows == 1
    paired = pd.read_csv(tmp_path / "coverage_paired_differences.csv")
    assert paired.candidate_failure.notna().sum() == 1


def test_nonfinite_identity_or_arm_metric_is_explicitly_unavailable(tmp_path):
    scores = score_fixture()
    row_index = ((scores.arm == "central_frontal_vmd_context") &
                 (scores.scope == "whole")).idxmax()
    scores.loc[row_index, "snr_energy_db"] = np.inf
    intervals = uncertainty_report(scores, tmp_path, repetitions=20)
    row = intervals[(intervals.scope == "whole") &
                    (intervals.metric == "snr_energy_db")].iloc[0]
    assert row.status == "unavailable_nonfinite_metric"
    assert pd.isna(row.ci95_lower)


def test_two_distinct_recipients_and_donors_are_required(tmp_path):
    scores = score_fixture()
    scores = scores[scores.recipient == "r0"]
    intervals = uncertainty_report(scores, tmp_path, repetitions=20)
    row = intervals[(intervals.scope == "whole") &
                    (intervals.metric == "snr_energy_db")].iloc[0]
    assert row.status == "unavailable_insufficient_sources"


def test_clean_control_nan_snr_is_excluded_before_pairing(tmp_path):
    intervals = uncertainty_report(score_fixture(), tmp_path, repetitions=20)
    whole = intervals[intervals.scope == "whole"]
    assert set(whole.status) == {"available"}
    paired = pd.read_csv(tmp_path / "coverage_paired_differences.csv")
    assert not (paired.condition == "clean").any()


def test_donor_source_metadata_must_match_for_each_pair(tmp_path):
    scores = score_fixture()
    row = ((scores.arm == "central_frontal_vmd_context") &
           (scores.scope == "whole") &
           (scores.condition != "clean")).idxmax()
    scores.loc[row, "donor"] = "different-donor"
    with pytest.raises(ValueError, match="metadata differs"):
        uncertainty_report(scores, tmp_path, repetitions=10)


def test_unequal_channel_counts_are_averaged_within_example_first(tmp_path):
    scores = score_fixture()
    example = scores.loc[(scores.scope == "whole") &
                         (scores.condition != "clean"), "example_id"].iloc[0]
    extra = scores[(scores.scope == "whole") & (scores.example_id == example)].copy()
    extra["channel"] = "C2"
    extra.loc[extra.arm == "central_frontal_vmd_context", "snr_energy_db"] = 14.0
    extra.loc[extra.arm == "frozen_regional", "snr_energy_db"] = 10.0
    scores = pd.concat([scores, extra], ignore_index=True)
    intervals = uncertainty_report(scores, tmp_path, repetitions=20)
    row = intervals[(intervals.scope == "whole") &
                    (intervals.metric == "snr_energy_db")].iloc[0]
    assert row.mean_delta == pytest.approx(2.125)
