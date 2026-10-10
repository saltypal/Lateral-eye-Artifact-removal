"""Figure inputs must preserve participant weights and unavailable values."""
import json
import numpy as np
import pandas as pd
from vmd_eog.paper_figures import controlled_participants, participant_interval, plot_paper_review


def test_primary_figure_balances_levels_and_excludes_legacy():
    rows=[]
    for dataset,values in (("controlled_LEMON_OSF",[0.,20.]),("legacy_klados",[1000.])):
        for value in values:
            rows.append({"dataset":dataset,"condition":"blink","recipient":"p1","method":"direct",
                         "rrmse_time":.1,"mse":2.,"pearson_cc":.9,"snr_energy_db":value})
    summary=controlled_participants(pd.DataFrame(rows))
    assert len(summary)==1
    assert summary.iloc[0].snr_energy_db==10.
    assert participant_interval([10.,10.])==(10.,10.,10.)


def test_nonfinite_plot_cohort_is_unavailable_instead_of_dropping_participants():
    assert np.isnan(participant_interval([10.,np.nan])).all()
    assert np.isnan(participant_interval([10.,np.inf])).all()


def test_primary_and_native_figures_render_with_adapted_chance_schema(tmp_path):
    regional,native,output=(tmp_path/name for name in ("regional","native","output"))
    for directory in (regional,native,output):
        directory.mkdir()
    paired=[]
    real=[]
    for method in ("identity","direct","regional_mwf","regional_ica"):
        for condition in ("blink","lateral","mixed"):
            paired.append({"dataset":"controlled_LEMON_OSF","condition":condition,"recipient":"fixture",
                "method":method,"rrmse_time":.1,"mse":1e-12,"pearson_cc":.9,"snr_energy_db":10.})
        for condition in ("rest","blink","lateral","vertical"):
            real.append({"source":"fixture","condition":condition,"method":method,
                "rest_rmse":0. if condition=="rest" else np.nan,
                "eeg_eog_abs_r_after":np.nan if condition=="rest" else .1})
    pd.DataFrame(paired).to_csv(regional/"paper_paired_source_means.csv",index=False)
    pd.DataFrame(real).to_csv(native/"native_condition_participants.csv",index=False)
    (native/"native_chance_summary.json").write_text(json.dumps({"conditions":[
        {"condition":"blink","status":"computed adapted null","p95":.05}]}))
    plot_paper_review(regional,native,output,42)
    for name in ("paper_primary_metrics.png","paper_condition_SNR.png","paper_native_conditions.png"):
        assert (output/name).stat().st_size>1000
