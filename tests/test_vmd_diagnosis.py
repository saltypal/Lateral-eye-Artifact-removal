"""Controls distinguish a solver failure policy from a mode-selection failure."""
import numpy as np
import pytest
from vmd_eog.vmd_diagnosis import correction_controls, dual_gate_correction


def test_mode_residual_controls_use_actual_components():
    modes=np.array([[1.,2.,3.],[4.,5.,6.]])
    residual=np.array([.1,.2,.3])
    results=correction_controls(modes,residual,[True,False],True,.5,False)
    np.testing.assert_allclose(results["vmd_all_modes_channel_gate"],.5*modes.sum(0))
    np.testing.assert_allclose(results["vmd_selected_plus_residual"],.5*(modes[0]+residual))
    np.testing.assert_allclose(results["vmd_all_components_closure"],.5*(modes.sum(0)+residual))


def test_capped_modes_pass_through_but_algebraic_control_remains_inspectable():
    results=correction_controls([[1.,2.],[3.,4.]],[.5,.5],[False,False],True,1.,True)
    for method in ("vmd_all_modes_channel_gate","vmd_selected_plus_residual","vmd_all_components_converged"):
        np.testing.assert_array_equal(results[method],[0.,0.])
    np.testing.assert_allclose(results["vmd_all_components_closure"],[4.5,6.5])
    with pytest.raises(ValueError,match="sample lengths"):
        correction_controls([[1.,2.]],[1.],[True],True,1.,False)


def test_outer_gate_cannot_be_bypassed_by_lower_mode_threshold():
    modes=np.array([[1.,2.,3.],[4.,5.,6.]])
    for unusable,association in ((False,.59),(True,.9)):
        actual=dual_gate_correction(modes,[.99,.99],association,.6,.2,1.,unusable)
        np.testing.assert_array_equal(actual,[0.,0.,0.])
    actual=dual_gate_correction(modes,[.3,.1],-.8,.6,.2,.5)
    np.testing.assert_allclose(actual,.5*modes[0])
