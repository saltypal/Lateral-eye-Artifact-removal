# Fixed nonfrontal strength and blend diagnosis

14October2026. This is a bounded exploratory experiment using immutable full001
predictions. It does not select a final teacher or open confirmation data.

## Evidence and hypothesis

Frozen review found VMD+rank2 whole mean-channel SNR15.155692dB but pooled-window
energy SNR13.480444dB. Raw-frontal+same rank2:15.399070 and13.686979dB.
Verified middle14.141899dB versus temporal17.452148dB. Severe lateral and weak
blink/mixed cases remain bottlenecks. The current method fits the map using
calibration but estimates region gain on scoring EEG; excessive gain is one
possible cause. It is not assumed before error diagnosis.

## Fixed target-free arms

Four unchanged controls:identity,oldVMDcontext,oldrawcontext,rank2joint.
Six new arms use predeclared weights0.25/0.5/0.75:

- strength:artifact=weight*saved_rank2_artifact on nonfrontal rows.
- blend:artifact=(1-weight)*old_nonfrontal_artifact+weight*rank2_artifact.

Frontal VMD is unchanged. Both fields refer to the same original input. Apply one
subtraction; unknown/masked rows pass through. No input-SNR label, target,
condition, participant outcome or evaluation gain enters these fixed recipes.
No VMD, ICA or new calibration fit runs. Static inspection locally; numerical
fixtures and execution on Kaggle only. Fifty-example pilot precedes full320.

## Evaluator-only bounds

After every output is generated, use paired references to calculate an optimistic
scalar gain bound(0..8) and old/rank2 blend bound(0..1), separately for posterior,
central/shared,true middle and temporal. These are nondeployable diagnostics
and cannot determine per-window inference weights. Save their coefficients only
in an explicitly evaluator-only report, never a recipe/checkpoint. Close current
residual energy into channel mean-bias and centered error energy. If even the
optimistic scalar cannot help, strength alone cannot fix field shape.

## Evidence and decisions

Verify exact full001/1 identity/manifest/split and every prediction/decision.
Check archived paper metrics, frontal/unknown parity, weights, masks, scalar
accounting and failure lineage. Keep historical parent flags labelled separately
from new failures. Save all fits(no new ones), progress,predictions,per-channel
metrics,condition/input tables, middle/temporal results and source-aware intervals.
Report channel-first mean and pooled-window SNR side-by-side; no aggregate swap.
Paired paper MSE/RRMSE/CC remain primary; fixed gains do not fit clean targets.
Clean engineering gates are unchanged and strongest raw frontal failure persists.

No globally selected weight is claimed from this exposed pilot/full cohort.
Grouped recipient/donor-excluded tuning and native qualification are later stages.
A successful weaker-artifact correction must not degrade severe lateral cases.
If these fixed variants fail, retain rank2 and move to distinct reference CCA or
eligible SGEYESUB; do not keep expanding a blind strength grid. No models yet.
