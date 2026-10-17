# Independent planning review — GPT-6.1 Sol, extra-high reasoning

Requested by the user on 13 October 2026. Read-only review of the approved plan,
interfaces and saved evidence. No numerical work was performed by the planner.

The four-arm experiment is defensible as a bounded development comparison.
The error diagnostic supports testing central coverage next; it does not
guarantee the 15 dB target is attainable with the preservation constraints.

## Findings and dispositions

1. Frozen frontal correlation deterioration is .00744148, above the .005 gate.
   Every primary arm inherits that failure. Report frontal/central/posterior/
   whole-cap gates separately. A whole-cap average cannot qualify the frontal
   branch or repair native preservation.
2. Unscaled EOG references do not identify the changing controlled EEG ocular
   amplitude. Reference-only correction can overcorrect clean controls while
   detecting a real donor eye event. Preserve that outcome; never use stored
   mixture amplitude or targets to adjust correction. Frontal estimated-field
   context tests a different source of strength information.
3. Valid zero resting chunks must remain in OOF calibration. Implemented:
   retain individual zero chunks, require nonzero aggregate context, and cover
   this case in a Kaggle fixture.
4. Persist computed metrics per example before advancing. Implemented:
   channel and example metric shards alongside waveform/decision shards.
5. Four chunks limit event coverage. Leaving out an event chunk can remove the
   corresponding training event. Keep offsets, types, decisions, usable
   frontal counts and fit failures visible. Do not swap chunks after viewing
   scoring outcomes or increase the cap silently.
6. The full diagnostic is complete. Updated the active checklist and linked
   its verified report; the central pilot is the immediate experiment.

## Execution order and stop rules

Source/config/notebook publication → Kaggle tests and 50-example pilot →
checksummed recovery and review → same-recipe 320-example development run →
native preservation/suppression → grouped qualification review.

Numerical contracts, exact parent identities, correct anatomy and frozen
waveform parity control dependent execution. Scientific failures remain
measured results, with `qualification=false`; they do not cause hidden gate
retuning. An all-unavailable context arm is an unavailable comparison, even
though its pass-through output may match baseline metrics. Neural research
and confirmation stay closed under this campaign's explicit scope. AutoVMD
remains stopped; unknown and masked channels remain unchanged.

## Follow-up after the verified pilot

The pilot's whole-cap context result is 13.994230 dB against its matched
10.002235 dB frozen regional baseline. Central/shared output SNR is
11.121655 dB. These development results support testing frontal-context
transfer, but the existing raw-frontal comparator leaves central channels
unchanged and therefore cannot establish VMD's contribution to that transfer.

The planner recommends this order:

1. Finish the unchanged 320-example run and verify identities, checksums,
   numerical contracts, frozen-branch parity and all inherited failures.
   Report recipient/donor-aware paired uncertainty on Kaggle before making
   an improvement claim with uncertainty.
2. Add one matched raw frontal-artifact context arm. Keep the four calibration
   chunks, leave-one-chunk-out fitting, boundaries, association threshold,
   shared gain, lag bank, penalty and five signed summaries unchanged.
   Use raw calibration transfer without VMD bundles to form its held-out
   predictors and fit a separate central ridge. At scoring, obtain raw
   context from original frontal EEG minus the archived raw-calibration
   frontal output. Keep scored frontal and posterior outputs identical
   across both context arms. No scoring target may enter this process.
3. Address native frontal preservation with the separately documented
   shared-gain/separate-axis-support ablation and matched raw control.
   Central coverage gains do not repair frontal resting modification,
   covariance or correlation failures.

Keep method-specific failures and fit availability in complete-cohort results.
Run leakage, masks, parity, zero-rest, boundaries, identity and persistence
fixtures before the new context comparison. Do not increase calibration
coverage or introduce a new recipe bank before this attribution comparison
and the frontal safety checkpoint. If raw context matches VMD context,
attribute the evidence to coverage and strength context; VMD's added value
remains unestablished. No planner recommendation changes the active full run.
