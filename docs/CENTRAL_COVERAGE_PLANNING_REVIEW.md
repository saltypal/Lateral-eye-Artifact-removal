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
