# Retain 17.11 dB and target lateral reconstruction error

14 October 2026 IST. Planning deliverable only: no new numerical run or model
launched. This replaces an open-ended sequence of speculative correction variants
with a bounded, residual-driven decision. The separate reference-coupling repair
remains a data-consistency requirement, not a promised accuracy improvement.

## Verified starting point

| Method | Frontal controlled development output SNR | Pearson CC |
| --- | ---: | ---: |
| Original calibrated full-field VMD gain | 17.112814 dB | 0.973700 |
| Matched calibrated raw gain | 18.193247 dB | 0.981142 |
| Rejected DC-preserving selected-mode artifact shape | 2.521295 dB | 0.644872 |

The last row was an isolated challenger, not a replacement for the first row.
Its projection discarded useful estimated correction. Preserve the original
full-field subtraction and reconstruction residual. The development comparison
has 320 windows, 32 recipients, five donors and 13 verified frontal channels.
It is regression-generated and development-exposed, not fresh independent truth.
The strongest-comparator CC deterioration is 0.007441, exceeding the 0.005 gate;
older weaker-comparator pass statements do not establish current qualification.

| Condition | Nominal -5 dB input | Nominal 0 dB input | Nominal +5 dB input |
| --- | ---: | ---: | ---: |
| Blink: original VMD | 16.842724 | 20.062975 | 22.110642 |
| Lateral: original VMD | 8.954402 | 13.129189 | 16.622938 |
| Mixed: original VMD | 15.992406 | 19.100446 | 21.199606 |
| Lateral: matched raw | 11.028577 | 15.704485 | 19.374809 |

All table entries are output SNR. The input labels describe mixture construction;
they are not measured frontal per-channel input SNR. Lateral reconstruction is
the priority. Native resting modification/covariance still fail qualification.
Whole-montage matched VMD context was 13.162704 dB, not 17.112814 dB.

## Fixed constraints

Keep frontal-only K=3, alpha=2000; AutoVMD and broad searches stay stopped.
Keep the original corpus immutable, frozen recipient/donor partitions and closed
confirmation sources. All numerical work runs on Kaggle with pinned published
source and versioned parents. Use paper MSE, temporal RRMSE and ordinary CC;
report output SNR and improvement separately. Apply the existing preservation
gates against the strongest declared matched comparator. No target enters inference.

Do not repeat separate HEOG/VEOG gains, event-mask banks, selected-mode field
projection, DC shape projection or the unsuccessful frozen neural transfer as
new hypotheses. Their negative results are already documented.

## Bounded execution and decisions

| Task | Inputs and deliverable | Acceptance / next decision |
| --- | --- | --- |
| SI01 Freeze baseline evidence | Exact original/raw predictions, fields, modes, decisions and grouped identities | Recoverable parent checksums and baseline parity; no changed predictions |
| SI02 One lateral residual diagnosis | Saved development predictions and evaluator-only clean targets; report trigger misses, field-shape mismatch, mode-related gain bias and spatial gain disagreement separately | Close the error accounting and identify which limitation dominates; diagnostic target-fitted gains are labelled optimistic bounds and never deployed |
| SI03 Repair reference coupling | RC01-RC05 source/baseline/lag contracts and separately named tiny corpus pilot | Consistent EEG/reference excursions; no benchmark substitution or claimed independent validity; if original unscored sources cannot be verified, report that dependency and do not reconstruct them by assumption |
| SI04 One conditional classical challenger | Choose only from the measured SI02 mechanism; keep full artifact field | Pass focused fixtures and a small source-grouped development pilot before broader evaluation |
| SI05 Matched development comparison | Identity, original VMD, strongest raw, challenger and the same challenger without VMD | Better lateral reconstruction without overall accuracy/CC or clean-preservation regression; uncertainty must support a VMD benefit before claiming one |
| SI06 Native preservation check | Unchanged-reference blink/lateral proxies and resting modification/spectra/covariance | Existing gates unchanged; low EEG-EOG correlation alone cannot qualify the system |
| SI07 Regional integration | Qualified frontal correction and other chat's versioned posterior candidates | Disjoint masks, original-input artifact estimates, one subtraction and whole-montage scoring; never add regional dB |
| SI08 EEG-only student | Selected qualified classical system and grouped paired targets | Classical-first sequencing retained; compact temporal CNN/TCN baseline before a recurrent model, with matched no-VMD control and teacher supervision only when it improves paired-only learning |

### SI02 diagnosis contract

Do not launch another artifact-shape recipe simply because its name sounds more
advanced. On saved lateral cases, compare the current inferred scalar gain with
an evaluator-only least-squares scalar bound for the same full field. Then use
an evaluator-only left/right/midline bound to determine whether one global gain
is the bottleneck. Report remaining error after each bound by recipient, donor,
input level and condition. Include untriggered and decomposition-failed cases.
Bounds fit scoring targets and are optimistic diagnostics, not validation scores.

If the full field cannot explain the true controlled artifact even with an
optimistic scalar, changing gain alone is insufficient. If grouping adds no
material headroom, do not implement spatial gain fitting. Saved metadata must
verify anatomical groups; insufficient grouping means the hypothesis is ineligible.

### SI04 conditional challenger

If regional gain mismatch dominates, test partially pooled left/right/midline
gains regularized toward the original global gain. This changes spatial sharing,
not the previously rejected independent HEOG/VEOG-axis recipe. Scoring fits may
use EEG/references and verified frontal modes, never paired targets. Fit any
shrinkage selection only within grouped development training/validation sources;
preserve the complete calibration-derived artifact field. Risks include fitting
neural activity and calibration mismatch; clean and native gates are mandatory.

If the diagnostic instead identifies transfer-waveform mismatch, repair the
calibration transfer representation first with a matched raw control. Declare
one bounded hypothesis before coding; do not automatically open a new lag grid.
If selected-mode observations bias gain while raw observations are accurate,
report that VMD currently hurts this stage. Reusing raw observations is a valid
control, but does not establish that VMD improves the method.

If neither diagnosis supports a credible classical improvement, stop this
challenger sequence. Record a concrete proposal to change classical-first
sequencing for paired neural research rather than silently starting training or
weakening teacher/native qualification requirements.

## Stop rules and scope

Run one diagnostic notebook and at most one supported challenger pilot before
reassessment. A completed audit or passing software test is not an SNR gain.
Do not promote a pilot trained/selected on its scoring sources. Retain all
failed arms and per-record predictions. The 15/20 dB targets remain targets;
there is no guaranteed route to 20 dB or claim of production qualification.

An independent paired benchmark is still unresolved. Do not let its metadata
search replace the bounded development improvement experiment, and do not call
the reference-coupling repair an independent benchmark.
