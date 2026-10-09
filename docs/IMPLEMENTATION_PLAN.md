# High-level and low-level execution plan

## Objective and boundaries

Recover recipient EEG from blink and lateral-eye contamination, measure source-level uncertainty, and retain declared clean EEG. Controlled SNR >=15dB is the target; 20dB is a stretch target. Numerical work, tests, figures and training run on Kaggle. Git modules are the source of truth. The user conditionally authorizes neural work only after every classical gate passes.

## High-level flow

1. Verify original source bytes and metadata; preserve historical exposure labels.
2. Freeze source identities, reserving 20% of new cohort identities with seed 42 before extracting windows.
3. Validate signal and solver contracts; create independent calibration/scoring segments.
4. Generate controlled recipient/donor mixtures and retain native real EEG separately.
5. Search forty VMD K/alpha settings, correction parameters and posterior algorithms.
6. Select configurations outside each held-out recipient/donor bucket; compare regional, shared and no-context alternatives.
7. Review all preservation, correlation and SNR gates. Require VMD value beyond matched direct regression.
8. After a passing gate: source-isolated teacher targets, EEG-only baseline, distillation and grouped multi-seed model selection.
9. Freeze exactly one model and analysis; open reserved and independent evaluation once.
10. Verify CPU loading, whole-record overlap-add and cap scaling; deliver artifacts and limitations.

## Module contracts

| Module | Input | Output | Dependencies | Owner | Acceptance |
|---|---|---|---|---|---|
| `contracts.py` | run specification, approval evidence | validated immutable identity | strict JSON | primary agent | invalid identities and unauthorized stages rejected |
| `io.py` | versioned source bundle | verified original source tree, hashes | archive manifest | primary agent | checksum mismatch/path escape rejected |
| `osf_reader.py` | original `.set/.fdt` | `[trial,channel,time]`, names, integer labels | publisher format | primary agent | EEG/EOG/annotation axes and FDT length verified |
| `data.py` | original files and IDs | audit, source partitions, development caches | I/O, reader | primary agent | metadata-backed EOG axes; reserved cache absent |
| `signal.py` | one continuous segment | 200Hz 0.5–40Hz EEG, overlap-add correction | scipy | primary agent | no trial/calibration joins filtered together; identity overlap-add |
| `reference.py` | EEG `[C,T]`, EOG `[2,T]` | signed association and artifact projection | lag convention | primary agent | lag oracle, polarity invariance, collinearity |
| `vmd.py` / `frontal.py` | one channel plus references | `[K,T]` vectors, residual, selective correction | reference implementation | primary agent | vmdpy parity; retain residual; record convergence |
| `posterior.py` | unscored calibration and original scoring EEG | ICA/MWF artifact on posterior channels | scipy/Picard | primary agent | analytic GEVD fixtures; sufficient rank/rest/ocular duration |
| `corpus.py` | recipient prefix/scoring, donor calibration/scoring | controlled mixtures, clean controls, recipes | frozen source partitions | primary agent | exact-name montage intersection; recipient/donor held out together |
| `experiments.py` | archived corpus and search artifacts | full tables, grouped comparison, strict classical gate | classical modules | primary agent | all eligible failures retained; no target-dependent inference |
| notebook builder | stage and published SHA | clean checkout/execution notebook | Git remote | primary agent | actual SHA recorded and matches launch SHA |
| Kaggle CLI launcher | clean published branch | versioned kernel, run manifest | Kaggle OAuth | primary agent | no stale source; output identity checked |
| monitor | kernel status/logs | completion/error reports | Kaggle API | GPT-6 Luna high | distinguish API errors from numerical failures |
| neural modules | passing gate, approved recipe | EEG-only model and export | not implemented before gate | primary agent | identity/mask/permutation tests and final independent evaluation |

## Research controls

OSF has no paired native clean signal. Its globally unique participant IDs are documented by the publisher readme. Klados's unknown montage and participant mapping remain unknown. Raw LEMON is a low-ocular recipient reference, not guaranteed biological clean EEG. Controlled artifacts are spatially coherent donor-calibration projections; this family can favor regression, so superiority cannot be inferred from its SNR alone.

Configuration selection excludes the target recipient/donor bucket. Window-adaptive EOG projection uses no clean target. ICA/MWF use unscored calibration only. Modes are adaptive vectors, not physiological bands. Close centers are recorded as diagnostics. Iteration-limited/error fits pass through and remain scored. Regional experts all see original input; their disjoint corrections are subtracted once.

## Execution checkpoints and repairs

Every stage records source SHA, configuration, split identities, parent evidence, package/hardware information, logs, hashes and status. New source revisions require a new Kaggle notebook version/run ID. Failed evidence is retained. A focused contract check precedes dependent execution. Reproduction uses archived SHAs; latest is resolved once at launch.

The live completion ledger is `TARGETS.md`. An authored module is not marked research-validated merely because it parses or its kernel runs. Parameter searches require result inspection; classical gate failure keeps models closed and never opens confirmation data for tuning.
