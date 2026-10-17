# Work so far

## Current experiment

The active experiment compares six frontal methods on Klados: historical
K3/alpha2000 VMD plus EOG model, SVMD–FCM adaptation, VME–GMETV adaptation,
Klados-fine-tuned EEGOAR-Net, GEVD–MWF and trained author-code BandRouteNet.
Frontal names follow the user-requested paper list order; row binding is an
explicit assumption. The same archived 37/8/8 record split and paired windows
are used. No LEMON or broad AutoVMD search is active.

The old historical model was reproduced at **17.927210 dB across all 19 rows**.
That is development-exposed EOG-assisted evidence. The new seven-frontal-row
comparison is measured independently and must not inherit that number.

## Evidence and explanations

- [RESULTS.md](RESULTS.md): current verified results and limitations.
- [PLAN.md](PLAN.md): exact six methods, checklist and posterior/central plan.
- [Method decisions](docs/FRONTAL_SIX_METHODS.md): dataset definitions, published
  settings, project adaptations, source references, reference access, missing
  parameters, metrics, numerical safeguards and executed-notebook recovery.
- [Work log](docs/WORK_LOG.md): setup failures, root causes, repairs, source SHAs
  and numerical run outcomes, including negative results.
- [Readable notebook](notebooks/Frontal_Comparison.ipynb): exact execution flow.

## Executed runs

| Run | Source | Evidence |
|---|---|---|
| klados-osf-pilot-001 | dfe4093 | Dependency-stack failure; repaired without changing scientific gates |
| klados-osf-pilot-002 | da7e277 | 20 contracts; historical replay; native OSF mode/reference and preservation diagnostics |
| frontal-six-smoke-001 | 53a8cd3 | 27 contracts; EEGOAR label-array adapter failure; executed error notebook recovered |
| frontal-six-smoke-002 | 025be10 | 29 contracts; original montage lacks FP1/FP2; executed error notebook recovered |
| frontal-six-smoke-003 | ed82c07 | 29 contracts; spatial-layer guard rejected frozen anonymous slots; error notebook recovered |
| frontal-six-smoke-004 | 8d51804 | 29 contracts; all six methods; one window/one epoch only; 88 verified payloads, 12 inline figures, 6/6 executed cells, zero error cells |
| frontal-six-full-001 | c196374 | Completed: 29 contracts, 24 windows/8 records; 922 verified payloads, 13 figures, 6/6 executed cells, zero error cells |

Results live under `results/<campaign>/<run_id>/`. Retrieval receipts state
execution completion separately from accuracy qualification. Each actual
executed notebook includes output tables/images and a verified source identity.
Failed runs remain preserved. Test records have been reused historically;
record separation does not establish participant independence.

## Historical ledger

The complete previous ledger remains at
[archived Work So Far.md](archive/legacy_20261017/Work%20So%20Far.md).
It includes AutoVMD, LEMON experiments, earlier regional challengers, bottlenecks
and negative evidence. Those entries are historical; the active method is
defined above. The physical relocation manifest and original source ZIP retain
the old code and hashes. Nothing in that archive qualifies the current system.

## Regional next step

After frontal selection, compare posterior pass-through, GEVD–MWF and ICA
component regression. Test one frontal-artifact transfer route (common blink
plus signed lateral context), fitted only on calibration. Central starts with
pass-through and conservative regression/MWF comparisons. Use disjoint region
masks and subtract the combined artifact once. Regional dB values are never
added. See PLAN.md for the detailed checks.

## Current frontal-only correlation extension

User requested explicit clean-EEG CC and native OSF before/afterHEOG/VEOG. Klados beforeCC0.698520, historical frontal afterCC0.984644 and outputSNR16.134624dB. The frontal-only extension independently recomputes CC from saved predictions and writes names/source row indices inside NPZ copies. Native OSF uses five exact common frontal names F3/F4/F7/F8/Fz, fixed Klados neural checkpoints, separate MWF calibration and unchanged H/V evaluation references. Sourceb9df0e6, runfrontal-osf-correlations-001; pending final recovery. No paired clean target exists in native OSF.
