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
| frontal-osf-correlations-001 | b9df0e6 | Completed: 31 contracts, four participants/32 trials; 240 verified payloads, 16 figures, 5/5 cells; native preservation failure |
| frontal-mode-clean-audit-001 | 405d7aa | Completed: 32 contracts, 2005 paired component rows; 22 payloads, three figures, 3/3 cells |
| frontal-mode-clean-audit-002 | 0876931 | Completed: 32 contracts; all 1280 native prediction rows verified and 5071 component rows repaired; 25 payloads, three figures, 3/3 cells |

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

Completed and recovered: the native OSF run uses source b9df0e6, run
frontal-osf-correlations-001. All 240 payloads verify; its actual executed
notebook contains five executed code cells, 16 figures, four tables and zero
error cells. Thirty-one numerical contracts passed. The independent Klados
check verifies 1512 clean-EEG correlations and writes names/source rows inside
labelled NPZ copies. Klados raw CC0.698520 becomes historical frontal CC0.984644
at output SNR16.134624 dB.

Native OSF uses four participants and 32 scored trials, five exact shared
frontal names F3/F4/F7/F8/Fz, frozen Klados models, separate MWF calibration and
unchanged H/V evaluation references. No native clean target exists. Historical
blink abs VEOG CC0.918516 becomes0.328867; lateral abs HEOG CC0.358906
becomes0.214818. Mean resting relative RMS modification is0.636283 (63.63%).
This fails preservation; it is not a measured percentage of neural signal lost.
The comparison does not qualify a native winner. Posterior/central are untouched.

Mode audit version1 (405d7aa) passed32 contracts, recovered22 payloads and
three figures, and checked2005 paired component rows against clean EEG/HEOG/VEOG.
Version2 (0876931) repairs5071 native mode/residual reference diagnostics and
independently verifies all1280 native prediction correlation rows. It passed32
contracts;25 payloads and the actual3/3-cell notebook with three figures were
retrieved and verified. No denoising/training was repeated.
See docs/WORK_LOG.md for the exact mismatch and retained original evidence.
