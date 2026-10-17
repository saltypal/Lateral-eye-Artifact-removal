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

## Pretrained-only frontal extension

User requested EEG-WNet and Diff-ADN with released weights only, without training
or guessed architectures. Discovery source snapshot/configuration and literature
notes are committed. Successful exact/compact-name GitHub/Hugging Face searches
found no matches; access failures prevent claiming all supplements were inspected.
WNet XML supplies metadata only, IOP redirects to a bot challenge. Neither
compatible code/checkpoint pair is verified. Both are explicitly not run.
Source266657f, runfrontal-pretrained-closeout-001, CPU Kaggle kernel
satyapaladugu/frontal-pretrained-closeout/1. The notebook verifies archived
frontal comparison parents and produces closeout tables/figures; it creates
no new predictions. Numerical completion/recovery remains pending at this entry.

### Pretrained extension completed

Version1 source266657f passed35 contracts, then failed the source discovery
JSON byte check because WindowsCRLF and GitLF differed. Retrieved6 failure
payloads including the actual2/3-cell notebook and error. Version2 source
a8c7b7171ad15b373422e233357f670159b0dee8 fixes text-snapshot canonicalization
with an added newline/content-change regression fixture; scientific parent
checksums stay byte-exact. It passed36 contracts, produced22 verified payloads,
3/3 executed cells,2 figures,4 HTML tables and zero errors. Both figures were
visually inspected. Actual notebook: notebooks/executed/frontal-pretrained-closeout-002.ipynb.
New models run:0. EEG-WNet and Diff-ADN remain unavailable for verified
pretrained-only reproduction; no fake checkpoint, replacement or training.
Existing scores and native preservation failure stand unchanged. The frontal
comparison/review campaign is complete with limitations; production
qualification and regional integration have not been achieved.

## Central/temporal stage launched

User confirmed FC/C/CP central and separate FT/T/TP temporal results. Source
20584ab, Kaggle CPU central-temporal-eog-comparison/1, runcentral-temporal-001.
Identity plus calibration-only ridge0.001 and GEVD–MWF rank1 at fixed0.25/1
strengths. Fits, masks, metrics and predictions remain separate per group.
No new VMD, neural training, frontal communication, LEMON or confirmation.
Use eight existing Klados record identities with independent calibration/score
filtering and four development OSF participants with whole-trial boundaries.
Numerical results and retrieval remain pending at this chronological entry.

### Central/temporal numerical outcomes

Source20584ab completed eight Klados records and four OSF participants with
38contracts, zero calibration failures. Central native21 FC/C/CP channels;
temporal6 FT/T/TP channels. Standalone fits per group, no front/back context.
Central raw10.5743dB becomes ridge25%12.6422, ridge100%18.3343, MWF25%10.4977,
MWF100%9.0605dB. Temporal raw12.3977 becomes13.9872,16.7267,12.2443,9.8424dB.
All corrective preservation gates fail. Central native mean rest changes3.55%,
14.21%,7.92%,31.70% respectively; worst-participant ridge25%4.06% above1%.
Clean-input worst-record ridge100% central modification89.35%, temporal136.29%.
Record16/start3024 central ridge100%−1.073532dB and temporal−4.110142dB remain
in the results; means do not establish reliable correction. No neural-loss
percentage is inferred from relative signal modification. Eight figures
produced; SNR, lateral correlation and both preservation plots visually
inspected. Full details and exact run links are in RESULTS.md.

Recoverycomplete:248/248 verified payloads, actual3/3-cell notebook,eight plots,
four tables,zero errors. Kaggle status COMPLETE. The central experiment is
finished with negative preservation findings; no corrective winner was
qualified or composed with frontal output. Pass-through remains the working
default. All source/numerical identities and failure evidence are preserved.

## Frontal-first original SGEYESUB/REG-ICA extension

User approved full-montage fitting and then explicitly requested frontal
comparison before further central/posterior execution. SGEYESUB/REG-ICA use
full EEG spatial information (19Klados under disclosed mapping,58nativeOSF),
while only frontal outputs are scored. This is not VMD-to-posterior messaging.
SGEYESUB author commit2c95b4f, original EEGLAB Infomax8ac485f and checksum-
verified SCCN REGICA archive are invoked through Octave. Frozen calibration
ICA/FD and per-trial/window-reset SRLS are declared REGICA evaluation adaptations;
first2 component samples retained to avoid zero-initialization deletion.

Readiness runfrontal-spatial-readiness-001/sourcea77ca55 passed40 contracts,
real full58channel SGEYESUB calibration (all6originalsampleclasses) and original
REGICA runtime fixture.52/52 payloads recovered, actual3/3 executed cells,
one figure/table, zero errors; plot inspected. SGEYESUB runtime success is not
suppression/preservation qualification. Klados lacks the required six-class
calibration, so its exact SGEYESUB reproduction is not run.

Full runfrontal-spatial-full-001/sourcee8d0021 submitted as Kaggle
frontal-sgeyesub-regica/2. Paired REGICA uses exactly the saved frontal benchmark
scoring arrays and a new calibration-frozen ridge comparator, separately from
the old scoring-window regression. Both methods evaluate the four native OSF
development sessions on five common frontal channels with unchanged H/V
references. Clean targets never fit new models. Save failures, operators,
original ICA learning histories/caps, per-record outputs, preservation tables
and record/participant-cluster uncertainty. Full results remain pending here.

Subsequent regional literature shortlist is documented in
docs/NONFRONTAL_CANDIDATES_20261017.md: targeted waveletICA (published2025,
released RELAX code), author GEVD-MWF, and later separately scored SGEYESUB/
REGICA outputs. These are candidates, not new measured winners. Retired LEMON
posterior results cannot rank the active Klados/OSF comparisons. No new central,
posterior, neural, AutoVMD or LEMON numerical experiment is launched here.
