# AutoVMD and regional band routing execution checklist

Campaign: `vmd-bandroute-20261010`. The user authorized paired neural research after data/numerical readiness, independently of classical 15 dB qualification. Older campaign artifacts and gates remain immutable.

All numerical tests, decompositions, optimization, plots and model evaluation run on Kaggle. Source modules live in Git; notebooks pin the exact published SHA at launch. This document records implementation status, not measured scientific success.

## Delivery checklist

- [x] Separate configuration, authorization and explicit checksummed legacy imports.
- [x] Require the exact original study04 restoration archive; preflight development source paths/hashes.
- [x] Persist native per-record progress, spectra and chance banks.
- [ ] Verify repaired source loading and complete paired-data readiness on Kaggle.
- [ ] Neural contracts: decomposition closure, identity, masks, variable C/K, permutation and optimization.
- [ ] Forty-setting all-frontal AutoVMD caches, balanced across conditions/input levels.
- [ ] Source-excluded successive screening (40/8/2 candidates; 5/15/80 epochs).
- [ ] Fixed-band, regional VMD, no-context, all-VMD and raw-only matched neural experiments.
- [ ] Paired TCN-BiGRU deployment baseline and loss profiles.
- [ ] Grouped five-outer/three-inner-fold selection and shortlisted seeds 42/3407/2026.
- [ ] Optional qualified, nested cross-fitted teachers and student distillation.
- [ ] Native OSF, legacy Klados and separate EEGdenoiseNet benchmark.
- [ ] Frozen model/analysis; one reserved and independent Magdeburg evaluation.
- [ ] Fresh CPU/ONNX parity, whole-record inference and measured 10/20/30/40/50-channel scaling.
- [ ] Final report with achieved SNR, K/alpha evidence, modes/bins, regional benefit, complexity and limitations.

## Design and scientific boundaries

Research hybrid: raw full-band path; VMD plus residual on verified frontal channels; complete Fourier partition on posterior/shared/unknown channels; soft time/component routing; signed masked frontal context; one artifact subtraction. Mode identities use physical frequency/bandwidth/energy and component type rather than assumed IMF index. K=5 is not justified by five EEG frequency bands.

AutoVMD freezes one recipe per development fold and final refit. Parameters do not change during epochs or seed repetitions. Every cache saves source/window/preprocessing/solver identities and convergence/failure diagnostics. Failed decompositions remain scored using the trained raw fallback.

Deployment student: shared width-32 five-block TCN, BiGRU 32 per direction, signed frontal pooling, zero-initialized regional artifact heads. No EOG, VMD or spatial fitting is required at deployment. This is an offline quality reference, not causal streaming.

Primary evaluation follows VMD papers and EEGOAR-Net/OSF, augmented with the explicit energy-ratio output SNR target and separately labelled engineering preservation gates. Native unpaired EEG has no reconstruction SNR. Klados channel/participant identities remain unknown. Controlled LEMON references are low-ocular rather than proven artifact-free; regression-derived donor fields may favor regression.

Fresh controlled mean output SNR >=15 dB, stretch 20 dB. Engineering gates: worst-source mean clean modification <=1%, clean alpha/beta error <=0.5 dB, relative clean covariance error <=0.02, paired-correlation deterioration <=0.005 against the declared comparator. Failure does not permit threshold relaxation.

## Launch and recovery

Use `python tools/kaggle_campaign.py submit --config configs/bandroute.json --stage <stage> --run-id <unique-id> --kernel-source <owner/slug/version>` with immutable parent versions and experiment JSON when needed. Use separate kernel slugs for active shards. Every output records the exact source, configuration, split, environment and parent identities.

Readiness permits paired research only. Teacher eligibility and final freeze are separately checksummed gates. The Luna monitoring agent observes kernel status; the primary engineer diagnoses and publishes repairs. API outages are distinct from failed kernels. Never silently use another saved version's outputs.
