# Frozen regional error diagnostic: full development results

The full diagnostic completed on Kaggle version 1 at exact source
`220e83d1b713cff862cf800fe4dc978dc4b87d5b`. It examined 320 frozen examples
from 32 recipients across nine saved arms. All 134 tests, 988,200 channel-metric
parity checks, 2,880 whole pooled checks and 162 archived balanced-mean checks
passed. Twenty-two retrieved payloads were checksum-verified. Diagnostic
runtime was 173.21 seconds, excluding setup, parent verification and tests.

For the frozen K3 calibration-mode-gain + direct posterior arm:

| Region | Channel-first recipient-balanced output SNR dB | Mean per-example output-error share |
|---|---:|---:|
| Frontal | 17.112814 | 2.9326% |
| Posterior | 13.541703 | 9.1608% |
| Central/shared | 1.954134 | 87.9066% |
| Whole cap | 9.094213 | 100% |

Central/shared outputs are exact pass-through. This cohort has no unknown
channels. Error shares are first computed per example, then balanced over
recipient, condition and input level; they are not ratios of global sums.
The supplementary mean per-example whole pooled-energy SNR is 5.565526 dB.
These differently aggregated numbers must not be substituted for one another.
Regional decibel results cannot be added.

This confirms a coverage bottleneck. It does not establish that removing all
central error is feasible without losing neural activity. Existing frontal
CC deterioration versus the declared raw-regression comparator is .00744148,
which exceeds the .005 project gate. Native preservation remains unresolved.
No weights were trained, no corrections changed, no AutoVMD executed and no
reserved sources opened. Scientific qualification remains false.

Next: the four-arm central/shared extension specified in
[CENTRAL_COVERAGE_PLAN.md](CENTRAL_COVERAGE_PLAN.md). Its frozen calibration
reference arm must disclose the EEG/EOG amplitude mismatch in the controlled
mixture construction. Its context arm must derive calibration frontal fields
without fitting scored targets. All failures remain scored and saved.
