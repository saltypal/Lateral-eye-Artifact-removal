# Failed-run evidence and repairs

## corpus-001 → source-fixture-001 → repaired corpus

`corpus-001`, source `38b580b01de95f89439fa6f1eb48985a8dd3d775`, failed with zero eligible controlled examples. All 32 development raw LEMON archives failed while MNE opened marker files referenced in BrainVision headers (e.g. participant sub-032305's header expects sub-010006.vmrk). Legacy caches were retained; eight confirmation LEMON participants stayed closed. This was a real kernel exception, not an API status error.

Focused diagnosis/repair checks actual archive filenames on Kaggle. A temporary loading header may resolve a missing declaration to one uniquely identified same-directory renamed companion. The original header, binary EEG and marker payloads remain unchanged; original and repaired header hashes and every resolution are recorded. Ambiguous/missing companions fail rather than inventing annotations. The full corpus resumes only after the one-source fixture succeeds.

`source-fixture-001` passed at `ce959d81af7a22328b6507933d0c9ece9435920f`: 26 tests passed; sub-032305's archive contains sub-032305.eeg/.vmrk/.vhdr but declares sub-010006.eeg/.vmrk. Both links were resolved; original payload hashes were recorded. MNE loaded 61 EEG channels and VEOG at 2500Hz; the unscored prefix resampled to [61,24000], with four eligible scoring windows. No confirmation participant was opened. The repaired full corpus is `corpus-002`.
