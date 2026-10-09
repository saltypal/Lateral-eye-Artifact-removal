# Failed-run evidence and repairs

## corpus-001 → source-fixture-001 → repaired corpus

`corpus-001`, source `38b580b01de95f89439fa6f1eb48985a8dd3d775`, failed with zero eligible controlled examples. All 32 development raw LEMON archives failed while MNE opened marker files referenced in BrainVision headers (e.g. participant sub-032305's header expects sub-010006.vmrk). Legacy caches were retained; eight confirmation LEMON participants stayed closed. This was a real kernel exception, not an API status error.

Focused diagnosis/repair checks actual archive filenames on Kaggle. A temporary loading header may resolve a missing declaration to one uniquely identified same-directory renamed companion. The original header, binary EEG and marker payloads remain unchanged; original and repaired header hashes and every resolution are recorded. Ambiguous/missing companions fail rather than inventing annotations. The full corpus resumes only after the one-source fixture succeeds.
