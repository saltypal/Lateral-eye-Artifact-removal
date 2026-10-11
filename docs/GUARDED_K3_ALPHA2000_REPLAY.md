# Return to the archived 17 dB recipe

The user explicitly stopped K4/alpha1000 and the remaining forty-setting
screening queue on 11 October 2026. The local controller is stopped and the
queue is locked against accidental restart. Cancellation of the actual Kaggle
saved-version session is unconfirmed; manual stop has been requested because
the public cancellation interface needs a session ID that was not exposed.

The earlier 17.9272 dB result used **K=3, alpha=2000 plus EOGContextGainStudent**,
the frozen epoch-110/configuration-2 weights, and a hard support gate with
offset 0 and correction strength 1. Identity weight was .2 and training rate
.001. This is a complete EOG-assisted neural recipe; simply selecting K3/2000
for the current band router does not recreate that result.

Archived source: `0b183e6c09f6ecfbb3ed4818028eb5abbdf842fc`. The source checkpoint
comes from context source `2ec0b3cd3da636cc3aed3b476ff890a6f56e2627`, SHA256
`4bb7c1b0cfa0e8341c4053ab3099efb16832f045f99e37afde1904a93c2cd0a9`.

Notebook 23 replays the archived guard phase, including its declared twenty
validation inference rules and frozen neural weights; it does not retrain the
network. It pins all parent notebook versions and verifies that the reproduced
source-checkpoint identity and selected hard-gate recipe match the archived
ones. CPU execution avoids dependency on the stalled GPU run. Original
numerical fixtures, data audit, cache checks, predictions, metrics and plots
remain in the archived reusable Git modules.

The original score averages three windows within eight reused Klados records
and then records in dB, using all 19 unknown-mapped channel rows. Its raw and
frozen-VMD scores were 2.9021 and 9.4057 dB. The guarded pipeline needs runtime
HEOG, VEOG and VMD. It is development-exposed, lacks verified participant and
electrode provenance, and one record scored below 15 dB. It is neither a fresh
controlled result nor a verified frontal-VMD/posterior-ICA model. The archived
aggregation also differs from the current channel-first posterior protocol.

This replay must not open reserved sources, qualify the current teacher, or
replace the current campaign's preservation thresholds. A successful replay
establishes reproducibility of the old recipe, not its performance on LEMON/OSF
or EEG-only deployment. The next scientific step is a matched development-only
transfer comparison after reviewing the replay.

From the project root, after committing/pushing authored changes:

```powershell
python tools/guard17_replay.py --run-id guard17-replay-001
```

Use a new run ID for subsequent versions. Artifacts are saved under
`results/vmd-bandroute-20261010/<run_id>/` on Kaggle; the source SHA, parent
versions and separate orchestrator SHA are recorded.

