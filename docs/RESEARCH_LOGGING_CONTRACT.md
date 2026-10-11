# Continuing research documentation

The user requested a complete continuing ledger in [Work So Far.md](<../Work So Far.md>).

For each meaningful experiment or repair, append a narrative entry **before** the
`<!-- BEGIN GENERATED EVIDENCE -->` marker. Include UTC timestamp, question,
method and exact parameter changes, run ID, published SHA, notebook version,
versioned parents, source/split identities, calibration/inference information,
record/channel scope, tests, complete metrics, failures/pass-through/exclusions,
runtime/memory, interpretation and next decision. Record hypotheses before
testing and retain negative outcomes. Do not replace historical entries with
later claims.

Distinguish launched, running, complete, retrieved, checksum-verified and
qualified. A local output directory is not live status evidence. A completed
kernel is not qualified accuracy. Native EEG has no reconstruction SNR without
clean truth. Regional dB values cannot be added. Stopped AutoVMD is historical.

Refresh locally available evidence with `python tools/build_work_ledger.py` from
the project root. The script reads saved metrics/metadata and hashes files; it
does not perform numerical EEG work or poll Kaggle. Original result artifacts
remain authoritative. Large metric tables are indexed losslessly, not rewritten.
Never alter the other chat's `posterior results.md` as part of ledger refresh.
