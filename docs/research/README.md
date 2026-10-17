# Research documentation

Open **[Methods and Results.html](Methods%20and%20Results.html)** in a browser for the portable report. All 54 plots are embedded, so the HTML can be shared by itself. The editable [Markdown report](Methods%20and%20Results.md) uses the adjacent assets directory.

The report contains all nine requested subjects: datasets and label provenance; VMD/AutoVMD/history; regional grouping; regression limitations; frontal papers and adaptations; frontal accuracy and native preservation; nonfrontal candidates/results; measured regional composition; and the proposed student.

The [evidence manifest](evidence_manifest.json) records source runs, commits, paths and hashes. [RESULTS_snapshot.md](RESULTS_snapshot.md) freezes the result ledger used during assembly. This is a documentation snapshot, not an assertion that pending experiments have finished.

From the project root, rebuild using the installed `mistune` package:

```powershell
python tools/build_research_document.py
```

The builder verifies the six parent runs' retrieval status, checks every included PNG against its original artifact manifest, copies plots byte-for-byte, embeds them in HTML and checks section/image counts. It performs no signal processing, training or numerical plotting. Rebuilding refreshes the result snapshot; review prose if new experiments have completed.

Actual executed notebooks:

- [Frontal six-method comparison](../../notebooks/executed/frontal-six-full-001.ipynb)
- [Native frontal correlations](../../notebooks/executed/frontal-osf-correlations-001.ipynb)
- [Final mode correlation audit](../../notebooks/executed/frontal-mode-clean-audit-002.ipynb)
- [Central/temporal comparison](../../notebooks/executed/central-temporal-001.ipynb)
- [Frontal/posterior composition](../../notebooks/executed/frontal-posterior-combined-001.ipynb)

The old native mode-correlation plot is deliberately excluded because final audit version 2 supersedes its reference definition. Other original failures and warnings are retained in the report.

- [Accepted SGEYESUB/REG-ICA frontal extension](../../notebooks/executed/frontal-spatial-full-002.ipynb)
