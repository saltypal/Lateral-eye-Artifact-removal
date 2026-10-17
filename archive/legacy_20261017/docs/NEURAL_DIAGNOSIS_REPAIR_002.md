# Diagnosis dispatch repair

Failed run: `neural-diagnosis-001`, source `11da0d6`.

The Kaggle job passed all 78 existing tests, then the runner routed
`neural-diagnosis` into the legacy campaign. That dispatcher raised
`NotImplementedError` before any diagnosis was performed. This is a software
integration failure, not a failed preservation measurement.

The repaired runner uses the neural campaign's single list of implemented
stages. A regression test covers routing each implemented research stage.
Numerical tests and the actual diagnosis run on Kaggle; local checks are
syntax compilation and Git whitespace checks only.

Replacement: `neural-diagnosis-002`, a new version of the same Kaggle notebook.
The original run specification and failure log remain archived. Dependent
preservation pilots require the repaired run to finish successfully. No
scientific gate or scoring definition changed.
