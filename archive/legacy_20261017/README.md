# EEG ocular artifact removal

**Active implementation: [active/README.md](active/README.md).**

The replacement campaign uses **Klados and original OSF only**. Its source,
configuration, tests and single Kaggle notebook are isolated under `active/`.
Start with [the execution checklist](active/PLAN.md). Numerical execution stays
on Kaggle; local work handles source, Git, submission and retrieval.

The [legacy archive](archive/README.md) preserves the prior implementation at
Git `ca91a42bcd741d69af22d1d97c45f8eb8e1d6dc4`, including AutoVMD. Existing root
source/notebook paths are dormant compatibility paths; they are not dependencies
of the replacement. Old results and other chats' documents remain in place.

Klados channel order is unverified, so paired Klados results are nonregional.
OSF has verified electrode labels but no clean target, so its regional results
use suppression and preservation measures rather than reconstruction SNR.
Historical 17.9272 dB came from VMD + EOG + a small learned model on reused data.

```powershell
python active/tools/kaggle.py submit --run-id klados-osf-pilot-001 --pilot
python active/tools/kaggle.py status
python active/tools/kaggle.py retrieve --run-id klados-osf-pilot-001
```

The committed notebook is a readable template. The submission tool embeds the
exact published SHA in the actual Kaggle notebook before launch.
