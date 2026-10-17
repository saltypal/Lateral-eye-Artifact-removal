# Legacy implementation: preserved 17 October 2026

`LEGACY_MANIFEST.json` identifies the complete tracked-source ZIP, its SHA256,
the original Git SHA and the unchanged AutoVMD module hash. The ZIP preserves
the old source, notebooks, configurations, tests, reports and work ledger.

Historical generated outputs remain in `../results/`, excluded from Git as
before. They are not deleted, recomputed, or treated as new results. The old
root-level source and orchestration paths remain dormant for compatibility
with other open chats and artifact recovery; the replacement `../active/`
package does not import them.

LEMON-derived analyses remain historical evidence only. They are excluded from
the new campaign. AutoVMD code and caches remain unchanged. No automatic search
or screening queue is launched by the replacement notebook.
