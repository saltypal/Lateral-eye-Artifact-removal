"""Assemble saved Kaggle figures and a portable research report; no EEG computation."""
from pathlib import Path
import base64
import hashlib
import html
import json
import re
import shutil
import subprocess

import mistune

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "docs" / "research"
RUNS = [
    ("klados-frontal-six-20261017", "frontal-six-full-001", "Paired frontal comparison"),
    ("frontal-correlations-20261017", "frontal-osf-correlations-001", "Native frontal and reference correlations"),
    ("frontal-mode-audit-20261017", "frontal-mode-clean-audit-002", "Corrected mode/reference audit"),
    ("central-temporal-20261017", "central-temporal-001", "Central and temporal comparison"),
    ("frontal-posterior-20261017", "frontal-posterior-combined-001", "Actual frontal/posterior composition"),
    ("frontal-spatial-20261017", "frontal-spatial-full-002", "Completed full-montage SGEYESUB and REG-ICA frontal extension"),
]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build():
    assets = DEST / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    source_path = DEST / "Methods and Results.md"
    source = source_path.read_text(encoding="utf-8").split("<!-- FIGURE_ATLAS -->")[0]
    entries = []
    sections = []
    figure_number = 0
    for campaign, run, title in RUNS:
        outer = ROOT / "results" / campaign / run
        inner = outer / "results" / campaign / run
        receipt_path = outer / "retrieval_receipt.json"
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if not receipt.get("all_manifest_payloads_verified"):
            raise RuntimeError(f"Unverified parent: {run}")
        if not receipt.get("execution_complete"):
            raise RuntimeError(f"Incomplete parent: {run}")
        original_manifest = json.loads((inner / "artifact_manifest.json").read_text(encoding="utf-8"))
        original_files = {item["path"]: item for item in original_manifest["files"]}
        sections.append(f"### {title}\n\nRun `{run}`; source `{receipt.get('git_sha', 'see receipt')}`.\n")
        for image_path in sorted((inner / "figures").glob("*.png")):
            if image_path.name == "10_native_mode_correlations.png":
                continue  # Its reference mismatch is superseded by the final audit.
            relative_image = image_path.relative_to(inner).as_posix()
            expected = original_files[relative_image]
            if digest(image_path) != expected["sha256"] or image_path.stat().st_size != expected["bytes"]:
                raise RuntimeError(f"Original manifest mismatch: {image_path}")
            figure_number += 1
            target = assets / f"{run}__{image_path.name}"
            shutil.copyfile(image_path, target)
            if digest(target) != digest(image_path):
                raise RuntimeError(f"Copy mismatch: {target}")
            caption = image_path.stem.replace("_", " ")
            scope = ("Reused paired Klados records; no fresh-confirmation claim."
                     if run == "frontal-six-full-001" else
                     "Read axis labels for paired versus native scope; native EEG has no clean-target SNR."
                     if run in {"frontal-osf-correlations-001", "central-temporal-001", "frontal-spatial-full-002"} else
                     "Diagnostic associations do not establish that a mode is pure artifact."
                     if run == "frontal-mode-clean-audit-002" else
                     "Whole-cap and regional aggregations are distinct; all combined qualification gates fail.")
            sections.append(f"![Figure {figure_number}: {caption}](assets/{target.name})\n\n"
                            f"**Figure {figure_number}. {caption}.** {scope} Original Kaggle output; no local replotting.\n")
            entries.append({"figure": figure_number, "run_id": run,
                            "source_git_sha": receipt.get("git_sha"),
                            "source": image_path.relative_to(ROOT).as_posix(),
                            "copy": target.relative_to(DEST).as_posix(),
                            "sha256": digest(target), "bytes": target.stat().st_size,
                            "retrieval_receipt": receipt_path.relative_to(ROOT).as_posix(),
                            "receipt_sha256": digest(receipt_path)})
    assembled = source + "<!-- FIGURE_ATLAS -->\n\n" + "\n".join(sections)
    source_path.write_text(assembled, encoding="utf-8")
    shutil.copyfile(ROOT / "RESULTS.md", DEST / "RESULTS_snapshot.md")
    manifest = {"date": "2026-10-17", "source_head_at_build": subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "figure_count": len(entries), "figures": entries,
        "results_snapshot_sha256": digest(DEST / "RESULTS_snapshot.md"),
        "report_sha256": digest(source_path),
        "excluded": ["10_native_mode_correlations.png: superseded reference diagnostic"]}
    (DEST / "evidence_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    render = mistune.create_markdown(plugins=["table"])
    body = render(assembled)
    def embed(match):
        path = DEST / html.unescape(match.group(1))
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        return f'src="data:image/png;base64,{encoded}"'
    body = re.sub(r'src="(assets/[^"]+)"', embed, body)
    page = """<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Regional EOG removal — Methods and Results</title>
<style>body{max-width:1120px;margin:40px auto;padding:0 24px;color:#182737;
font:17px/1.65 system-ui,sans-serif}h1,h2,h3{line-height:1.25;color:#113b56}
h2{border-top:2px solid #dbe7ed;padding-top:24px;margin-top:48px}table{border-collapse:collapse;
display:block;overflow:auto;font-size:14px;margin:24px 0}th,td{padding:9px 12px;border:1px solid #cbd6dd;
text-align:left}th{background:#e8f1f5}img{max-width:100%;height:auto;border:1px solid #dae2e7}
pre{overflow:auto;background:#f0f4f7;padding:18px}a{color:#075f93}code{font-size:.9em}
@media print{body{max-width:none;font-size:10pt}h2{break-before:page}img{max-height:230mm;object-fit:contain}
tr,img{break-inside:avoid}a{color:inherit}}</style><body>""" + body + "</body></html>"
    output = DEST / "Methods and Results.html"
    output.write_text(page, encoding="utf-8")
    assert page.count('src="data:image/png;base64,') == figure_number
    assert "src=\"assets/" not in page
    assert len(re.findall(r"^## \d+\.", assembled, flags=re.M)) == 12
    print(json.dumps({"report": str(source_path), "html": str(output),
                      "figures": figure_number, "tables": body.count("<table>"),
                      "words": len(source.split()), "copied_figures_byte_identical": True}, indent=2))


if __name__ == "__main__":
    build()
