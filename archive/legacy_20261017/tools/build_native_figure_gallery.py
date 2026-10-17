"""Package verified Kaggle-rendered image bytes into an offline HTML gallery.

Local authoring only: no EEG loading, plotting, numerical analysis or image edits.
"""
import argparse
import base64
from html import escape
import json
from pathlib import Path

from prepare_broader_stage import ROOT, recovered


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    parent, summary, pin, kernel = recovered(args.run_id, "native_visual_summary.json")
    inventory = json.loads((parent / "figure_inventory.json").read_text(encoding="utf-8"))
    if summary["figures"] != 9 or len(inventory) != 9 or summary["all_cached_selections_reproduced"] is not True:
        raise ValueError("Incomplete figure or mode-audit evidence")
    destination = ROOT / "results" / pin["campaign_id"] / args.run_id / "presentation"
    destination.mkdir(exist_ok=True)
    parts = ["<!doctype html><html lang='en'><meta charset='utf-8'><title>Native EEG diagnostic figures</title>",
        "<style>body{font:17px/1.55 system-ui,sans-serif;max-width:1150px;margin:40px auto;padding:0 24px;color:#192633}img{width:100%;height:auto}section{border-top:1px solid #ccd5de;margin-top:40px;padding-top:12px}aside{padding:20px;background:#fff0df}code{overflow-wrap:anywhere}nav a{display:inline-block;margin:5px 12px 5px 0}</style>",
        "<h1>Native frontal EEG: suppression and preservation diagnostics</h1>",
        "<aside>Two development OSF participants. All active methods fail the resting preservation requirements. Lower ocular correlation alone does not establish preservation. No native reconstruction SNR or qualified teacher is claimed.</aside>",
        "<p>Figures were calculated and rendered on Kaggle. This offline gallery embeds the original verified PNG bytes; local code performed no numerical analysis.</p>",
        "<p>Source: <code>" + escape(pin["git_sha"]) + "</code><br>Run: " + escape(args.run_id)
        + "<br>Manifest: <code>" + escape(pin["manifest_sha256"]) + "</code></p><nav>"]
    for item in inventory:
        title = item["figure"]
        parts.append("<a href='#" + escape(title, quote=True) + "'>" + escape(title) + "</a>")
    parts.append("</nav>")
    for item in inventory:
        title = item["figure"]
        data = base64.b64encode((parent / "figures" / (title + ".png")).read_bytes()).decode("ascii")
        parts.append("<section id='" + escape(title, quote=True) + "'><h2>" + escape(title)
            + "</h2><p>" + escape(item["caption"]) + "</p><img alt='" + escape(title, quote=True)
            + "' src='data:image/png;base64," + data + "'></section>")
    parts.append("</html>")
    path = destination / "native_diagnostic_gallery.html"
    path.write_text("\n".join(parts), encoding="utf-8")
    print(json.dumps({"gallery": str(path), "embedded_figures": len(inventory), "kernel": kernel,
                      "verified_parent": pin, "local_numerical_execution": False}, indent=2))


if __name__ == "__main__":
    main()
