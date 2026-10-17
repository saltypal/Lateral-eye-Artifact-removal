"""Read a bounded live Kaggle log snapshot without downloading waveform files.

This is non-numerical orchestration. SDK credentials and signed URLs are never
printed. A snapshot can be incomplete; absence of an error is not a passed run.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time

import requests

ROOT = Path(__file__).resolve().parents[1]
os.environ["KAGGLE_CONFIG_DIR"] = str(ROOT / ".kaggle-config")

from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.services.kernels_api_service import KernelsApiClient


def main():
    from run import ensure_auth
    ensure_auth()
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    work = ROOT / ".kaggle-work" / args.run_id
    spec = json.loads((work / "run_spec.json").read_text(encoding="utf-8"))
    original_send = requests.sessions.Session.send

    def bounded_send(session, request, **kwargs):
        kwargs["timeout"] = (10, 5)
        return original_send(session, request, **kwargs)

    requests.sessions.Session.send = bounded_send
    chunks, response, error_type = [], None, None
    try:
        api = KaggleApi()
        api.authenticate()
        request_type = KernelsApiClient.get_kernel_session_logs_stream.__globals__["ApiGetKernelSessionLogsStreamRequest"]
        with api.build_kaggle_client() as client:
            request = request_type()
            request.user_name, request.kernel_slug = spec["kernel"].split("/")
            request.wait_for_logs_url_seconds = 1
            response = client.kernels.kernels_api_client.get_kernel_session_logs_stream(request)
            response.raise_for_status()
            deadline = time.monotonic() + 8
            for chunk in response.iter_content(chunk_size=1024):
                chunks.append(chunk)
                if time.monotonic() >= deadline or len(chunks) >= 2048:
                    break
    except Exception as error:
        error_type = type(error).__name__
    finally:
        if response is not None:
            response.close()
    text = b"".join(chunks).decode("utf-8", errors="replace")
    (work / "live_log_snapshot.txt").write_text(text, encoding="utf-8")
    events = []
    for line in text.splitlines():
        if line.startswith("data: "):
            try:
                events.append(json.loads(line[6:]))
            except json.JSONDecodeError:
                pass
    if not events:
        try:
            persisted = json.loads(text)
            if isinstance(persisted, list):
                events = persisted
        except json.JSONDecodeError:
            pass
    bootstrap = [event.get("data", "") for event in events
                 if event.get("data", "").startswith("BOOTSTRAP_START ")]
    identity_verified = bool(bootstrap and all(spec["run_id"] in value and spec["git_sha"] in value
                                              for value in bootstrap))
    markers = []
    for event in events:
        data = event.get("data", "")
        if (data.startswith(("FROZEN_NATIVE_TRIAL_SAVED ", "BOOTSTRAP_PHASE ",
                             "SOURCE_VERIFIED ","AUTHOR_SOURCES_VERIFIED", "AUTHOR_MODELS_LOADED ",
                             "BANDROUTE_EPOCH_SAVED ", "EEGOAR_EPOCH_SAVED ", "METHOD_SAVED ", "METHOD_FAILED ",
                             "RUN_COMPLETE ", "Traceback (most recent call last)"))
                or (" passed" in data and data[:1].isdigit())):
            markers.append({"runtime_s": event.get("time"), "message": data.strip()[:800]})
    summary = {"run_id": spec["run_id"], "git_sha": spec["git_sha"],
               "utc": datetime.now(timezone.utc).isoformat(), "bytes": len(text),
               "bootstrap_identity_verified": identity_verified, "incomplete_snapshot": True,
               "stream_end_error_type": error_type, "last_markers": markers[-8:] if identity_verified else [],
               "not_a_completion_or_success_check": True}
    (work / "live_log_snapshot.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
