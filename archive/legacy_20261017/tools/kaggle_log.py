"""Fetch a completed kernel log without paginating all prediction payloads."""
import argparse
import json
import os
from pathlib import Path
import sys


def fetch_log(run_id):
    root=Path(__file__).resolve().parents[1]
    spec=json.loads((root/".kaggle-work"/run_id/"run_spec.json").read_text())
    os.environ["KAGGLE_CONFIG_DIR"]=str(root/".kaggle-config")
    from kaggle.api.kaggle_api_extended import KaggleApi
    from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest
    api=KaggleApi(); api.authenticate()
    owner,slug=spec["kernel"].split("/")
    request=ApiListKernelSessionOutputRequest()
    request.user_name=owner; request.kernel_slug=slug; request.page_size=1
    with api.build_kaggle_client() as client:
        response=client.kernels.kernels_api_client.list_kernel_session_output(request)
    log=response.log or ""
    output=root/"results"/spec["campaign_id"]/run_id
    output.mkdir(parents=True,exist_ok=True)
    (output/(slug+".log")).write_text(log,encoding="utf-8")
    # A log from an earlier saved version must not be mistaken for this run.
    if run_id not in log:
        raise RuntimeError("Requested run ID absent from returned log; outputs may belong to another version")
    return log


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--run-id",required=True)
    args=parser.parse_args()
    log=fetch_log(args.run_id)
    print(log[-18000:])


if __name__=="__main__": main()
