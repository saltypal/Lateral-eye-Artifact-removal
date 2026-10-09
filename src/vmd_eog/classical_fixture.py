"""Bounded Kaggle integration fixture; synthetic results are not research evidence."""
from copy import deepcopy
import json
from pathlib import Path
import numpy as np
from .contracts import canonical_hash
from .data import channel_metadata
from .experiments import run_vmd, run_posterior, run_regional, run_review
from .io import atomic_json, sha256_file, write_jsonl


def finish(directory):
    manifest={str(p.relative_to(directory)):sha256_file(p) for p in directory.rglob("*")
        if p.is_file() and p.name not in ("artifact_manifest.json","execution_state.json")}
    atomic_json(directory/"artifact_manifest.json",manifest)
    atomic_json(directory/"execution_state.json",{"status":"complete"})


def run_fixture(output,config):
    parameters=deepcopy(config)
    parameters["classical_grid"].update(K=[3],alpha=[1000],thresholds=[.8],strengths=[1.],
        lag_banks=[[0]],ridge_penalties=[.01],fcm_clusters=[2],mwf_ranks=[1],mwf_lags=[[0]])
    internal=output/"integration"; internal.mkdir()
    corpus=internal/"corpus"; corpus.mkdir()
    arrays=corpus/"arrays"; arrays.mkdir()
    calibrations=corpus/"calibration"; calibrations.mkdir()
    names=["Fp1","Fp2","Fz","P3","P4","Oz","C3","C4"]
    metadata=channel_metadata(names)
    rng=np.random.default_rng(314)
    rows=[]
    field=np.asarray([[1.,1.],[-1.,1.],[.1,1.],[.3,.2],[-.3,.2],[.1,.2],[.2,.3],[-.2,.3]])
    for participant in range(5):
        source=f"fixture:{participant}"
        length=4096; t=np.arange(length)/config["fs"]
        refs=np.stack([np.sin(2*np.pi*1.7*t),np.sin(2*np.pi*2.3*t)**9])
        labels=np.repeat([1,2,3,4],1024)
        rest=labels==1
        refs[:,rest]*=.01
        neural=rng.normal(size=(len(names),length))*.3+np.sin(2*np.pi*11*t)[None,:]
        calname=f"calibration/source-{participant}.npz"
        np.savez_compressed(corpus/calname,eeg=neural+field@refs,references=refs,
            trial_types=labels,boundaries=np.asarray([1024,2048,3072]),**metadata)
        for condition in ("clean","blink","lateral","mixed"):
            times=np.arange(1024)/config["fs"]
            reference=np.stack([np.sin(2*np.pi*1.7*times),np.sin(2*np.pi*2.3*times)**9])
            if condition=="blink": reference[0]*=.01
            if condition=="lateral": reference[1]*=.01
            target=rng.normal(size=(len(names),1024))*.3+np.sin(2*np.pi*11*times)[None,:]
            artifact=field@reference
            if condition=="clean": artifact*=0
            else: artifact*=np.linalg.norm(target)/np.linalg.norm(artifact)
            name=f"source-{participant}-{condition}"
            np.savez_compressed(arrays/(name+".npz"),eeg=target+artifact,paired_reference=target,
                references=reference,**metadata)
            rows.append({"example_id":name,"array_path":"arrays/"+name+".npz",
                "recipient":source,"donor":f"fixture-donor:{participant}","dataset":"synthetic_fixture",
                "condition":condition,"input_snr_db":None if condition=="clean" else 0.,
                "target_kind":"controlled_recipient_reference","calibration":calname,
                "partition":{"role":"development","fold":participant}})
    for kind,dataset in (("real_proxy","osf"),("legacy_paired","klados")):
        row={**rows[-1],"example_id":"fixture-"+kind,"target_kind":kind,"dataset":dataset}
        rows.append(row)
    write_jsonl(corpus/"corpus_manifest.jsonl",rows)
    atomic_json(corpus/"split_manifest.json",{"hash":canonical_hash({"synthetic":True})})
    atomic_json(corpus/"corpus_summary.json",{"synthetic_fixture":True,"reserved_confirmation_opened":False,
        "split_hash":canonical_hash({"synthetic":True})})
    finish(corpus)
    contract=internal/"contracts"; contract.mkdir()
    atomic_json(contract/"contracts_summary.json",{"passed":True,"scope":"upstream runner tests passed"})
    finish(contract)
    for stage,function in (("vmd",run_vmd),("posterior",run_posterior),("regional",run_regional),("review",run_review)):
        directory=internal/stage; directory.mkdir()
        function(internal,directory,parameters,"pilot")
        finish(directory)
        print("INTEGRATION_STAGE_PASSED",stage,flush=True)
    gate=json.loads((internal/"review"/"classical_gate.json").read_text())
    if gate["passed"]:
        raise AssertionError("A synthetic pilot must never authorize neural development")
    atomic_json(output/"classical_fixture_summary.json",{"passed":True,"stages":["vmd","posterior","regional","review"],
        "classical_gate_passed":False,"scope":"synthetic bounded software integration, not research validation",
        "reduced_fixture_grid":parameters["classical_grid"],"production_grid_unchanged":True})
