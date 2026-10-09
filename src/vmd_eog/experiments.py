"""Grouped classical search. Reserved confirmation sources are never selected."""
import itertools
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
from .contracts import canonical_hash
from .corpus import unique_parent
from .frontal import decompose, mode_features
from .io import atomic_json, read_jsonl, sha256_file, write_jsonl
from .metrics import paired, ocular, preservation_pass
from .posterior import fit_mwf, fit_ica
from .reference import correlations, project, signed_context


def corpus_parent(input_root):
    path=unique_parent(input_root,"corpus_summary.json").parent
    if json.loads((path/"execution_state.json").read_text())["status"]!="complete":
        raise ValueError("Corpus parent incomplete")
    rows=read_jsonl(path/"corpus_manifest.jsonl")
    if any(r["partition"]["role"]!="development" for r in rows):
        raise ValueError("Development job contains reserved confirmation examples")
    return path,rows


def selection_rows(rows):
    # Fixed, participant-balanced first recipient window, 0 dB mixtures and
    # challenging clean controls. The full corpus evaluates selected recipes.
    out=[]; keys=set()
    for row in rows:
        if row["target_kind"]!="controlled_recipient_reference": continue
        if row.get("input_snr_db") not in (None,0.): continue
        key=(row["recipient"],row["condition"])
        if key in keys: continue
        keys.add(key); out.append(row)
    return out


def candidate_summary(rows):
    frame=pd.DataFrame(rows)
    clean=frame[frame.condition=="clean"]
    dirty=frame[frame.condition!="clean"]
    source_clean=clean.groupby("recipient").rrmse.mean()
    source_dirty=dirty.groupby(["recipient","condition"]).snr_db.mean().groupby("recipient").mean()
    return {"mean_snr_db":float(source_dirty.mean()) if len(source_dirty) else None,
        "worst_source_mean_clean_change":float(source_clean.max()) if len(source_clean) else None,
        "recipients":int(frame.recipient.nunique()),"dirty_examples":len(dirty),"clean_examples":len(clean)}


def fast_metric(cleaned,target):
    error=float(np.sum((np.asarray(cleaned,dtype=float)-target)**2)); energy=float(np.sum(np.asarray(target,dtype=float)**2))
    rrmse=np.sqrt(error/max(energy,1e-30))
    return float(-20*np.log10(max(rrmse,1e-15))),float(rrmse)


def run_vmd(input_root,output,config,profile):
    parent,allrows=corpus_parent(input_root); rows=selection_rows(allrows)
    grid=config["classical_grid"]; cache=output/"decompositions"; cache.mkdir()
    diagnostics=[]; scores={}; direct_scores={}; selected_channels={}
    started=time.perf_counter()
    for example,row in enumerate(rows):
        data=dict(np.load(parent/row["array_path"],allow_pickle=False))
        refs=data["references"]; target=data["paired_reference"]; eeg=data["eeg"]
        frontal=np.flatnonzero(data["regions"]==0).tolist()
        preferred=[n for n in ("FP1","FP2","FZ") if n in [s.upper() for s in data["channel_names"]]]
        indices=[i for i in frontal if data["channel_names"][i].upper() in preferred]
        if not indices: indices=frontal[:3]
        if not indices: raise ValueError("Controlled montage has no verified frontal channels")
        selected_channels[row["example_id"]]=data["channel_names"][indices].tolist()
        eeg=eeg[indices]; target=target[indices]
        direct_association=np.max(np.abs(correlations(eeg,refs,20)),axis=1)
        for lags,penalty,threshold,strength in itertools.product(grid["lag_banks"],grid["ridge_penalties"],grid["thresholds"],grid["strengths"]):
            key=json.dumps({"method":"direct","lags":lags,"penalty":penalty,"threshold":threshold,"strength":strength},sort_keys=True)
            estimate,_=project(eeg,refs,lags,penalty)
            estimate*= (direct_association>=threshold)[:,None]*strength
            snr,rrmse=fast_metric(eeg-estimate,target)
            direct_scores.setdefault(key,[]).append({"recipient":row["recipient"],"condition":row["condition"],"snr_db":snr,"rrmse":rrmse})
        for k,alpha in itertools.product(grid["K"],grid["alpha"]):
            modes=[]; associations=[]
            for j,i in enumerate(indices):
                vectors,residual,info=decompose(eeg[j],k,alpha,config)
                feature=mode_features(vectors,refs,config["fs"])
                filename=f"{row['example_id']}-channel{i}-K{k}-alpha{alpha}.npz"
                np.savez_compressed(cache/filename,modes=vectors,residual=residual,features=feature,
                    channel_name=data["channel_names"][i],centers_hz=np.asarray(info["centers_hz"]))
                diagnostics.append({"example_id":row["example_id"],"recipient":row["recipient"],"condition":row["condition"],
                    "channel":str(data["channel_names"][i]),"K":k,"alpha":alpha,"array":"decompositions/"+filename,**info})
                modes.append(vectors); associations.append(feature[:,0])
            modes=np.asarray(modes); associations=np.asarray(associations)
            for lags,penalty in itertools.product(grid["lag_banks"],grid["ridge_penalties"]):
                projected=np.stack([project(v,refs,lags,penalty)[0] for v in modes])
                for threshold,strength,kind in itertools.product(grid["thresholds"],grid["strengths"],("whole","projected")):
                    estimate=projected if kind=="projected" else modes
                    estimate=(estimate*(associations>=threshold)[:,:,None]).sum(axis=1)*strength
                    snr,rrmse=fast_metric(eeg-estimate,target)
                    spec={"method":"vmd_"+kind,"K":k,"alpha":alpha,"lags":lags,"penalty":penalty,"threshold":threshold,"strength":strength}
                    scores.setdefault(json.dumps(spec,sort_keys=True),[]).append({"recipient":row["recipient"],"condition":row["condition"],"snr_db":snr,"rrmse":rrmse})
            print("VMD",example+1,len(rows),row["example_id"],k,alpha,flush=True)
        write_jsonl(output/"mode_diagnostics.jsonl",diagnostics)
    tables=[]
    for key,values in {**scores,**direct_scores}.items():
        summary=candidate_summary(values)
        tables.append({**json.loads(key),**summary,"candidate_hash":canonical_hash(json.loads(key)),
            "quick_clean_gate":summary["worst_source_mean_clean_change"]<=config["preservation"]["clean_change"]})
    table=pd.DataFrame(tables); table.to_csv(output/"vmd_search.csv",index=False)
    grouped=[]
    for key,values in {**scores,**direct_scores}.items():
        source=pd.DataFrame(values)
        for recipient,group in source.groupby("recipient"):
            summary=candidate_summary(group.to_dict("records"))
            grouped.append({"candidate_hash":canonical_hash(json.loads(key)),"recipient":recipient,
                "mean_snr_db":summary["mean_snr_db"],"clean_change":summary["worst_source_mean_clean_change"]})
    pd.DataFrame(grouped).to_csv(output/"vmd_grouped_candidates.csv.gz",index=False,compression="gzip")
    feasible=table[table.quick_clean_gate].sort_values("mean_snr_db",ascending=False)
    selected={}
    for method in ("direct","vmd_whole","vmd_projected"):
        candidates=feasible[feasible.method==method]
        if not len(candidates): selected[method]=None; continue
        row=candidates.iloc[0]
        spec=json.loads(next(key for key in {**scores,**direct_scores} if canonical_hash(json.loads(key))==row.candidate_hash))
        selected[method]=spec
        pd.DataFrame(({**json.loads(key),**item} for key,values in {**scores,**direct_scores}.items() if key==json.dumps(spec,sort_keys=True) for item in values)).to_csv(output/f"{method}_selection_scores.csv",index=False)
    atomic_json(output/"selected_frontal.json",selected)
    atomic_json(output/"vmd_summary.json",{"profile":profile,"settings":len(grid["K"])*len(grid["alpha"]),"mode_fits":len(diagnostics),
        "candidates":len(table),"selection_examples":len(rows),"recipient_sources":len(set(r["recipient"] for r in rows)),
        "frontal_selection_channels":selected_channels,"runtime_s":time.perf_counter()-started,
        "iteration_limit_fits":sum(d["hit_iteration_limit"] for d in diagnostics),"split_hash":json.loads((parent/"corpus_summary.json").read_text())["split_hash"],
        "caveat":"development search; final regional preservation/correlation gates not yet evaluated"})
    plot_modes(diagnostics,cache,output)


def grouped_selection(table,scores,recipient_folds,limits,method_key="method"):
    """Select recipes using sources outside each held-out recipient/donor bucket."""
    selections={}
    for fold in sorted(set(recipient_folds.values())):
        excluded={r for r,f in recipient_folds.items() if f==fold}
        train=scores[~scores.recipient.isin(excluded)]
        summaries=[]
        for key,group in train.groupby("candidate_hash"):
            if "clean_change" in group:
                snr=float(group.mean_snr_db.mean()); change=float(group.clean_change.max())
            else:
                summary=candidate_summary(group.to_dict("records"))
                snr=summary["mean_snr_db"]; change=summary["worst_source_mean_clean_change"]
            summaries.append({"candidate_hash":key,"fold_selection_snr":snr,"clean_change":change})
        joined=table.merge(pd.DataFrame(summaries),on="candidate_hash")
        feasible=joined[joined.clean_change<=limits["clean_change"]].sort_values("fold_selection_snr",ascending=False)
        recipes={}
        for name,group in feasible.groupby(method_key):
            recipes[str(name)]={key:(None if isinstance(value,float) and not np.isfinite(value) else value)
                for key,value in group.iloc[0].to_dict().items()}
        selections[str(fold)]=recipes
    return selections


def frontal_artifact(data,recipe,config,selected_ids=None):
    ids=np.arange(len(data["eeg"])) if selected_ids is None else np.asarray(selected_ids,int)
    correction=np.zeros_like(data["eeg"],dtype=float); diagnostic=[]
    if recipe is None: return correction,[{"failure":"no feasible recipe"}]
    method=recipe["method"]
    if method=="direct":
        artifact,_=project(data["eeg"][ids],data["references"],recipe["lags"],recipe["penalty"])
        association=np.max(np.abs(correlations(data["eeg"][ids],data["references"],20)),axis=1)
        correction[ids]=artifact*(association>=recipe["threshold"])[:,None]*recipe["strength"]
    else:
        for i in ids:
            try:
                modes,residual,info=decompose(data["eeg"][i],int(recipe["K"]),recipe["alpha"],config)
                features=mode_features(modes,data["references"],config["fs"])
                selection=features[:,0]>=recipe["threshold"]
                estimates=project(modes,data["references"],recipe["lags"],recipe["penalty"])[0] if method=="vmd_projected" else modes
                correction[i]=(estimates*selection[:,None]).sum(axis=0)*recipe["strength"]
                diagnostic.append({"channel":str(data["channel_names"][i]),"selected_modes":np.flatnonzero(selection).tolist(),**info})
            except Exception as error:
                diagnostic.append({"channel":str(data["channel_names"][i]),"failure":f"{type(error).__name__}: {error}","status":"passthrough"})
    return correction,diagnostic


def posterior_artifact(data,cal,recipe,config,fit_cache,calibration_key):
    correction=np.zeros_like(data["eeg"],dtype=float)
    if recipe is None: return correction,{"failure":"no feasible posterior recipe"}
    scoring,out_rows,ids=support_input(data,recipe["support"])
    calibration,cal_out,cal_ids=support_input(cal,recipe["support"])
    if len(ids)<2: return correction,{"failure":"insufficient verified posterior channels"}
    key=(calibration_key,canonical_hash(recipe))
    if key not in fit_cache:
        try:
            expert=fit_mwf(calibration,cal["trial_types"],cal["boundaries"],tuple(recipe["lags"]),int(recipe["rank"])) if recipe["method"]=="mwf" else fit_ica(calibration,cal["references"],recipe["threshold"],config["seed"])
            fit_cache[key]=(expert,None)
        except Exception as error:
            fit_cache[key]=(None,f"{type(error).__name__}: {error}")
    expert,error=fit_cache[key]
    if expert is not None:
        association=np.max(np.abs(correlations(data["eeg"][ids],data["references"],20)),axis=1)
        correction[ids]=expert.artifact(scoring)[out_rows]*recipe["strength"]*(association>=recipe["threshold"])[:,None]
    return correction,{"failure":error,"support":recipe["support"],"method":recipe["method"]}


def recipe_only(row,kind):
    keys=("method","lags","penalty","threshold","strength","K","alpha") if kind=="frontal" else ("method","support","lags","rank","threshold","strength")
    spec={key:row[key] for key in keys if key in row and not (isinstance(row[key],float) and np.isnan(row[key]))}
    if isinstance(spec.get("lags"),str):
        spec["lags"]=json.loads(spec["lags"])
    return spec


def run_regional(input_root,output,config,profile):
    parent,rows=corpus_parent(input_root)
    vmd_parent=unique_parent(input_root,"vmd_summary.json").parent
    posterior_parent=unique_parent(input_root,"posterior_summary.json").parent
    vt=pd.read_csv(vmd_parent/"vmd_search.csv"); vs=pd.read_csv(vmd_parent/"vmd_grouped_candidates.csv.gz")
    pt=pd.read_csv(posterior_parent/"posterior_search.csv"); ps=pd.read_csv(posterior_parent/"posterior_scores.csv")
    folds={r["recipient"]:r["partition"]["fold"] for r in rows if r["target_kind"]=="controlled_recipient_reference"}
    vf=grouped_selection(vt,vs,folds,config["preservation"])
    pf=grouped_selection(pt,ps,folds,config["preservation"])
    atomic_json(output/"grouped_recipe_selections.json",{"frontal":vf,"posterior":pf,
        "policy":"held-out recipient and donor fold excluded from global configuration selection"})
    results=[]; diagnostics=[]; fit_cache={}; failures=[]
    arrays=output/"predictions"; arrays.mkdir()
    method_names=["identity","direct","shared_vmd","regional_mwf","regional_ica","regional_no_context"]
    for number,row in enumerate(rows):
        # Paired legacy fixtures are separately scored later: unknown anatomy
        # never enters regional feature construction or grouped fresh results.
        if row["target_kind"]!="controlled_recipient_reference": continue
        data=dict(np.load(parent/row["array_path"],allow_pickle=False)); target=data["paired_reference"]
        cal=dict(np.load(parent/row["calibration"],allow_pickle=False))
        fold=str(row["partition"]["fold"]); front=vf[fold]; post=pf[fold]
        direct=recipe_only(front["direct"],"frontal") if "direct" in front else None
        vmd=recipe_only(front["vmd_projected"],"frontal") if "vmd_projected" in front else None
        mwf=recipe_only(post["mwf"],"posterior") if "mwf" in post else None
        ica=recipe_only(post["ica"],"posterior") if "ica" in post else None
        ids_front=np.flatnonzero(data["regions"]==0); ids_post=np.flatnonzero(data["regions"]==1)
        ids_shared=np.flatnonzero(~np.isin(data["regions"],[0,1]))
        begin=time.perf_counter(); raw=data["eeg"]
        direct_art,direct_diag=frontal_artifact(data,direct,config)
        front_art,front_diag=frontal_artifact(data,vmd,config,ids_front)
        shared_art,shared_diag=frontal_artifact(data,vmd,config)
        mwf_art,mwf_diag=posterior_artifact(data,cal,mwf,config,fit_cache,row["calibration"])
        ica_art,ica_diag=posterior_artifact(data,cal,ica,config,fit_cache,row["calibration"])
        posterior_only=None if ica is None else {**ica,"support":"posterior"}
        no_art,no_diag=posterior_artifact(data,cal,posterior_only,config,fit_cache,row["calibration"])
        common=np.zeros_like(raw,dtype=float); common[ids_shared]=direct_art[ids_shared]
        corrections={"identity":np.zeros_like(raw),"direct":direct_art,"shared_vmd":shared_art,
            "regional_mwf":front_art+mwf_art+common,"regional_ica":front_art+ica_art+common,"regional_no_context":front_art+no_art+common}
        diags={"identity":[],"direct":direct_diag,"shared_vmd":shared_diag,
            "regional_mwf":front_diag+[mwf_diag],"regional_ica":front_diag+[ica_diag],"regional_no_context":front_diag+[no_diag]}
        for method in method_names:
            cleaned=raw-corrections[method]
            metrics=paired(cleaned,target,config["fs"])
            before=ocular(raw,data["references"]); after=ocular(cleaned,data["references"])
            results.append({"example_id":row["example_id"],"recipient":row["recipient"],"donor":row["donor"],"condition":row["condition"],
                "input_snr_db":row.get("input_snr_db"),"method":method,"fold":int(fold),**metrics,
                "heog_before":before["heog_abs"],"heog_after":after["heog_abs"],"veog_before":before["veog_abs"],"veog_after":after["veog_abs"],
                "region":"whole_montage","failure_count":sum(bool(d.get("failure")) for d in diags[method])})
            for region,label in ((0,"frontal"),(1,"posterior")):
                ids=np.flatnonzero(data["regions"]==region)
                if len(ids): results.append({"example_id":row["example_id"],"recipient":row["recipient"],"donor":row["donor"],"condition":row["condition"],
                    "input_snr_db":row.get("input_snr_db"),"method":method,"fold":int(fold),"region":label,**paired(cleaned[ids],target[ids],config["fs"])})
            diagnostics.append({"example_id":row["example_id"],"method":method,"diagnostics":diags[method]})
            if row["condition"]!="clean" and row.get("input_snr_db")==0.:
                np.savez_compressed(arrays/f"{row['example_id']}-{method}.npz",cleaned=cleaned,**data)
        print("REGIONAL",number+1,len(rows),row["example_id"],time.perf_counter()-begin,flush=True)
        if number%20==0:
            pd.DataFrame(results).to_csv(output/"regional_scores.csv",index=False)
            write_jsonl(output/"regional_diagnostics.jsonl",diagnostics)
    pd.DataFrame(results).to_csv(output/"regional_scores.csv",index=False)
    write_jsonl(output/"regional_diagnostics.jsonl",diagnostics)
    atomic_json(output/"regional_summary.json",{"profile":profile,"results":len(results),"methods":method_names,"participants":len(folds),
        "source_isolation":"grouped recipient/donor fold exclusion","all_corrections_relative_to":"original input","composed_subtractions":1,
        "confirmation_opened":False})


def confidence_interval(values,seed=42):
    values=np.asarray(values,dtype=float)
    rng=np.random.default_rng(seed)
    if not len(values): return [None,None]
    bootstrap=rng.choice(values,(2000,len(values)),replace=True).mean(axis=1)
    return np.quantile(bootstrap,[.025,.975]).tolist()


def run_review(input_root,output,config,profile):
    region=unique_parent(input_root,"regional_summary.json").parent
    vmd=unique_parent(input_root,"vmd_summary.json").parent
    posterior=unique_parent(input_root,"posterior_summary.json").parent
    corpus=unique_parent(input_root,"corpus_summary.json").parent
    contract=unique_parent(input_root,"contracts_summary.json").parent
    results=pd.read_csv(region/"regional_scores.csv")
    scores=results[results.region=="whole_montage"]
    comparator=scores[scores.method=="direct"].set_index("example_id")
    reports=[]
    for method,group in scores.groupby("method"):
        clean=group[group.condition=="clean"]
        dirty=group[group.condition!="clean"]
        per_source=dirty.groupby(["recipient","condition","input_snr_db"]).snr_db.mean().groupby("recipient").mean()
        clean_source=clean.groupby("recipient")[["relative_error","alpha_db","beta_db","covariance"]].mean()
        deterioration=[]
        for _,row in dirty.iterrows():
            deterioration.append({"recipient":row.recipient,"deterioration":comparator.loc[row.example_id,"pearson"]-row.pearson})
        per_source_cc=pd.DataFrame(deterioration).groupby("recipient").deterioration.mean()
        gates={"snr15":bool(len(per_source) and per_source.mean()>=15.),
            "clean_change":bool(len(clean_source) and clean_source.relative_error.max()<=config["preservation"]["clean_change"]),
            "alpha":bool(len(clean_source) and clean_source.alpha_db.max()<=config["preservation"]["alpha_db"]),
            "beta":bool(len(clean_source) and clean_source.beta_db.max()<=config["preservation"]["beta_db"]),
            "covariance":bool(len(clean_source) and clean_source.covariance.max()<=config["preservation"]["covariance"]),
            "pearson_comparator":bool(len(per_source_cc) and per_source_cc.max()<=config["preservation"]["pearson_deterioration"]),
            "at_least_five_sources":len(per_source)>=5,
            "both_blink_lateral":{"blink","lateral"}.issubset(set(dirty.condition)),
            "full_profile":profile=="full"}
        reports.append({"method":method,"mean_snr_db":float(per_source.mean()),"snr_ci95":confidence_interval(per_source,config["seed"]),
            "recipient_sources":len(per_source),"worst_source_mean_clean_change":float(clean_source.relative_error.max()),
            "worst_source_mean_alpha_db":float(clean_source.alpha_db.max()),"worst_source_mean_beta_db":float(clean_source.beta_db.max()),
            "worst_source_mean_covariance":float(clean_source.covariance.max()),"worst_source_mean_pearson_deterioration":float(per_source_cc.max()),
            "failure_count":int(group.failure_count.fillna(0).sum()),"gates":gates,"passed":all(gates.values()),
            "conditions":dirty.groupby("condition").snr_db.mean().to_dict()})
    direct=next(r for r in reports if r["method"]=="direct")
    valid=[r for r in reports if r["method"] in ("regional_mwf","regional_ica") and r["passed"] and r["mean_snr_db"]>=direct["mean_snr_db"]+.1]
    selected=max(valid,key=lambda r:r["mean_snr_db"]) if valid else None
    contracts_ok=json.loads((contract/"contracts_summary.json").read_text())["passed"]
    source=json.loads((corpus/"corpus_summary.json").read_text())
    passed=selected is not None and contracts_ok and not source["reserved_confirmation_opened"]
    recipe={"method":None if selected is None else selected["method"],
        "frontal":json.loads((vmd/"selected_frontal.json").read_text())["vmd_projected"],
        "posterior":json.loads((posterior/"selected_posterior.json").read_text()),
        "grouped_selections":json.loads((region/"grouped_recipe_selections.json").read_text()),
        "input":"EEG+HEOG+VEOG classical teacher; exported student EEG-only","offline":True}
    atomic_json(output/"approved_recipe.json",recipe)
    evidence={name:sha256_file(path) for name,path in {"regional_scores":region/"regional_scores.csv","vmd_grid":vmd/"vmd_search.csv",
        "posterior_grid":posterior/"posterior_search.csv","corpus":corpus/"corpus_summary.json","contracts":contract/"contracts_summary.json"}.items()}
    gate={"campaign_id":config["campaign_id"],"passed":passed,"model_authorization":config["model_authorization"],
        "selected_recipe_hash":canonical_hash(recipe),"evidence_hashes":evidence,"selected_method":None if selected is None else selected["method"],
        "requirement":"All preservation/paired-correlation gates, >=15dB controlled development SNR and >0.1dB advantage over matched direct regression",
        "failure_reasons":[] if passed else ["No regional VMD hybrid passed every gate and the matched direct-regression benefit requirement"],
        "reserved_confirmation_opened":False,"native_clean_recovery_claim":False}
    atomic_json(output/"classical_gate.json",gate)
    atomic_json(output/"review_summary.json",{"classical_gate_passed":passed,"methods":reports,"profile":profile,
        "status":"models conditionally authorized" if passed else "classical gate failed; models remain closed"})
    pd.DataFrame([{k:v for k,v in report.items() if not isinstance(v,(dict,list))} for report in reports]).to_csv(output/"review_methods.csv",index=False)
    lines=["# Classical approach review",f"\nGate passed: **{passed}**. These are grouped development results; confirmation remains closed.",
        "\n| Method | Mean controlled SNR dB | Worst source clean change | Gates |", "|---|---:|---:|---|"]
    for report in reports:
        lines.append(f"| {report['method']} | {report['mean_snr_db']:.3f} | {100*report['worst_source_mean_clean_change']:.3f}% | {report['passed']} |")
    lines += ["\n## Original scientific concerns",
        "K=5 was a hypothesis, not a justified fixed answer. The search covers K=3..10 and alpha=250..4000, with mode vectors, residuals, center distances, bandwidths and spectral overlap archived.",
        "VMD solves an adaptive variational band-limited decomposition. Each u_k is a length-1024 vector; modes are not predefined delta/theta/alpha/beta bins. Physiological frequency bins are separately defined for preservation metrics.",
        "Center duplication is diagnostic, not automatic rejection. K selection must improve grouped recovery while preserving EEG and remaining computationally practical.",
        "Frontal channels receive independent VMD correction. Posterior ICA/MWF sees raw frontal or signed common/lateral context; frontal/posterior estimates share the original input and are subtracted once.",
        "VMD work is O(iterations*K*T) after FFT setup, with two spectral buffers. Spatial covariance/GEVD scales cubically in embedded channel dimension; lag embedding increases that dimension. Actual runtimes and fit failures are archived.",
        "No transformer is required. Neural code remains unimplemented until this gate passes; the approved student will use shared per-channel temporal features and signed pooled context for variable caps.",
        "Controlled SNR is relative to retained LEMON recipient EEG with unknown native cleanliness. Reference-projected mixture recipes can favor regression; native OSF receives ocular suppression proxies, not reconstruction SNR.",
        "If the gate fails, do not tune reserved data, relax preservation limits or train a student from an unvalidated teacher."]
    (output/"APPROACH_REVIEW.md").write_text("\n\n".join(lines))


def plot_modes(diagnostics,cache,output):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    first=diagnostics[0]; same=[d for d in diagnostics if d["example_id"]==first["example_id"] and d["channel"]==first["channel"]]
    fig,axes=plt.subplots(1,5,figsize=(18,4),sharey=True)
    for axis,alpha in zip(axes,sorted(set(d["alpha"] for d in same))):
        for row in same:
            if row["alpha"]==alpha: axis.scatter([row["K"]]*row["K"],row["centers_hz"],s=16)
        axis.set(title=f"alpha={alpha}",xlabel="K",ylim=(0,45))
    axes[0].set_ylabel("VMD center frequency (Hz)")
    fig.suptitle(f"{first['channel']}: centers are learned, not fixed EEG bins")
    fig.tight_layout(); fig.savefig(output/"K_sweep_centers.png",dpi=160); plt.close(fig)
    item=next((d for d in same if d["K"]==5 and d["alpha"]==1000),first)
    data=np.load(cache/Path(item["array"]).name,allow_pickle=False)
    fig,axes=plt.subplots(item["K"]+1,1,figsize=(12,10),sharex=True)
    t=np.arange(data["modes"].shape[-1])/200
    for i,mode in enumerate(data["modes"]): axes[i].plot(t,mode); axes[i].set_ylabel(f"u{i+1}")
    axes[-1].plot(t,data["residual"]); axes[-1].set_ylabel("residual"); axes[-1].set_xlabel("seconds")
    fig.tight_layout(); fig.savefig(output/"mode_vectors_example.png",dpi=160); plt.close(fig)


def support_input(data,kind):
    posterior=np.flatnonzero(data["regions"]==1)
    frontal=np.flatnonzero(data["regions"]==0)
    if kind=="posterior":
        ids=posterior
    elif kind=="raw_frontal":
        ids=np.concatenate([posterior,frontal])
    elif kind=="full":
        ids=np.arange(len(data["eeg"]))
    elif kind=="signed_context":
        context=signed_context(data["eeg"],data["channel_names"],data["regions"],data["hemispheres"])
        extra=np.stack([context[key] for key in ("common","lateral","midline")])
        # Remove unavailable or rank-zero virtual channels before calibration.
        available=np.std(extra,axis=-1)>1e-12
        return np.concatenate([data["eeg"][posterior],extra[available]]),np.arange(len(posterior)),posterior
    else: raise ValueError("Unknown support kind")
    output_rows=np.asarray([np.flatnonzero(ids==i)[0] for i in posterior],int)
    return data["eeg"][ids],output_rows,posterior


def run_posterior(input_root,output,config,profile):
    parent,rows=corpus_parent(input_root); rows=selection_rows(rows)
    grids=config["classical_grid"]; fits={}; fit_rows=[]; results=[]
    # Each method's window gate uses available EOG association, never target
    # knowledge or the known clean-condition label.
    for number,row in enumerate(rows):
        data=dict(np.load(parent/row["array_path"],allow_pickle=False))
        calibration_key=row["calibration"]
        cal=dict(np.load(parent/calibration_key,allow_pickle=False))
        for method,support in [("mwf","posterior"),("mwf","signed_context"),("ica","posterior"),("ica","raw_frontal"),("ica","full")]:
            scoring,out_rows,ids=support_input(data,support)
            calibration,cal_out,cal_ids=support_input(cal,support)
            if not np.array_equal(ids,cal_ids) or len(ids)<2: continue
            bank=grids["mwf_lags"] if method=="mwf" else [[0]]
            ranks=grids["mwf_ranks"] if method=="mwf" else [None]
            thresholds=grids["thresholds"]
            for lags,rank,threshold in itertools.product(bank,ranks,thresholds):
                fit_key=(calibration_key,method,support,tuple(lags),rank,threshold if method=="ica" else None)
                if fit_key not in fits:
                    start=time.perf_counter()
                    try:
                        expert=fit_mwf(calibration,cal["trial_types"],cal["boundaries"],tuple(lags),rank) if method=="mwf" else fit_ica(calibration,cal["references"],threshold,config["seed"])
                        fits[fit_key]=(expert,None)
                    except Exception as error:
                        fits[fit_key]=(None,f"{type(error).__name__}: {error}")
                    fit_rows.append({"calibration":calibration_key,"method":method,"support":support,"lags":lags,"rank":rank,
                        "threshold":threshold if method=="ica" else None,"fit_runtime_s":time.perf_counter()-start,
                        "failure":fits[fit_key][1],"status":"passthrough" if fits[fit_key][1] else "fitted"})
                expert,error=fits[fit_key]
                correction=np.zeros_like(data["eeg"][ids]) if expert is None else expert.artifact(scoring)[out_rows]
                assoc=np.max(np.abs(correlations(data["eeg"][ids],data["references"],20)),axis=1)
                correction*= (assoc>=threshold)[:,None]
                for strength in grids["strengths"]:
                    cleaned=data["eeg"][ids]-strength*correction
                    target=data["paired_reference"][ids]
                    snr,rrmse=fast_metric(cleaned,target)
                    spec={"method":method,"support":support,"lags":lags,"rank":rank,"threshold":threshold,"strength":strength}
                    metrics=paired(cleaned,target,config["fs"])
                    results.append({"example_id":row["example_id"],"recipient":row["recipient"],"condition":row["condition"],"candidate_hash":canonical_hash(spec),
                        **spec,"snr_db":snr,"rrmse":rrmse,"fit_failure":error,"clean_alpha_db":metrics["alpha_db"],
                        "clean_beta_db":metrics["beta_db"],"covariance_error":metrics["covariance"]})
        print("POSTERIOR",number+1,len(rows),row["example_id"],flush=True)
        write_jsonl(output/"posterior_fit_diagnostics.jsonl",fit_rows)
    frame=pd.DataFrame(results); frame.to_csv(output/"posterior_scores.csv",index=False)
    candidates=[]
    for key,group in frame.groupby("candidate_hash"):
        spec={name:group.iloc[0][name] for name in ("method","support","lags","rank","threshold","strength")}
        # Convert numpy scalars before strict JSON serialization.
        spec={k:(None if k=="rank" and pd.isna(v) else v.item() if isinstance(v,np.generic) else v) for k,v in spec.items()}
        summary=candidate_summary(group.to_dict("records"))
        clean=group[group.condition=="clean"]
        gates={"change":summary["worst_source_mean_clean_change"]<=config["preservation"]["clean_change"],
            "alpha":bool((clean.groupby("recipient").clean_alpha_db.mean()<=config["preservation"]["alpha_db"]).all()),
            "beta":bool((clean.groupby("recipient").clean_beta_db.mean()<=config["preservation"]["beta_db"]).all()),
            "covariance":bool((clean.groupby("recipient").covariance_error.mean()<=config["preservation"]["covariance"]).all())}
        candidates.append({**spec,**summary,"candidate_hash":key,"quick_preservation_gate":all(gates.values()),"fit_failures":int(group.fit_failure.notna().sum())})
    table=pd.DataFrame(candidates); table.to_csv(output/"posterior_search.csv",index=False)
    feasible=table[table.quick_preservation_gate].sort_values("mean_snr_db",ascending=False)
    selected=None
    if len(feasible):
        selected={key:feasible.iloc[0][key] for key in ("method","support","lags","rank","threshold","strength")}
        selected={k:(None if k=="rank" and pd.isna(v) else v.item() if isinstance(v,np.generic) else v) for k,v in selected.items()}
    atomic_json(output/"selected_posterior.json",selected)
    atomic_json(output/"posterior_summary.json",{"profile":profile,"candidates":len(table),"fits":len(fits),
        "fit_failures":sum(f[1] is not None for f in fits.values()),"selected":selected,
        "SGEYESUB":{"qualified":False,"reason":"Original MATLAB calibration implementation not reproduced; no unverified Python substitute scored"}})


def run_stage(stage,input_root,output,config,profile):
    if stage=="vmd": run_vmd(input_root,output,config,profile)
    elif stage=="posterior": run_posterior(input_root,output,config,profile)
    elif stage=="regional": run_regional(input_root,output,config,profile)
    elif stage=="review": run_review(input_root,output,config,profile)
    else: raise ValueError("Unknown classical stage")
