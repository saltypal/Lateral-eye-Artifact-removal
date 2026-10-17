"""Grouped classical search. Reserved confirmation sources are never selected."""
import itertools
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
from .contracts import canonical_hash
from .corpus import unique_parent
from .frontal import decompose, mode_features, FCMSelection
from .io import atomic_json, read_jsonl, sha256_file, write_jsonl
from .metrics import paired, ocular, preservation_pass
from .posterior import fit_mwf, fit_ica
from .reference import correlations, project, signed_context
from .artifacts import verify_parent
from .paper_evaluation import save_protocol, write_paired, write_native


def corpus_parent(input_root):
    path=unique_parent(input_root,"corpus_summary.json").parent
    if json.loads((path/"execution_state.json").read_text())["status"]!="complete":
        raise ValueError("Corpus parent incomplete")
    rows=read_jsonl(path/"corpus_manifest.jsonl")
    verify_parent(path,("corpus_summary.json","corpus_manifest.jsonl","split_manifest.json"))
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
                try:
                    vectors,residual,info=decompose(eeg[j],k,alpha,config)
                    feature=mode_features(vectors,refs,config["fs"])
                except Exception as error:
                    vectors=np.zeros((k,eeg.shape[-1])); residual=eeg[j].copy()
                    feature=np.zeros((k,4))
                    info={"hit_iteration_limit":False,"centers_hz":[],"status":"passthrough",
                        "failure":f"{type(error).__name__}: {error}"}
                filename=f"{row['example_id']}-channel{i}-K{k}-alpha{alpha}.npz"
                np.savez_compressed(cache/filename,modes=vectors,residual=residual,features=feature,
                    channel_name=data["channel_names"][i],centers_hz=np.asarray(info["centers_hz"]))
                diagnostics.append({"example_id":row["example_id"],"recipient":row["recipient"],"condition":row["condition"],
                    "channel":str(data["channel_names"][i]),"K":k,"alpha":alpha,"array":"decompositions/"+filename,**info})
                # Iteration-limited fits are preserved for diagnosis but score
                # as passthrough; a stalled solver cannot silently correct EEG.
                unusable=info["hit_iteration_limit"] or bool(info.get("failure"))
                modes.append(np.zeros_like(vectors) if unusable else vectors)
                associations.append(np.zeros(k) if unusable else feature[:,0])
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
        "failed_mode_fits":sum(bool(d.get("failure")) for d in diagnostics),
        "caveat":"development search; final regional preservation/correlation gates not yet evaluated"})
    plot_modes(diagnostics,cache,output)
    run_fcm_ablation(parent,rows,table,pd.DataFrame(grouped),diagnostics,cache,output,config)


def run_fcm_ablation(parent,rows,table,grouped,diagnostics,cache,output,config):
    """FCM centers are fit outside each target's recipient/donor source bucket."""
    folds={r["recipient"]:r["partition"]["fold"] for r in rows}
    recipes=grouped_selection(table,grouped,folds,config["preservation"])
    index={}
    for info in diagnostics:
        index.setdefault((info["example_id"],info["K"],info["alpha"]),[]).append(info)
    models={}; records=[]
    for fold,choices in recipes.items():
        if "vmd_projected" not in choices: continue
        recipe=recipe_only(choices["vmd_projected"],"frontal")
        k=int(recipe["K"]); alpha=recipe["alpha"]
        training=[r for r in rows if str(folds[r["recipient"]])!=fold]
        features=[]
        for row in training:
            for info in index.get((row["example_id"],k,alpha),[]):
                if info["hit_iteration_limit"] or info.get("failure"): continue
                features.append(np.load(cache/Path(info["array"]).name,allow_pickle=False)["features"])
        if not features: continue
        for clusters in config["classical_grid"]["fcm_clusters"]:
            model=FCMSelection.fit(np.concatenate(features),clusters,config["seed"])
            models[f"fold{fold}-clusters{clusters}"]={"recipe":recipe,"centers":model.centers.tolist(),"mean":model.mean.tolist(),
                "scale":model.scale.tolist(),"ocular_clusters":model.ocular_clusters.tolist(),
                "fitted_recipients":sorted({r["recipient"] for r in training}),
                "fitted_donors":sorted({r["donor"] for r in training}),"heldout_fold":int(fold)}
            for row in rows:
                if str(folds[row["recipient"]])!=fold: continue
                data=dict(np.load(parent/row["array_path"],allow_pickle=False)); cleaned=[]; targets=[]
                for info in index.get((row["example_id"],k,alpha),[]):
                    item=np.load(cache/Path(info["array"]).name,allow_pickle=False)
                    modes=item["modes"]; channel=data["channel_names"].tolist().index(info["channel"])
                    estimate,_=project(modes,data["references"],recipe["lags"],recipe["penalty"])
                    selected=(item["features"][:,0]>=recipe["threshold"]) & (model.predict(item["features"])>=.5)
                    artifact=np.zeros(modes.shape[-1]) if info["hit_iteration_limit"] or info.get("failure") else (estimate*selected[:,None]).sum(axis=0)*recipe["strength"]
                    cleaned.append(data["eeg"][channel]-artifact); targets.append(data["paired_reference"][channel])
                if cleaned:
                    snr,rrmse=fast_metric(np.asarray(cleaned),np.asarray(targets))
                    records.append({"example_id":row["example_id"],"recipient":row["recipient"],"donor":row["donor"],"condition":row["condition"],
                        "fold":int(fold),"clusters":clusters,"snr_db":snr,"rrmse":rrmse,"method":"vmd_fcm_projected"})
    atomic_json(output/"fcm_train_only_models.json",models)
    pd.DataFrame(records).to_csv(output/"vmd_fcm_scores.csv",index=False)


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
            additional=True
            if "clean_alpha_db" in group:
                clean=group[group.condition=="clean"].groupby("recipient")[["clean_alpha_db","clean_beta_db","covariance_error"]].mean()
                additional=bool(len(clean) and clean.clean_alpha_db.max()<=limits["alpha_db"] and clean.clean_beta_db.max()<=limits["beta_db"] and clean.covariance_error.max()<=limits["covariance"])
            summaries.append({"candidate_hash":key,"fold_selection_snr":snr,"clean_change":change,"training_preservation":additional})
        joined=table.merge(pd.DataFrame(summaries),on="candidate_hash")
        feasible=joined[(joined.clean_change<=limits["clean_change"]) & joined.training_preservation].sort_values("fold_selection_snr",ascending=False)
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
                if info["hit_iteration_limit"]:
                    diagnostic.append({"channel":str(data["channel_names"][i]),"failure":"VMD iteration cap","status":"passthrough",**info})
                    continue
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
    save_protocol(output)
    method_names=["identity","direct","shared_vmd","regional_mwf","regional_ica","regional_no_context",
        "regional_mwf_direct_frontal","regional_ica_direct_frontal"]
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
        shared_art,shared_diag=frontal_artifact(data,vmd,config)
        # The shared ablation already decomposes these exact frontal inputs.
        # Reuse its estimate without changing any regional information access.
        front_art=np.zeros_like(raw,dtype=float); front_art[ids_front]=shared_art[ids_front]
        front_names=set(data["channel_names"][ids_front].tolist())
        front_diag=[d for d in shared_diag if d.get("channel") in front_names or "channel" not in d]
        mwf_art,mwf_diag=posterior_artifact(data,cal,mwf,config,fit_cache,row["calibration"])
        ica_art,ica_diag=posterior_artifact(data,cal,ica,config,fit_cache,row["calibration"])
        posterior_only=None if ica is None else {**ica,"support":"posterior"}
        no_art,no_diag=posterior_artifact(data,cal,posterior_only,config,fit_cache,row["calibration"])
        common=np.zeros_like(raw,dtype=float); common[ids_shared]=direct_art[ids_shared]
        corrections={"identity":np.zeros_like(raw),"direct":direct_art,"shared_vmd":shared_art,
            "regional_mwf":front_art+mwf_art+common,"regional_ica":front_art+ica_art+common,"regional_no_context":front_art+no_art+common}
        direct_front=np.zeros_like(raw,dtype=float); direct_front[ids_front]=direct_art[ids_front]
        corrections.update(regional_mwf_direct_frontal=direct_front+mwf_art+common,
            regional_ica_direct_frontal=direct_front+ica_art+common)
        diags={"identity":[],"direct":direct_diag,"shared_vmd":shared_diag,
            "regional_mwf":front_diag+[mwf_diag],"regional_ica":front_diag+[ica_diag],"regional_no_context":front_diag+[no_diag]}
        diags.update(regional_mwf_direct_frontal=direct_diag+[mwf_diag],regional_ica_direct_frontal=direct_diag+[ica_diag])
        for method in method_names:
            cleaned=raw-corrections[method]
            metrics=paired(cleaned,target,config["fs"])
            write_paired(output,{"example_id":row["example_id"],"recipient":row["recipient"],
                "donor":row["donor"],"condition":row["condition"],"method":method,
                "input_snr_db":row.get("input_snr_db"),"dataset":row["dataset"]},
                cleaned,target,raw,data["channel_names"],data["regions"])
            before=ocular(raw,data["references"]); after=ocular(cleaned,data["references"])
            results.append({"example_id":row["example_id"],"recipient":row["recipient"],"donor":row["donor"],"condition":row["condition"],
                "input_snr_db":row.get("input_snr_db"),"method":method,"fold":int(fold),**metrics,
                "heog_before":before["heog_abs"],"heog_after":after["heog_abs"],"veog_before":before["veog_abs"],"veog_after":after["veog_abs"],
                "region":"whole_montage","eog_joint_r2_before":before["joint_eog_r2"],"eog_joint_r2_after":after["joint_eog_r2"],
                "failure_count":sum(bool(d.get("failure")) for d in diags[method])})
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
    run_native_comparison(parent,rows,vf,pf,output,config)
    from .paper_report import write_reports
    write_reports(output)
    atomic_json(output/"regional_summary.json",{"profile":profile,"results":len(results),"methods":method_names,"participants":len(folds),
        "source_isolation":"grouped recipient/donor fold exclusion","all_corrections_relative_to":"original input","composed_subtractions":1,
        "confirmation_opened":False})


def run_native_comparison(parent,rows,frontal_recipes,posterior_recipes,output,config):
    """OSF has association proxies; unknown-montage Klados has legacy pairs."""
    results=[]; diagnostics=[]; fits={}
    for number,row in enumerate(rows):
        kind=row["target_kind"]
        if kind=="controlled_recipient_reference": continue
        fold=str(row["partition"]["fold"])
        front=frontal_recipes.get(fold,{})
        post=posterior_recipes.get(fold,{})
        direct=recipe_only(front["direct"],"frontal") if "direct" in front else None
        vmd=recipe_only(front["vmd_projected"],"frontal") if "vmd_projected" in front else None
        data=dict(np.load(parent/row["array_path"],allow_pickle=False))
        raw=data["eeg"]
        direct_art,direct_diag=frontal_artifact(data,direct,config)
        shared_art,shared_diag=frontal_artifact(data,vmd,config)
        corrections={"identity":np.zeros_like(raw),"direct":direct_art,"shared_vmd":shared_art}
        diags={"identity":[],"direct":direct_diag,"shared_vmd":shared_diag}
        if kind=="real_proxy":
            cal=dict(np.load(parent/row["calibration"],allow_pickle=False))
            frontal_ids=np.flatnonzero(data["regions"]==0)
            common_ids=np.flatnonzero(~np.isin(data["regions"],[0,1]))
            common=np.zeros_like(raw,dtype=float)
            common[frontal_ids]=shared_art[frontal_ids]
            common[common_ids]=direct_art[common_ids]
            for method in ("mwf","ica"):
                recipe=recipe_only(post[method],"posterior") if method in post else None
                artifact,diag=posterior_artifact(data,cal,recipe,config,fits,row["calibration"])
                corrections["regional_"+method]=common+artifact
                diags["regional_"+method]=shared_diag+[diag]
        before=ocular(raw,data["references"])
        for method,artifact in corrections.items():
            cleaned=raw-artifact
            after=ocular(cleaned,data["references"])
            record={"example_id":row["example_id"],"source":row["recipient"],"condition":row["condition"],
                "dataset":row["dataset"],"method":method,"target_kind":kind,
                "heog_before":before["heog_abs"],"heog_after":after["heog_abs"],
                "veog_before":before["veog_abs"],"veog_after":after["veog_abs"],
                "joint_r2_before":before["joint_eog_r2"],"joint_r2_after":after["joint_eog_r2"],
                "failure_count":sum(bool(d.get("failure")) for d in diags[method])}
            if kind=="legacy_paired":
                write_paired(output,{"example_id":row["example_id"],"recipient":row["recipient"],
                    "donor":row.get("donor"),"condition":row["condition"],"method":method,
                    "input_snr_db":row.get("input_snr_db"),"dataset":row["dataset"]},
                    cleaned,data["paired_reference"],raw,data["channel_names"],data["regions"])
                record.update(paired(cleaned,data["paired_reference"],config["fs"]))
                record["scope"]="legacy development-exposed; unknown participants, units and montage"
            else:
                write_native(output,{"example_id":row["example_id"],"source":row["recipient"],
                    "condition":row["condition"],"method":method,"dataset":row["dataset"]},
                    cleaned,raw,data["references"],data["channel_names"],data["regions"],config["fs"])
                change=paired(cleaned,raw,config["fs"])
                record.update({"modification_relative_rms":change["relative_error"],
                    "alpha_modification_db":change["alpha_db"],"beta_modification_db":change["beta_db"],
                    "covariance_modification":change["covariance"]})
                record["scope"]="native ocular association and modification proxies; no reconstruction SNR"
            results.append(record)
            diagnostics.append({"example_id":row["example_id"],"method":method,"diagnostics":diags[method]})
        print("NATIVE_COMPARISON",number+1,len(rows),row["example_id"],flush=True)
        if number%20==0: pd.DataFrame(results).to_csv(output/"native_legacy_scores.csv",index=False)
    pd.DataFrame(results).to_csv(output/"native_legacy_scores.csv",index=False)
    write_jsonl(output/"native_legacy_diagnostics.jsonl",diagnostics)
    atomic_json(output/"native_legacy_summary.json",{"examples":sum(r["target_kind"]!="controlled_recipient_reference" for r in rows),
        "result_rows":len(results),"regional_klados_anatomy_inferred":False,
        "native_snr_claim":False,"used_for_confirmation_selection":False})


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
    verify_parent(region,("regional_scores.csv","grouped_recipe_selections.json","native_legacy_scores.csv"))
    verify_parent(vmd,("vmd_search.csv","selected_frontal.json","mode_diagnostics.jsonl"))
    verify_parent(posterior,("posterior_search.csv","selected_posterior.json"))
    verify_parent(contract,("contracts_summary.json",))
    native_files=list(Path(input_root).rglob("native_protocol_summary.json"))
    if len(native_files)>1:
        raise ValueError("Attach one immutable full native protocol run")
    native=None if not native_files else native_files[0].parent
    if native is not None:
        verify_parent(native)
    from .paper_qualification import qualify_paper_evidence
    qualification=qualify_paper_evidence(region,corpus,native)
    atomic_json(output/"paper_evidence_qualification.json",qualification)
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
    # The requested output-SNR target now uses channel-first paper tables.
    # Retain archived pooled-energy numbers as supplementary diagnostics.
    for report in reports:
        primary=qualification["paired_synopsis"].get(report["method"],{})
        channel_snr=primary.get("snr_energy_db")
        report["paper_channel_mean_snr_db"]=channel_snr
        report["gates"]["paper_channel_snr15"]=channel_snr is not None and channel_snr>=15.
        report["passed"]=all(report["gates"].values())
    direct=next(r for r in reports if r["method"]=="direct")
    for report in reports:
        if report["method"] not in ("regional_mwf","regional_ica"): continue
        matched=next(r for r in reports if r["method"]==report["method"]+"_direct_frontal")
        report["matched_frontal_regression_gain_db"]=report["mean_snr_db"]-matched["mean_snr_db"]
        report["gates"]["vmd_matched_benefit"]=report["matched_frontal_regression_gain_db"]>=.1
        primary=report["paper_channel_mean_snr_db"]
        matched_primary=matched["paper_channel_mean_snr_db"]
        report["paper_matched_frontal_regression_gain_db"]=None if primary is None or matched_primary is None else primary-matched_primary
        report["gates"]["paper_vmd_matched_benefit"]=report["paper_matched_frontal_regression_gain_db"] is not None and report["paper_matched_frontal_regression_gain_db"]>=.1
        report["passed"]=all(report["gates"].values())
    valid=[r for r in reports if r["method"] in ("regional_mwf","regional_ica") and r["passed"]
        and direct["paper_channel_mean_snr_db"] is not None
        and r["paper_channel_mean_snr_db"]>=direct["paper_channel_mean_snr_db"]+.1]
    selected=max(valid,key=lambda r:r["paper_channel_mean_snr_db"]) if valid else None
    contracts_ok=json.loads((contract/"contracts_summary.json").read_text())["passed"]
    source=json.loads((corpus/"corpus_summary.json").read_text())
    passed=selected is not None and contracts_ok and not source["reserved_confirmation_opened"]
    project_gates_passed=passed
    # User requested the prescribed papers' evaluation on 2026-10-10. A
    # project-only review must not automatically release neural development.
    paper_evaluation_complete=qualification["complete"]
    passed=passed and paper_evaluation_complete
    recipe={"method":None if selected is None else selected["method"],
        "frontal":json.loads((vmd/"selected_frontal.json").read_text())["vmd_projected"],
        "posterior":json.loads((posterior/"selected_posterior.json").read_text()),
        "grouped_selections":json.loads((region/"grouped_recipe_selections.json").read_text()),
        "input":"EEG+HEOG+VEOG classical teacher; exported student EEG-only","offline":True}
    if selected is not None:
        method="mwf" if selected["method"]=="regional_mwf" else "ica"
        posterior_table=pd.read_csv(posterior/"posterior_search.csv")
        choices=posterior_table[(posterior_table.method==method)&posterior_table.quick_preservation_gate].sort_values("mean_snr_db",ascending=False)
        if len(choices): recipe["posterior"]=recipe_only(choices.iloc[0].to_dict(),"posterior")
    atomic_json(output/"approved_recipe.json",recipe)
    import shutil
    evidence_paths={"regional_scores":region/"regional_scores.csv","vmd_grid":vmd/"vmd_search.csv",
        "posterior_grid":posterior/"posterior_search.csv","corpus":corpus/"corpus_summary.json","contracts":contract/"contracts_summary.json"}
    evidence_paths["paper_qualification"]=output/"paper_evidence_qualification.json"
    if native is not None:
        for name in ("native_protocol_summary.json","native_source_ledger.jsonl","native_condition_participants.csv",
                     "native_condition_permutations.csv","native_chance_summary.json","native_recipe_selections.json"):
            evidence_paths["native_"+Path(name).stem]=native/name
    paper_files=("paper_evaluation_protocol.json","paper_metric_report_summary.json","paper_paired_source_means.csv",
        "paper_native_source_means.csv","paper_paired_permutation_tests.csv","paper_native_permutation_tests.csv","EVALUATION_PROTOCOL.md")
    for name in paper_files:
        if (region/name).exists(): evidence_paths["paper_"+Path(name).stem]=region/name
    evidence={name:sha256_file(path) for name,path in evidence_paths.items()}
    evidence_directory=output/"evidence"; evidence_directory.mkdir()
    evidence_files={}
    for name,path in evidence_paths.items():
        saved=evidence_directory/(name+path.suffix)
        shutil.copy2(path,saved)
        evidence_files[name]=str(saved.relative_to(output))
    gate={"campaign_id":config["campaign_id"],"passed":passed,"model_authorization":config["model_authorization"],
        "project_gates_passed":project_gates_passed,"paper_evaluation_complete":paper_evaluation_complete,
        "selected_recipe_hash":canonical_hash(recipe),"evidence_hashes":evidence,"evidence_files":evidence_files,
        "selected_method":None if selected is None else selected["method"],
        "requirement":"All preservation/paired-correlation gates, >=15dB controlled development SNR and >=0.1dB advantage over direct regression both alone and with identical posterior expert",
        "failure_reasons":(["No regional VMD hybrid passed every project gate and matched regression benefit requirement"] if not project_gates_passed else [])
            + qualification["reasons"],
        "reserved_confirmation_opened":False,"native_clean_recovery_claim":False}
    atomic_json(output/"classical_gate.json",gate)
    atomic_json(output/"review_summary.json",{"classical_gate_passed":passed,"methods":reports,"profile":profile,
        "paper_evaluation_complete":paper_evaluation_complete,
        "primary_evaluation_protocol":"paper-grounded-v1-20261010",
        "status":"models conditionally authorized" if passed else "classical gate failed; models remain closed"})
    pd.DataFrame([{k:v for k,v in report.items() if not isinstance(v,(dict,list))} for report in reports]).to_csv(output/"review_methods.csv",index=False)
    lines=["# Classical approach review",f"\nGate passed: **{passed}**. These are grouped development results; confirmation remains closed.",
        "\nPrimary evaluation follows the prescribed VMD and EEGOAR-Net sources. Exact equations, limitations and adapted aggregation are archived in the paper protocol. Old project metrics are supplementary; a project-only pass does not authorize models."]
    if (region/"paper_paired_source_means.csv").exists():
        paper=pd.read_csv(region/"paper_paired_source_means.csv")
        paper=paper[(paper.dataset=="controlled_LEMON_OSF") & (paper.condition!="clean")]
        if len(paper):
            # Source/condition/input-level means are balanced in this synopsis;
            # the original stratified tables and eligibility are also retained.
            from .paper_report import paired_synopsis
            synopsis=paired_synopsis(paper)
            synopsis.to_csv(output/"paper_primary_synopsis.csv")
            lines += ["\n## Paper-based paired metrics (adapted source-balanced synopsis)",
                "\n| Method | RRMSE | MSE (V²) | Pearson CC | Channel-first output SNR (dB) |",
                "|---|---:|---:|---:|---:|"]
            for method,values in synopsis.iterrows():
                lines.append(f"| {method} | {values.rrmse_time:.5g} | {values.mse:.5g} | {values.pearson_cc:.5g} | {values.snr_energy_db:.4g} |")
    lines += ["\n## Supplementary project gates (archived pooled-energy objective)",
        "\n| Method | Pooled-energy controlled SNR dB | Worst source clean change | Project gates |", "|---|---:|---:|---|"]
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
    plot_review(results,pd.read_csv(region/"native_legacy_scores.csv"),reports,output,config)
    from .paper_figures import plot_paper_review
    plot_paper_review(region,native,output,config["seed"])
    import shutil
    review_files={"vmd_search.csv":vmd,"posterior_search.csv":posterior,
        "vmd_fcm_scores.csv":vmd,"mode_diagnostics.jsonl":vmd,"posterior_fit_diagnostics.jsonl":posterior,
        "native_legacy_scores.csv":region,"native_legacy_summary.json":region}
    for name,parent in review_files.items():
        verify_parent(parent,(name,))
        shutil.copy2(parent/name,output/name)
    for name in ("K_sweep_centers.png","mode_vectors_example.png"):
        if (vmd/name).exists():
            verify_parent(vmd,(name,))
            shutil.copy2(vmd/name,output/name)


def plot_review(results,native,reports,output,config):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    whole=results[(results.region=="whole_montage")&(results.condition!="clean")]
    grouped=whole.groupby(["method","condition","recipient"]).snr_db.mean()
    rows=[]
    for (method,condition),values in grouped.groupby(level=[0,1]):
        interval=confidence_interval(values,config["seed"])
        rows.append({"method":method,"condition":condition,"mean_snr_db":float(values.mean()),
            "ci95_lower":interval[0],"ci95_upper":interval[1],"sources":len(values)})
    table=pd.DataFrame(rows); table.to_csv(output/"condition_source_uncertainty.csv",index=False)
    labels={"identity":"Input / identity","direct":"Direct EOG regression","shared_vmd":"Shared VMD",
        "regional_mwf":"VMD + posterior MWF","regional_ica":"VMD + posterior ICA",
        "regional_no_context":"VMD + posterior-only ICA",
        "regional_mwf_direct_frontal":"Regression + posterior MWF",
        "regional_ica_direct_frontal":"Regression + posterior ICA"}
    methods=list(labels)
    fig,axes=plt.subplots(1,3,figsize=(17,6),sharey=True)
    for axis,condition in zip(axes,("blink","lateral","mixed")):
        data=table[table.condition==condition].set_index("method").reindex(methods)
        axis.errorbar(data.mean_snr_db,np.arange(len(methods)),
            xerr=np.stack([data.mean_snr_db-data.ci95_lower,data.ci95_upper-data.mean_snr_db]),
            fmt="o",color="#315e82",ecolor="#333333",capsize=3)
        axis.axvline(15,color="#444444",linestyle="--",label="15 dB target")
        axis.set_title(condition); axis.set_xlabel("Supplementary pooled-energy SNR (dB)")
        axis.grid(axis="x",alpha=.15)
    axes[0].set_yticks(np.arange(len(methods)),[labels[m] for m in methods]); axes[0].invert_yaxis()
    fig.suptitle("Supplementary archived objective: recipient means and participant-bootstrap 95% intervals")
    fig.tight_layout(); fig.savefig(output/"condition_SNR.png",dpi=160); plt.close(fig)
    report_by_method={r["method"]:r for r in reports}
    fig,axes=plt.subplots(1,2,figsize=(14,6),sharey=True)
    for axis,key,scale,threshold,title in ((axes[0],"worst_source_mean_clean_change",100.,1.,"Worst recipient mean clean modification (%)"),
        (axes[1],"mean_snr_db",1.,15.,"Supplementary pooled-energy SNR (dB)")):
        values=[scale*report_by_method[m][key] for m in methods]
        axis.scatter(values,np.arange(len(methods)),color="#315e82",s=45)
        axis.axvline(threshold,color="#444444",linestyle="--")
        for i,value in enumerate(values): axis.annotate(f"{value:.3g}",(value,i),xytext=(5,5),textcoords="offset points")
        axis.set_xlabel(title); axis.grid(axis="x",alpha=.15)
    axes[0].set_yticks(np.arange(len(methods)),[labels[m] for m in methods]); axes[0].invert_yaxis()
    fig.suptitle("Recovery and EEG preservation must pass together; dashed lines are acceptance limits")
    fig.tight_layout(); fig.savefig(output/"recovery_preservation.png",dpi=160); plt.close(fig)
    proxy=native[native.target_kind=="real_proxy"]
    if len(proxy):
        summaries=proxy.groupby(["method","condition","source"])[["joint_r2_before","joint_r2_after","modification_relative_rms"]].mean()
        summaries.groupby(level=[0,1]).mean().to_csv(output/"OSF_source_balanced_proxies.csv")
    legacy=native[native.target_kind=="legacy_paired"]
    if len(legacy):
        legacy.groupby(["method","source"]).snr_db.mean().groupby("method").mean().to_csv(output/"Klados_legacy_record_balanced_SNR.csv")


def plot_modes(diagnostics,cache,output):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    valid=[d for d in diagnostics if not d.get("failure")]
    if not valid: return
    first=next((d for d in valid if d["condition"]=="blink"),valid[0])
    same=[d for d in valid if d["example_id"]==first["example_id"] and d["channel"]==first["channel"]]
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
                fit_key=(calibration_key,method,support,tuple(lags),rank)
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
                correction=np.zeros_like(data["eeg"][ids]) if expert is None else (expert.artifact(scoring,threshold=threshold) if method=="ica" else expert.artifact(scoring))[out_rows]
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
