"""Paired objectives; preservation penalties are project design choices."""
import torch
from .neural import input_scale


def masked_mean(values,mask):
    return (values*mask).sum()/mask.sum().clamp_min(1)


def welch_torch(values,fs=200):
    length = int(2*fs)
    if values.shape[-1] < length:
        raise ValueError("Welch requires at least two seconds")
    segments = values.unfold(-1,length,int(fs))
    segments = segments-segments.mean(-1,keepdim=True)
    window = torch.hann_window(length,device=values.device,dtype=values.dtype)
    power = torch.fft.rfft((segments*window).float(),dim=-1).abs().square()
    power = power/(fs*window.square().sum())
    power[...,1:-1] *= 2
    return torch.fft.rfftfreq(length,1/fs,device=values.device),power.mean(-2)


def paired_objective(cleaned,target,original,mask,clean_examples,config,profile="mse",references=None):
    if profile not in ("mse","preservation","snr"):
        raise ValueError("Unknown loss profile")
    scale = input_scale(original)
    prediction,truth = cleaned/scale,target/scale
    error = prediction-truth
    per_channel = error.square().mean(-1)
    terms = {"mse":masked_mean(per_channel,mask)}
    total = terms["mse"]
    if profile != "mse":
        clean_mask = mask & clean_examples[:,None]
        terms["identity"] = masked_mean(((cleaned-original)/scale).square().mean(-1),clean_mask)
        frequency,predicted_psd = welch_torch(prediction,config["fs"])
        _,target_psd = welch_torch(truth,config["fs"])
        bins = (frequency >= .5)&(frequency <= 40)
        log_error = ((predicted_psd[...,bins]+1e-8).log()-(target_psd[...,bins]+1e-8).log()).square().mean(-1)
        terms["psd"] = masked_mean(log_error,mask)
        first = (prediction-prediction.mean(-1,keepdim=True))*mask[:,:,None]
        second = (truth-truth.mean(-1,keepdim=True))*mask[:,:,None]
        pred_cov = first@first.transpose(1,2)/(first.shape[-1]-1)
        true_cov = second@second.transpose(1,2)/(second.shape[-1]-1)
        relative = (pred_cov-true_cov).square().sum((1,2))/true_cov.square().sum((1,2)).clamp_min(1e-12)
        terms["covariance"] = relative.mean()
        parameters = config["neural"]
        total = total+parameters["identity_weight"]*terms["identity"]+parameters["psd_weight"]*terms["psd"]+parameters["covariance_weight"]*terms["covariance"]
    if profile == "snr":
        snr = 10*torch.log10(truth.square().mean(-1).clamp_min(1e-12)/per_channel.clamp_min(1e-12))
        terms["snr_shortfall"] = masked_mean((20-snr).clamp_min(0).square()/400,mask)
        total = total+config["neural"]["snr_weight"]*terms["snr_shortfall"]
    # Privileged references may reweight paired reconstruction error only.
    # They never force native EEG/reference correlation to vanish.
    weight = config["neural"].get("reference_error_weight",0.)
    if weight:
        if references is None:
            raise ValueError("Reference-error loss needs training-only HEOG/VEOG")
        refs = references/references.square().mean(-1,keepdim=True).sqrt().clamp_min(1e-12)
        gram = refs@refs.transpose(1,2)
        regularizer = .01*gram.diagonal(dim1=1,dim2=2).mean(-1)[:,None,None]
        operator = torch.linalg.solve(gram+regularizer*torch.eye(2,device=refs.device)[None],refs)
        projection = (error@refs.transpose(1,2))@operator
        terms["reference_error"] = masked_mean(projection.square().mean(-1),mask)
        total = total+weight*terms["reference_error"]
    terms["total"] = total
    return total,terms
