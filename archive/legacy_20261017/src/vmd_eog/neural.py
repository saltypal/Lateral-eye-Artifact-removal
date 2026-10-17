"""EEG-only regional routing and deployment student.

BandRouteNet-inspired adaptation, not an author-code replication. Spatial
weights are shared; attention exchanges components within each EEG channel.
HEOG/VEOG and clean targets are deliberately absent from the forward API.
"""
import torch
from torch import nn
from torch.nn import functional as F

BANDS = ((.5,4.), (4.,8.), (8.,13.), (13.,30.), (30.,40.))


def input_scale(eeg):
    return eeg.square().mean(-1, keepdim=True).sqrt().clamp_min(1e-12)


def fourier_components(eeg, fs=200):
    """Five disjoint physical bands and an exact complementary component."""
    length = eeg.shape[-1]
    frequencies = torch.fft.rfftfreq(length, 1/fs, device=eeg.device)
    spectrum = torch.fft.rfft(eeg.float(), dim=-1)
    pieces = []
    for index, (lower, upper) in enumerate(BANDS):
        included = (frequencies >= lower) & ((frequencies <= upper) if index == 4 else (frequencies < upper))
        pieces.append(torch.fft.irfft(spectrum*included, n=length, dim=-1))
    bands = torch.stack(pieces, dim=-2)
    residual = eeg.float()-bands.sum(-2)
    components = torch.cat((bands,residual.unsqueeze(-2)), dim=-2)
    return components, component_descriptors(components,fs)


def component_descriptors(components, fs):
    power = torch.fft.rfft(components.float(),dim=-1).abs().square()
    frequencies = torch.fft.rfftfreq(components.shape[-1],1/fs,device=components.device)
    energy = power.sum(-1).clamp_min(1e-20)
    centers = (power*frequencies).sum(-1)/energy
    bandwidth = ((power*(frequencies-centers.unsqueeze(-1)).square()).sum(-1)/energy).sqrt()
    fractions = energy/energy.sum(-1,keepdim=True).clamp_min(1e-20)
    residual = torch.zeros_like(centers)
    residual[..., -1] = 1
    return torch.stack((centers/fs,bandwidth/fs,fractions,torch.zeros_like(centers),residual),dim=-1)


def frontal_context(features, mask, regions, hemispheres):
    """[B,C,D,L] -> common, signed R-L, midline, global, availability."""
    frontal = mask & (regions == 0)
    pools, availability = [], []
    for side in range(3):
        selected = frontal & (hemispheres == side)
        count = selected.sum(1,keepdim=True)
        pooled = (features*selected[:,:,None,None]).sum(1)/count.clamp_min(1)[:,:,None]
        pools.append(pooled)
        availability.append((count > 0).to(features.dtype).unsqueeze(-1).expand(-1,-1,features.shape[-1]))
    left,right,midline = pools
    common_mask = frontal
    common = (features*common_mask[:,:,None,None]).sum(1)/common_mask.sum(1).clamp_min(1)[:,None,None]
    global_pool = (features*mask[:,:,None,None]).sum(1)/mask.sum(1).clamp_min(1)[:,None,None]
    bilateral_available = (availability[0]*availability[1])
    signed = (right-left)*bilateral_available
    return torch.cat((common,signed,midline,global_pool,*availability),dim=1)


class InceptionBlock(nn.Module):
    def __init__(self,width):
        super().__init__()
        self.branches = nn.ModuleList(nn.Conv1d(width,width//4,k,padding=k//2) for k in (3,7,15,31))
        self.normalization = nn.GroupNorm(4,width)

    def forward(self,features):
        mixed = torch.cat([branch(features) for branch in self.branches],dim=1)
        return features+F.gelu(self.normalization(mixed))


class TemporalEncoder(nn.Module):
    def __init__(self,width):
        super().__init__()
        self.input = nn.Conv1d(1,width,5,padding=2)
        self.first = nn.Sequential(InceptionBlock(width),InceptionBlock(width))
        self.second = nn.Sequential(InceptionBlock(width),InceptionBlock(width))
        self.gru = nn.GRU(width,width,batch_first=True)

    def forward(self,values):
        first = self.first(self.input(values))
        second = self.second(F.avg_pool1d(first,2,ceil_mode=True))
        bottleneck = F.avg_pool1d(second,2,ceil_mode=True)
        recurrent,_ = self.gru(bottleneck.transpose(1,2))
        return recurrent.transpose(1,2),(first,second)


class RegionalHeads(nn.Module):
    def __init__(self,width):
        super().__init__()
        self.heads = nn.ModuleList(nn.Conv1d(width,1,1) for _ in range(3))
        for head in self.heads:
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(self,features,regions):
        result = torch.zeros_like(features[:,:1])
        for index,head in enumerate(self.heads):
            selected = (regions == index) if index < 2 else (regions >= 2)
            result = result+head(features)*selected[:,None,None]
        return result


class RegionalBandRouter(nn.Module):
    def __init__(self,width=32,variant="regional",fs=200,component_dropout=.1):
        super().__init__()
        if variant not in ("fixed","regional","no_context","all_vmd","raw"):
            raise ValueError("Unknown routing arm")
        self.variant,self.width,self.fs = variant,width,fs
        self.component_dropout = component_dropout
        self.raw_encoder = TemporalEncoder(width)
        self.component_encoder = TemporalEncoder(width)
        self.frequency = nn.Sequential(nn.Linear(5,width),nn.GELU(),nn.Linear(width,width))
        self.context = nn.Conv1d(2*(4*width+3),width,1)
        self.film = nn.Conv1d(2*width,2*width,1)
        self.router = nn.Sequential(nn.Conv1d(width,width,7,padding=3,groups=width),nn.GELU(),nn.Conv1d(width,width,1),nn.Sigmoid())
        self.refinement = InceptionBlock(width)
        self.attention = nn.MultiheadAttention(width,4,batch_first=True)
        self.fusion = nn.Conv1d(2*width,width,1)
        self.decoder_second = nn.Sequential(InceptionBlock(width),InceptionBlock(width))
        self.decoder_first = nn.Sequential(InceptionBlock(width),InceptionBlock(width))
        self.heads = RegionalHeads(width)

    def representations(self,normalized,regions,mode_bundle):
        components,descriptors = fourier_components(normalized,self.fs)
        valid = torch.ones(components.shape[:-1],device=normalized.device,dtype=torch.bool)
        if self.variant == "raw":
            components = normalized.unsqueeze(-2)
            descriptors = component_descriptors(components,self.fs)
            valid = torch.ones(components.shape[:-1],device=normalized.device,dtype=torch.bool)
        elif self.variant in ("regional","no_context","all_vmd"):
            selected = torch.ones_like(regions,dtype=torch.bool) if self.variant == "all_vmd" else regions == 0
            if selected.any():
                if mode_bundle is None:
                    raise ValueError("This arm requires verified VMD components; no silent Fourier substitution")
                vmd = mode_bundle["components"]/mode_bundle["scale"]
                number = max(components.shape[-2],vmd.shape[-2])
                components = F.pad(components,(0,0,0,number-components.shape[-2]))
                descriptors = F.pad(descriptors,(0,0,0,number-descriptors.shape[-2]))
                valid = F.pad(valid,(0,number-valid.shape[-1]),value=False)
                vmd = F.pad(vmd,(0,0,0,number-vmd.shape[-2]))
                vd = F.pad(mode_bundle["descriptors"],(0,0,0,number-mode_bundle["descriptors"].shape[-2]))
                vm = F.pad(mode_bundle["valid"],(0,number-mode_bundle["valid"].shape[-1]),value=False)
                components = torch.where(selected[:,:,None,None],vmd,components)
                descriptors = torch.where(selected[:,:,None,None],vd,descriptors)
                valid = torch.where(selected[:,:,None],vm,valid)
        if self.training and self.component_dropout:
            dropped = torch.rand(normalized.shape[:2],device=normalized.device) < self.component_dropout
            fallback = torch.zeros_like(components)
            fallback[:,:,0] = normalized
            fallback_valid = torch.zeros_like(valid)
            fallback_valid[:,:,0] = True
            fallback_descriptor = component_descriptors(fallback,self.fs)
            components = torch.where(dropped[:,:,None,None],fallback,components)
            descriptors = torch.where(dropped[:,:,None,None],fallback_descriptor,descriptors)
            valid = torch.where(dropped[:,:,None],fallback_valid,valid)
        if not valid.any(-1).all():
            raise ValueError("Each channel must have a valid component or explicit raw fallback")
        return components,descriptors,valid

    def route(self,encoded,descriptors,valid,raw,context):
        batch,channels,number,width,length = encoded.shape
        guidance = context[:,None].expand(-1,channels,-1,-1).reshape(batch*channels,width,length)
        scale,offset = self.film(torch.cat((raw.reshape(batch*channels,width,length),guidance),1)).chunk(2,1)
        conditioned = encoded+self.frequency(descriptors).unsqueeze(-1)
        conditioned = conditioned*(1+scale.tanh().reshape(batch,channels,1,width,length))+offset.reshape(batch,channels,1,width,length)
        flat = conditioned.reshape(-1,width,length)
        gate = self.router(flat)
        routed = (1-gate)*flat+gate*self.refinement(flat)
        tokens = routed.reshape(batch,channels,number,width,length).permute(0,1,4,2,3).reshape(-1,number,width)
        keys = (~valid)[:,:,None].expand(-1,-1,length,-1).reshape(-1,number)
        attended,_ = self.attention(tokens,tokens,tokens,key_padding_mask=keys,need_weights=False)
        attended = attended.reshape(batch,channels,length,number,width).permute(0,1,3,4,2)
        pooled = (attended*valid[:,:,:,None,None]).sum(2)/valid.sum(2).clamp_min(1)[:,:,None,None]
        return pooled,gate.reshape(batch,channels,number,width,length).mean(3)

    def forward(self,eeg,channel_mask,metadata,mode_bundle=None):
        batch,channels,length = eeg.shape
        mask = channel_mask.bool()
        values = torch.where(mask[:,:,None],eeg,torch.zeros_like(eeg))
        scale = input_scale(values)
        normalized = values/scale
        regions,hemispheres = metadata["regions"],metadata["hemispheres"]
        raw,skips = self.raw_encoder(normalized.reshape(-1,1,length))
        reduced = raw.shape[-1]
        raw = raw.reshape(batch,channels,self.width,reduced)
        components,descriptors,valid = self.representations(normalized,regions,mode_bundle)
        encoded,_ = self.component_encoder(components.reshape(-1,1,length))
        encoded = encoded.reshape(batch,channels,components.shape[-2],self.width,reduced)
        raw_context = frontal_context(raw,mask,regions,hemispheres)
        if self.variant == "no_context":
            raw_context = torch.zeros_like(raw_context)
        initial_context = self.context(torch.cat((raw_context,torch.zeros_like(raw_context)),1))
        routed,gates = self.route(encoded,descriptors,valid,raw,initial_context)
        routed_context = frontal_context(routed,mask,regions,hemispheres)
        if self.variant == "no_context":
            routed_context = torch.zeros_like(routed_context)
        communicated = self.context(torch.cat((raw_context,routed_context),1))
        supported,supported_gates = self.route(encoded,descriptors,valid,raw,communicated)
        frontal = (regions == 0)[:,:,None,None]
        fused = torch.where(frontal,routed,supported)
        fused = self.fusion(torch.cat((raw,fused),2).reshape(-1,2*self.width,reduced))
        first,second = skips
        decoded = self.decoder_second(F.interpolate(fused,size=second.shape[-1],mode="linear",align_corners=False)+second)
        decoded = self.decoder_first(F.interpolate(decoded,size=length,mode="linear",align_corners=False)+first)
        artifact = self.heads(decoded,regions.reshape(-1)).reshape(batch,channels,length)*scale*mask[:,:,None]
        return {"cleaned":eeg-artifact,"artifact":artifact,
                "diagnostics":{"router":torch.where(frontal,gates,supported_gates),"component_valid":valid}}


class DeploymentStudent(nn.Module):
    """Shared TCN-BiGRU; no frequency decomposition at deployment."""
    def __init__(self,width=32,fs=200):
        super().__init__()
        self.fs,self.width = fs,width
        self.input = nn.Conv1d(1,width,5,padding=2)
        self.blocks = nn.ModuleList(nn.Sequential(
            nn.Conv1d(width,width,5,padding=2*dilation,dilation=dilation),
            nn.GroupNorm(4,width),nn.GELU()) for dilation in (1,2,4,8,16))
        self.gru = nn.GRU(width,32,batch_first=True,bidirectional=True)
        self.project = nn.Conv1d(64,width,1)
        self.context = nn.Conv1d(4*width+3,width,1)
        self.fusion = nn.Conv1d(2*width,width,1)
        self.heads = RegionalHeads(width)

    def forward(self,eeg,channel_mask,metadata):
        batch,channels,length = eeg.shape
        mask = channel_mask.bool()
        values = torch.where(mask[:,:,None],eeg,torch.zeros_like(eeg))
        scale = input_scale(values)
        encoded = self.input((values/scale).reshape(-1,1,length))
        for block in self.blocks:
            encoded = encoded+block(encoded)
        recurrent,_ = self.gru(encoded.transpose(1,2))
        encoded = self.project(recurrent.transpose(1,2)).reshape(batch,channels,self.width,length)
        context = self.context(frontal_context(encoded,mask,metadata["regions"],metadata["hemispheres"]))
        context = context[:,None].expand(-1,channels,-1,-1)
        fused = self.fusion(torch.cat((encoded,context),2).reshape(-1,2*self.width,length))
        artifact = self.heads(fused,metadata["regions"].reshape(-1)).reshape(batch,channels,length)*scale*mask[:,:,None]
        return {"cleaned":eeg-artifact,"artifact":artifact,"diagnostics":{}}


def build_model(arm,config):
    if arm == "student":
        return DeploymentStudent(config["neural"]["width"],config["fs"])
    return RegionalBandRouter(config["neural"]["width"],arm,config["fs"],config["neural"]["component_drop_probability"])
