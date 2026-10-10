"""Whole-record offline inference; source windows never cross trial boundaries."""
from pathlib import Path
import time
import numpy as np
import torch
from .data import channel_metadata
from .neural import DeploymentStudent,RegionalBandRouter,input_scale


class EEGCleaner:
    def __init__(self,model,config,device="cpu",recipe=None):
        self.model = model.to(device).eval()
        self.config,self.device,self.recipe = config,torch.device(device),recipe

    @classmethod
    def from_checkpoint(cls,path,config,device="cpu"):
        from .neural import build_model
        from .contracts import canonical_hash
        saved = torch.load(Path(path),map_location="cpu",weights_only=True)
        if saved["configuration_hash"] != canonical_hash(config):
            raise ValueError("Checkpoint/configuration identity differs")
        model = build_model(saved["experiment"]["arm"],config)
        model.load_state_dict(saved["state_dict"])
        recipe = None
        if isinstance(model,RegionalBandRouter) and model.variant in ("regional","no_context","all_vmd"):
            from .autovmd import FrozenVMDRecipe
            recipe = FrozenVMDRecipe(saved["experiment"]["K"],saved["experiment"]["alpha"],config["fs"],
                config["classical_grid"]["tolerance"],config["classical_grid"]["max_iterations"])
        return cls(model,config,device,recipe)

    def denoise_record(self,eeg,channel_names,channel_mask=None):
        """Input must already satisfy declared fs/band contract, one continuous trial.

        Names may be unknown; unknown anatomy uses the shared branch. Missing
        EEG channels are masked and preserved, never reconstructed by guessing.
        """
        values = np.asarray(eeg,dtype=np.float32)
        if values.ndim != 2 or values.shape[-1] < 2 or len(channel_names) != values.shape[0]:
            raise ValueError("Expected channel-aligned continuous EEG")
        mask = np.ones(values.shape[0],dtype=bool) if channel_mask is None else np.asarray(channel_mask,dtype=bool)
        if mask.shape != (values.shape[0],) or not np.isfinite(values[mask]).all():
            raise ValueError("Invalid available-channel samples or mask")
        metadata = channel_metadata(channel_names)
        tensor_metadata = {key:torch.from_numpy(metadata[key]).unsqueeze(0).to(self.device) for key in ("regions","hemispheres")}
        tensor_mask = torch.from_numpy(mask).unsqueeze(0).to(self.device)
        window,hop = self.config["window"],self.config["hop"]
        pad = window//2
        padded = np.pad(values,((0,0),(pad,pad+window)),mode="reflect")
        artifact = np.zeros_like(padded)
        denominator = np.zeros(padded.shape[-1],dtype=np.float32)
        weights = np.hanning(window+2)[1:-1].astype(np.float32)
        diagnostics = []
        started = time.perf_counter()
        with torch.no_grad():
            for start in range(0,padded.shape[-1]-window+1,hop):
                block = padded[:,start:start+window]
                tensor = torch.from_numpy(block.copy()).unsqueeze(0).to(self.device)
                bundle = None
                if self.recipe:
                    from .autovmd import AutoVMD
                    scope = "all" if self.model.variant == "all_vmd" else "frontal"
                    transformed = AutoVMD(self.config).transform(np.where(mask[:,None],block,0),self.recipe,metadata["regions"],scope)
                    bundle = {key:torch.from_numpy(transformed[key]).unsqueeze(0).to(self.device) for key in ("components","descriptors","valid")}
                    bundle["scale"] = input_scale(torch.where(tensor_mask[:,:,None],tensor,torch.zeros_like(tensor))).unsqueeze(-2)
                    diagnostics.extend({"window_start":start-pad,**issue} for issue in transformed["diagnostics"] if issue.get("failure"))
                if isinstance(self.model,DeploymentStudent):
                    result = self.model(tensor,tensor_mask,tensor_metadata)
                else:
                    result = self.model(tensor,tensor_mask,tensor_metadata,bundle)
                estimate = result["artifact"][0].cpu().numpy()
                if not np.isfinite(estimate).all():
                    estimate = np.zeros_like(block)
                    diagnostics.append({"window_start":start-pad,"failure":"nonfinite correction","fallback":"input"})
                artifact[:,start:start+window] += estimate*weights
                denominator[start:start+window] += weights
        original = slice(pad,pad+values.shape[-1])
        if (denominator[original] <= 0).any():
            raise RuntimeError("Overlap-add leaves uncovered samples")
        artifact = artifact[:,original]/denominator[original]
        return {"cleaned":values-artifact,"artifact":artifact,
                "diagnostics":diagnostics,"runtime_s":time.perf_counter()-started,
                "offline":True,"sampling_rate":self.config["fs"]}


def export_student_onnx(model,path,channels=10,window=1024):
    """Dynamic batch/channel export; numerical parity is a separate required test."""
    if not isinstance(model,DeploymentStudent):
        raise ValueError("Deployment export must be VMD-free")

    class ExportWrapper(torch.nn.Module):
        def __init__(self,student):
            super().__init__()
            self.student = student

        def forward(self,eeg,mask,regions,hemispheres):
            return self.student(eeg,mask,{"regions":regions,"hemispheres":hemispheres})["cleaned"]

    model = model.cpu().eval()
    inputs = (torch.zeros(1,channels,window),torch.ones(1,channels,dtype=torch.bool),
              torch.full((1,channels),3,dtype=torch.long),torch.full((1,channels),3,dtype=torch.long))
    torch.onnx.export(ExportWrapper(model),inputs,str(path),opset_version=17,dynamo=False,
        input_names=["eeg","mask","regions","hemispheres"],output_names=["cleaned"],
        dynamic_axes={name:{0:"batch",1:"channels"} for name in ("eeg","mask","regions","hemispheres","cleaned")})
