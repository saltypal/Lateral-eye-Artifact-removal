"""Small adapters around pinned author code; no replacement architectures."""
import base64
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
import requests
import numpy as np
from scipy.signal import resample_poly
import torch


def fetch_author_source(spec, directory):
    """Download only declared files; verify Git blob identity before importing."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    def fetch(entry):
        target = directory/entry['path']
        content = None
        if target.is_file():
            content = target.read_bytes()
        for attempt in range(4):
            if content is not None:
                blob = b'blob '+str(len(content)).encode()+b'\0'+content
                if hashlib.sha1(blob).hexdigest() == entry['sha']:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(content)
                    return
            url = f"https://raw.githubusercontent.com/{spec['repository']}/{spec['git_sha']}/{entry['path']}"
            try:
                response = requests.get(url, timeout=(10, 35))
                response.raise_for_status()
                content = response.content
                if content.startswith(b'version https://git-lfs'):
                    raise ValueError('LFS pointer is not a weight payload')
            except (requests.RequestException, ValueError):
                response = requests.get(f"https://api.github.com/repos/{spec['repository']}/git/blobs/{entry['sha']}", timeout=35)
                response.raise_for_status()
                content = base64.b64decode(response.json()['content'])
        raise ValueError('Unverified author file: '+entry['path'])

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(fetch, spec['files']))
    (directory/'SOURCE.json').write_text(json.dumps(spec, indent=2))
    return directory


def import_file(name, path):
    specification = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


def pack_frontal_montage(values,names,expected):
    """Place verified labels into a trained layout; reject electrode aliases."""
    normalized=[name.upper() for name in expected]
    if len(set(normalized))!=len(normalized) or len(values)!=len(names):
        raise ValueError('Ambiguous custom montage labels or channel axis')
    positions=[normalized.index(name.upper()) for name in names]
    if len(set(positions))!=len(positions):
        raise ValueError('Duplicate native electrode labels')
    padded=np.zeros((len(expected),values.shape[-1]),dtype=np.float32)
    present=np.zeros(len(expected),dtype=bool)
    padded[positions]=values
    present[positions]=True
    return padded,present,positions


class EEGOAR:
    """Author architecture, pretrained initialisation, custom-montage fine-tuning.

    The published 64-row montage omits FP1/FP2 and uses spatial convolutions.
    Therefore frozen inference with renamed/anonymous slots is invalid. A fixed
    seven-row Klados order plus one absent row defines a new eight-column layout;
    its spatial adaptation must be trained on the training split before scoring.
    """
    def __init__(self, source, names, device):
        module = import_file('pinned_eegoar', Path(source)/'model/eegoarnet_torch.py')
        self.width = 8
        self.model = module.EEGOARNET(ncha=self.width).to(device).eval()
        weights = torch.load(Path(source)/'weights/EEGOARNet_torch_weights.pt',
                             map_location=device, weights_only=True)
        self.model.load_state_dict(weights, strict=True)
        # Author example uses list(np.load(...)[()]); the payload is currently
        # a label array, not a scalar dictionary. Accept either explicit form.
        channel_set = np.load(Path(source)/'materials/channel_set_64ch.npy', allow_pickle=True)
        if channel_set.ndim == 0:
            channel_set = channel_set.item()
        order = list(channel_set) if isinstance(channel_set,dict) else list(np.asarray(channel_set).ravel())
        if len(order) != 64:
            raise ValueError('Author montage must contain exactly 64 labels')
        self.indices = list(range(len(names)))
        self.slot_contract = {'anatomical_aliasing':False,
            'mode':'author architecture with custom montage, requires Klados paired fine-tuning',
            'input_channel_names':names,'fixed_column_indices':self.indices,
            'padded_width':self.width,'absent_column':7,
            'frozen_original_montage_inference_claimed':False,
            'spatial_convolutions_present':True,
            'author_labels':order}
        self.device = device
        self.parameters = sum(value.numel() for value in self.model.parameters())
        self.finetuned = False

    def training_blocks(self, cache):
        inputs,targets=[],[]
        for raw,clean in zip(cache['raw'],cache['clean']):
            scale=max(float(np.sqrt(np.mean(raw**2))),1e-8)
            noisy=resample_poly(raw/scale,16,25,axis=-1)
            target=resample_poly(clean/scale,16,25,axis=-1)
            for start in range(0,noisy.shape[-1]-128+1,128):
                x,y=np.zeros((1,128,self.width),np.float32),np.zeros((1,128,self.width),np.float32)
                x[0,:,:len(raw)]=noisy[:,start:start+128].T
                y[0,:,:len(raw)]=target[:,start:start+128].T
                inputs.append(x);targets.append(y)
        return np.stack(inputs),np.stack(targets)

    def forward_training(self, inputs):
        mask=torch.zeros((len(inputs),self.width),device=inputs.device)
        mask[:,self.indices]=1
        return self.model(inputs,mask)

    def predict(self, values, present=None):
        if not self.finetuned:
            raise RuntimeError('Custom EEGOAR montage must be fine-tuned before evaluation')
        values = np.asarray(values)
        present=np.ones(len(self.indices),dtype=bool) if present is None else np.asarray(present,dtype=bool)
        if values.shape[0]!=len(self.indices) or present.shape!=(len(self.indices),) or not present.any():
            raise ValueError('Custom montage needs seven columns and a nonempty presence mask')
        if np.any(values[~present]!=0):
            raise ValueError('Absent custom montage columns must contain zeros')
        scale=max(float(np.sqrt(np.mean(values[present]**2))),1e-8)
        downsampled = resample_poly(values/scale, 16, 25, axis=-1)
        samples = downsampled.shape[-1]
        padded = np.pad(downsampled, ((0,0),(0,(-samples)%128)), mode='reflect')
        windows = padded.shape[-1]//128
        inputs = np.zeros((windows, 1, 128, self.width), dtype=np.float32)
        for channel, index in enumerate(self.indices):
            inputs[:, 0, :, index] = padded[channel].reshape(windows, 128)
        mask = torch.zeros((windows,self.width), device=self.device)
        mask[:, self.indices] = torch.from_numpy(present.astype(np.float32)).to(self.device)
        with torch.inference_mode():
            outputs = self.model(torch.from_numpy(inputs).to(self.device), mask).cpu().numpy()
        selected = np.stack([outputs[:, 0, :, index].reshape(-1)[:samples]
                             for index in self.indices])
        return resample_poly(selected, 25, 16, axis=-1)[:, :values.shape[-1]]*scale


def bandroute_model(source, device):
    """The author's BRAANetV2; only fs, bands and input count adapt to Klados."""
    sys.path.insert(0, str(source))
    from configs.braanetv2_config import BRAANetV2Config
    module = import_file('pinned_bandroute', Path(source)/'model.py')
    config = BRAANetV2Config(fs=200., signal_length=512, input_channels=1,
        band_defs=[(.5,4.),(4.,8.),(8.,12.),(12.,30.),(30.,80.),(80.,100.)])
    return module.BRAANetV2(config).to(device), config


def bandroute_predict(model, values, device):
    values = np.asarray(values, np.float32)
    channels, samples = values.shape
    if samples % 512:
        raise ValueError('BandRoute comparison needs whole 512-sample blocks')
    scale = np.maximum(np.sqrt(np.mean(values**2, axis=-1, keepdims=True)), 1e-8)
    blocks = (values/scale).reshape(-1, 1, 512)
    with torch.inference_mode():
        prediction = model(torch.from_numpy(blocks).to(device))['y_hat'].cpu().numpy()
    return prediction.reshape(channels, samples)*scale
