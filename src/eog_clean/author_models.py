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


class EEGOAR:
    """Author pretrained weights, seven present electrodes in a 64-row mask."""
    def __init__(self, source, names, device):
        module = import_file('pinned_eegoar', Path(source)/'model/eegoarnet_torch.py')
        self.model = module.EEGOARNET().to(device).eval()
        weights = torch.load(Path(source)/'weights/EEGOARNet_torch_weights.pt',
                             map_location=device, weights_only=True)
        self.model.load_state_dict(weights, strict=True)
        # Only this hash-verified author metadata dictionary needs pickle.
        channel_set = np.load(Path(source)/'materials/channel_set_64ch.npy', allow_pickle=True).item()
        order = list(channel_set.keys())
        upper = [str(name).upper() for name in order]
        self.indices = [upper.index(name.upper()) for name in names]
        self.device = device
        self.parameters = sum(value.numel() for value in self.model.parameters())

    def predict(self, values):
        values = np.asarray(values)
        downsampled = resample_poly(values, 16, 25, axis=-1)
        samples = downsampled.shape[-1]
        padded = np.pad(downsampled, ((0,0),(0,(-samples)%128)), mode='reflect')
        windows = padded.shape[-1]//128
        inputs = np.zeros((windows, 1, 128, 64), dtype=np.float32)
        for channel, index in enumerate(self.indices):
            inputs[:, 0, :, index] = padded[channel].reshape(windows, 128)
        mask = torch.zeros((windows,64), device=self.device)
        mask[:, self.indices] = 1
        with torch.inference_mode():
            outputs = self.model(torch.from_numpy(inputs).to(self.device), mask).cpu().numpy()
        selected = np.stack([outputs[:, 0, :, index].reshape(-1)[:samples]
                             for index in self.indices])
        return resample_poly(selected, 25, 16, axis=-1)[:, :values.shape[-1]]


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
