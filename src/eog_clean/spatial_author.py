"""Call pinned MATLAB author routines through Octave; do not replace algorithms."""
from pathlib import Path
import hashlib
import io
import json
import os
import subprocess
import zipfile
import numpy as np
import requests
from scipy.io import savemat, loadmat
from .author_models import fetch_author_source
from .data import save_json


def prepare_backend(source_root, destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    specs = json.loads((Path(source_root)/'docs/provenance/spatial_author_sources.json').read_text())
    for key in ['sgeyesub','eeglab']:
        fetch_author_source(specs[key],destination/key)
    archive = requests.get(specs['regica']['url'],timeout=(10,40))
    archive.raise_for_status()
    if hashlib.sha256(archive.content).hexdigest()!=specs['regica']['sha256']:
        raise ValueError('Original REG-ICA archive checksum differs')
    (destination/'REG-ICAv1.00.zip').write_bytes(archive.content)
    with zipfile.ZipFile(io.BytesIO(archive.content)) as stream:
        for name in stream.namelist():
            target=(destination/name).resolve()
            if not target.is_relative_to(destination.resolve()):
                raise ValueError('Author archive path escapes destination')
        stream.extractall(destination)
    # Equivalent one-hot encoding replaces the MATLAB neural-toolbox helper.
    (destination/'ind2vec.m').write_text('''function Y=ind2vec(labels)
labels=labels(:)';
assert(all(labels>=1 & labels==round(labels)));
Y=sparse(labels,1:numel(labels),1,max(labels),numel(labels));
end
''')
    save_json(destination/'source_identities.json',specs)
    return destination


def octave_call(backend, directory, body, timeout=1200):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    def quote(path):return str(Path(path).resolve()).replace("'","''")
    program="addpath(genpath('"+quote(backend)+"'));\npkg load signal;\n"
    program+="rand('seed',42);randn('seed',42);\n"+body
    script=directory/'author_call.m';script.write_text(program,encoding='utf-8')
    environment=os.environ.copy()
    environment.update(OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    try:
        result=subprocess.run(['octave','--no-gui','--quiet',str(script.resolve())],
            capture_output=True,text=True,timeout=timeout,env=environment,cwd=directory)
    except subprocess.TimeoutExpired as error:
        (directory/'octave.log').write_text(str(error),encoding='utf-8')
        raise RuntimeError('Original author routine exceeded declared runtime cap') from error
    (directory/'octave.log').write_text(result.stdout+'\n'+result.stderr,encoding='utf-8')
    if result.returncode:
        raise RuntimeError('Original author backend failed: '+result.stderr[-1400:])


def fit_sgeyesub(values, labels, backend, directory):
    labels=np.asarray(labels).reshape(-1)
    if labels.shape!=(values.shape[-1],) or not set(range(1,7)).issubset(set(labels)):
        raise ValueError('SGEYESUB needs verified calibration sample classes1–6')
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    savemat(directory/'input.mat',{'X':np.asarray(values,float),'labels':labels[None]})
    octave_call(backend,directory,"""load('input.mat');
model=sgeyesub();model.fit(X,labels,1:size(X,1));
C=model.C;W=model.W;A=model.A;
assert(all(isfinite(C(:))));
save('-mat7-binary','model.mat','C','W','A');
""")
    result=loadmat(directory/'model.mat')
    return result['C']


def fit_regica(values, backend, directory):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    savemat(directory/'input.mat',{'X':np.asarray(values,float)})
    octave_call(backend,directory,"""load('input.mat');
mu=mean(X,2);centered=X-mu;
dim=rank(centered);
assert(dim>=2);
if dim<size(X,1)
 [weights,sphere]=runica(centered,'extended',1,'pca',dim,'maxsteps',512,'stop',1e-7,'verbose','off','interrupt','off');
else
 [weights,sphere]=runica(centered,'extended',1,'maxsteps',512,'stop',1e-7,'verbose','off','interrupt','off');
end
U=weights*sphere;A=pinv(U);S=U*centered;
selected=regica_fd(S);
assert(all(isfinite(U(:))) && ~isempty(selected));
save('-mat7-binary','model.mat','U','A','mu','selected');
""")
    return loadmat(directory/'model.mat')


def predict_regica(values, eyes, model, backend, directory):
    """Frozen calibration ICA/selector; author SRLS adapts separately within a trial.

    This fit/score wrapper is a documented adaptation. It uses the original
    plugin's fractal selector and SRLS correction, not ridge regression.
    """
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    savemat(directory/'input.mat',{'X':np.asarray(values,float),'EOG':np.asarray(eyes,float),
        **{key:model[key] for key in ['U','A','mu','selected']}})
    octave_call(backend,directory,"""load('input.mat');
S=U*(X-mu);opt=struct('refdata',EOG,'M',3,'lambda',0.999,'sigma',0.01,'prec',50,'verbose',0);
corrected=scrls_regica(S(selected,:),opt);
% Original SRLS initializes the first M-1 outputs to zero; retain them as raw
% component values to avoid a boundary deletion unrelated to artifact evidence.
corrected(:,1:2)=S(selected,1:2);
artifact=A(:,selected)*(S(selected,:)-corrected);
prediction=X-artifact;
assert(all(isfinite(prediction(:))));
save('-mat7-binary','prediction.mat','prediction');
""",timeout=300)
    return loadmat(directory/'prediction.mat')['prediction']


def calibration_sample_labels(item, trials):
    """Nearest sample mapping; categorical labels are never bandpass filtered."""
    if item['sample_labels'] is None:
        raise ValueError('Original OSF sample-level artifactclasses absent')
    labels=[]
    for trial in trials:
        original=item['sample_labels'][trial['index']]
        indices=np.minimum(np.floor(np.arange(trial['eeg'].shape[-1])*item['fs']/200).astype(int),len(original)-1)
        labels.append(original[indices])
    return np.concatenate(labels)
