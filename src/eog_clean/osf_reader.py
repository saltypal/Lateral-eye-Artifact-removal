"""Reviewed original-format OSF reader, copied from legacy Git 4c175ab."""
import re
from pathlib import Path
import numpy as np
from scipy.io import loadmat

def read_osf(path: Path) -> dict:
    """Read original EEGLAB arrays; never apply EEG unit scaling to labels.

    EEGLAB stores (channel, sample, trial), with external FDT in Fortran order.
    Keeping original units avoids MNE's EEG-to-volts conversion rounding integer
    annotation channels to zero. Each trial remains a separate continuous unit.
    """
    try:
        payload = loadmat(path, simplify_cells=True)
    except NotImplementedError:
        # Original OSF .set files use MATLAB v7.3/HDF5. pymatreader restores
        # MATLAB axis order and resolves references without EEG unit scaling.
        from pymatreader import read_mat
        payload = read_mat(path)
    eeg = payload.get("EEG", payload)
    channels, samples, trials = (int(eeg[name]) for name in ("nbchan", "pnts", "trials"))
    data = eeg["data"]
    external_source = None
    if isinstance(data, str):
        declared = data
        basename = declared.replace("\\", "/").rsplit("/", 1)[-1]
        external = (path.parent / basename).resolve()
        if not external.is_file():
            # EEGLAB exports can retain an old data filename. MNE's pinned
            # reader uses the same-session .fdt companion in this case.
            companion = path.with_suffix(".fdt").resolve()
            if not companion.is_file():
                raise FileNotFoundError(f"FDT absent: declared={declared!r}; companion={companion.name!r}")
            external = companion
        if external.parent != path.parent.resolve() or external.suffix.lower() != ".fdt":
            raise ValueError("External FDT must be a same-directory session file")
        if external.stat().st_size != channels * samples * trials * 4:
            raise ValueError("External FDT length does not match declared dimensions")
        external_source = {"declared": declared, "resolved": external.name,
                           "same_session_companion_fallback": basename != external.name}
        print("OSF external data", path.stem, external_source, flush=True)
        data = np.memmap(external, dtype="<f4", mode="r", shape=(channels, samples, trials), order="F")
    else:
        data = np.asarray(data).reshape(channels, samples, trials, order="F")
    data = np.moveaxis(data, -1, 0)
    locations = eeg["chanlocs"]
    if isinstance(locations, dict):
        labels = locations["labels"]
        if isinstance(labels, str):
            locations = [locations]
        else:
            # pymatreader represents struct arrays as columns of field values.
            if len(labels) != channels:
                raise ValueError("Channel-location columns have inconsistent length")
            columns = locations
            locations = []
            for index in range(channels):
                location = {}
                for key, column in columns.items():
                    if isinstance(column, (list, tuple, np.ndarray)) and len(column) == channels:
                        location[key] = column[index]
                    else:
                        location[key] = column
                locations.append(location)
    locations = list(locations)
    names = [str(item["labels"]).strip() for item in locations]
    if len(names) != channels or len(set(name.upper() for name in names)) != channels:
        raise ValueError("Missing or duplicate channel names")
    normalized = [name.upper().replace("-", "").replace("_", "") for name in names]
    annotations = {name: normalized.index(name) for name in ["ARTIFACTCLASSES", "LABEL", "BLOCK"] if name in normalized}
    eog_aliases = {"EOG", "HEOG", "VEOG", "EOGH", "EOGV", "LEOG", "REOG", "UEOG", "DEOG",
                   "EOG1", "EOG2", "EOG3", "EOG4", "HEOG1", "HEOG2", "VEOG1", "VEOG2"}
    eog_indices = [i for i, name in enumerate(normalized)
                   if name in eog_aliases or name.startswith(("EOG", "HEOG", "VEOG", "EYE"))
                   or str(locations[i].get("type", "")).upper() == "EOG"]
    excluded = set(annotations.values()) | set(eog_indices)
    eeg_indices = [i for i, item in enumerate(locations) if i not in excluded
                   and str(item.get("type", "EEG")).upper() in ("", "EEG")]
    match = re.fullmatch(r"(study\d+)_(p\d+)_prep", path.stem)
    if match is None:
        raise ValueError("Session filename does not establish participant identity")
    sample_labels = None
    if "ARTIFACTCLASSES" in annotations:
        original = data[:, annotations["ARTIFACTCLASSES"]]
        if not np.allclose(original, np.round(original), atol=1e-4):
            raise ValueError("Sample labels are not unscaled integers")
        sample_labels = np.round(original).astype(np.int16)
        if not np.isin(sample_labels, np.arange(7)).all():
            raise ValueError("Unexpected artifactclasses codes")
    trial_labels = None
    if "LABEL" in annotations:
        trial_labels = np.round(data[:, annotations["LABEL"], 0]).astype(int).tolist()
    return {"path": path, "study": match[1], "participant": match[2], "fs": float(eeg["srate"]),
            "data": data, "names": names, "locations": locations, "eeg_indices": eeg_indices,
            "eog_indices": eog_indices, "annotations": annotations, "sample_labels": sample_labels,
            "trial_labels": trial_labels, "external_source": external_source,
            "reference": str(eeg.get("ref", "not declared"))}
