"""
remove_eog_channels.py
======================
Reads  : Lateral Eye Dataset/dataset/study01_p01_prep.set/.fdt
         MATLAB v7.3 (HDF5) — 83 channels (58 EEG + EOG + label channels)

Reference: dataset/without_eog_channels.set (MATLAB v5, 58 channels)

Action : Removes the 25 non-EEG channels listed in REMOVE_CHANNELS,
         cross-checks that every channel in without_eog_channels.set
         is present and in the same order.

Saves  : dataset/prep_eeg_only.npz
           data      — float32 (n_epochs, 58, n_pnts)
           ch_names  — (58,) str
           srate     — scalar int
           events    — structured array with fields: epoch, latency, type, duration
         dataset/prep_eeg_only.fdt
           flat float32 binary, Fortran order (EEGLAB convention):
           shape (58, n_pnts * n_epochs)
         dataset/prep_eeg_only_meta.json
           JSON with channel list, srate, n_epochs, n_pnts, n_channels

Run from src/ directory:
    conda run -p D:\\Others\\Miniconda\\envs\\BCI --no-capture-output python remove_eog_channels.py
"""

from pathlib import Path
import json
import struct
import numpy as np
import h5py
import hdf5storage

# ─────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────
_SRC     = Path(__file__).resolve().parent
DATASET  = _SRC.parent / "dataset"
PREP_SET = DATASET / "study01_p01_prep.set"
REF_SET  = DATASET / "without_eog_channels.set"
OUT_NPZ  = DATASET / "prep_eeg_only.npz"
OUT_FDT  = DATASET / "prep_eeg_only.fdt"
OUT_JSON = DATASET / "prep_eeg_only_meta.json"


# ─────────────────────────────────────────────────────────────
# Step 1 — Read channel names from both files
# ─────────────────────────────────────────────────────────────

def _read_channels_h5(path: Path) -> list[str]:
    """Read chanlocs.labels from a MATLAB v7.3 HDF5 .set file."""
    names = []
    with h5py.File(str(path), "r") as f:
        labels_ds = f["EEG"]["chanlocs"]["labels"]   # (n_ch, 1) of obj refs
        for ref in labels_ds[:, 0]:
            names.append("".join(chr(c) for c in f[ref][()].flatten()))
    return names


def _read_channels_mat5(path: Path) -> list[str]:
    """Read chanlocs.labels from a MATLAB v5 .set file."""
    mat = hdf5storage.loadmat(str(path))
    return [str(ch["labels"][0]) for ch in mat["chanlocs"][0]]


print("=" * 60)
print("CHANNEL CROSS-CHECK")
print("=" * 60)

prep_ch = _read_channels_h5(PREP_SET)
ref_ch  = _read_channels_mat5(REF_SET)

print(f"  study01_p01_prep      : {len(prep_ch)} channels")
print(f"  without_eog_channels  : {len(ref_ch)} channels")

# Channels to remove: everything in prep that is NOT in the reference
to_remove  = [ch for ch in prep_ch if ch not in ref_ch]
missing    = [ch for ch in ref_ch  if ch not in prep_ch]

if missing:
    raise RuntimeError(
        f"ABORT: {len(missing)} reference channel(s) not found in prep file: {missing}"
    )

print(f"\n  Channels to remove ({len(to_remove)}):")
for ch in to_remove:
    print(f"    - {ch}")

# Indices of channels to KEEP, in reference order (matches without_eog_channels.set)
keep_idx = [prep_ch.index(ch) for ch in ref_ch]
keep_names = [prep_ch[i] for i in keep_idx]

assert keep_names == ref_ch, "Channel order mismatch — something went wrong"
print(f"\n  Keeping {len(keep_idx)} channels (indices: {keep_idx[:6]} ...)")
print(f"  Channel order matches without_eog_channels.set: YES")


# ─────────────────────────────────────────────────────────────
# Step 2 — Read EEG data from prep .set/.fdt (MATLAB v7.3)
# ─────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("READING DATA")
print("=" * 60)

with h5py.File(str(PREP_SET), "r") as f:
    eeg = f["EEG"]

    # Scalars
    n_ch    = int(np.array(eeg["nbchan"]).item())
    n_pnts  = int(np.array(eeg["pnts"]).item())
    n_ep    = int(np.array(eeg["trials"]).item())
    srate   = int(np.array(eeg["srate"]).item())

    print(f"  nbchan={n_ch}  pnts={n_pnts}  trials={n_ep}  srate={srate}")
    assert n_ch == len(prep_ch), f"Channel count mismatch: metadata={n_ch}, chanlocs={len(prep_ch)}"

    # ── Data ──────────────────────────────────────────────────
    # EEG.data stores the datfile filename as uint16 chars when data is
    # in an external .fdt companion file.
    data_field = eeg["data"][()]
    if data_field.dtype == np.dtype("uint16") or data_field.size < n_ch * n_pnts:
        # External .fdt file — decode the filename
        fdt_name = "".join(chr(c) for c in data_field.flatten())
        print(f"  EEG.data = filename string: '{fdt_name}'  → reading companion .fdt")
    else:
        fdt_name = None

# ── Read from companion .fdt binary file ──────────────────────────────────
if fdt_name is not None:
    # EEGLAB .fdt layout: MATLAB column-major (Fortran order)
    # EEG.data in MATLAB is (n_ch, n_pnts, n_ep), stored column-major:
    # fastest-varying = n_ch, then n_pnts, then n_ep
    fdt_path = DATASET / "study01_p01_prep.fdt"
    print(f"  Reading {fdt_path.name}  ({fdt_path.stat().st_size / 1e6:.1f} MB)")
    flat = np.fromfile(str(fdt_path), dtype=np.float32)
    expected = n_ch * n_pnts * n_ep
    print(f"  flat size={flat.size}  expected={expected}  match={'YES' if flat.size == expected else 'NO'}")
    if flat.size != expected:
        raise ValueError(f"FDT size mismatch: got {flat.size}, expected {expected}")
    # Restore MATLAB column-major (n_ch, n_pnts, n_ep) → Python (n_ep, n_ch, n_pnts)
    raw_matlab = flat.reshape((n_ch, n_pnts, n_ep), order="F")   # Fortran order
    raw = raw_matlab.transpose(2, 0, 1).astype(np.float32, copy=False)
    print(f"  Loaded shape: {raw.shape}  (n_ep={n_ep}, n_ch={n_ch}, n_pnts={n_pnts})")
else:
    raw = data_field.astype(np.float32)
    if raw.ndim == 3 and raw.shape != (n_ep, n_ch, n_pnts):
        raw = raw.transpose(1, 0, 2) if raw.shape == (n_ch, n_ep, n_pnts) else raw
    print(f"  Inline data shape: {raw.shape}")

assert raw.shape == (n_ep, n_ch, n_pnts), f"Shape sanity check failed: {raw.shape}"
print(f"  Final data shape: {raw.shape}   (n_ep={n_ep}, n_ch={n_ch}, n_pnts={n_pnts})")


# ─────────────────────────────────────────────────────────────
# Step 3 — Read events
# ─────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("READING EVENTS")
print("=" * 60)

events_out = []
with h5py.File(str(PREP_SET), "r") as fh:
    ev_grp = fh["EEG"]["event"]
    print(f"  event group keys: {list(ev_grp.keys())}")

    def _deref_str(ref, fh) -> str:
        try:
            val = fh[ref][()]
            if val.dtype.kind in ("u", "i"):
                return "".join(chr(c) for c in val.flatten())
            return str(val.flat[0])
        except Exception:
            return ""

    def _deref_num(ref, fh) -> float:
        try:
            val = fh[ref][()]
            return float(val.flat[0])
        except Exception:
            return 0.0

    # Each field: (n_events, 1) array of HDF5 object references
    type_ds    = ev_grp["type"]
    latency_ds = ev_grp["latency"]
    epoch_ds   = ev_grp["epoch"]
    dur_ds     = ev_grp["duration"]

    print(f"  type shape: {type_ds.shape}  latency shape: {latency_ds.shape}")
    n_events = type_ds.shape[0]
    print(f"  Total events: {n_events}")

    for i in range(n_events):
        events_out.append({
            "type":     _deref_str(type_ds[i, 0],    fh),
            "latency":  _deref_num(latency_ds[i, 0], fh),
            "epoch":    int(_deref_num(epoch_ds[i, 0], fh)),
            "duration": _deref_num(dur_ds[i, 0],     fh),
        })

types_found = sorted({e["type"] for e in events_out})
print(f"  Event types: {types_found}")


# ─────────────────────────────────────────────────────────────
# Step 4 — Select EEG channels only
# ─────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("SELECTING 58 EEG CHANNELS")
print("=" * 60)

data_eeg = raw[:, keep_idx, :]   # (n_ep, 58, n_pnts)
print(f"  Input  shape: {raw.shape}  ({n_ch} channels)")
print(f"  Output shape: {data_eeg.shape}  (58 channels)")

# Cross-verify with without_eog_channels data
print("\n  Verifying channel stats against without_eog_channels.set ...")
ref_mat = hdf5storage.loadmat(str(REF_SET))
# MNE shape from preprocess pipeline: (n_ep, n_ch, n_pnts)
import mne
ref_epochs = mne.io.read_epochs_eeglab(str(REF_SET), verbose=False)
ref_data   = ref_epochs.get_data()  # (n_ep, 58, n_pnts)

print(f"  Extracted  data  shape: {data_eeg.shape}")
print(f"  Reference  data  shape: {ref_data.shape}")

# Scale: extracted data might be in µV, reference in V (MNE converts V→µV internally)
# Check correlation on first epoch, first channel
d_prep = data_eeg[0, 0, :]
d_ref  = ref_data[0, 0, :] * 1e6   # V → µV
corr   = float(np.corrcoef(d_prep, d_ref)[0, 1])
ratio  = float(np.std(d_prep) / (np.std(d_ref) + 1e-12))
print(f"\n  Epoch 0, Ch 0 ({ref_ch[0]}):")
print(f"    prep  :: mean={d_prep.mean():.4f}  std={d_prep.std():.4f}")
print(f"    ref   :: mean={d_ref.mean():.4f}  std={d_ref.std():.4f}  (µV)")
print(f"    Pearson r = {corr:.6f}   std-ratio = {ratio:.4f}")
if corr > 0.999:
    print("    Data match: EXCELLENT (r > 0.999)")
elif corr > 0.99:
    print("    Data match: GOOD (r > 0.99)")
else:
    print("    WARNING: low correlation — check scaling or channel alignment")


# ─────────────────────────────────────────────────────────────
# Step 5 — Save outputs
# ─────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("SAVING OUTPUTS")
print("=" * 60)

# 5a — NumPy archive (.npz)
ev_epochs   = np.array([e["epoch"]    for e in events_out], dtype=np.int32)
ev_latency  = np.array([e["latency"]  for e in events_out], dtype=np.float64)
ev_duration = np.array([e["duration"] for e in events_out], dtype=np.float64)
ev_types    = np.array([e["type"]     for e in events_out], dtype=object)

np.savez(
    str(OUT_NPZ),
    data      = data_eeg,                          # (n_ep, 58, n_pnts)  float32
    ch_names  = np.array(ref_ch, dtype=object),    # (58,)
    srate     = np.array([srate], dtype=np.int32),
    n_epochs  = np.array([n_ep],  dtype=np.int32),
    n_pnts    = np.array([n_pnts], dtype=np.int32),
    ev_epochs   = ev_epochs,
    ev_latency  = ev_latency,
    ev_duration = ev_duration,
    ev_types    = ev_types,
)
print(f"  Saved  NPZ : {OUT_NPZ}  ({OUT_NPZ.stat().st_size / 1e6:.1f} MB)")

# 5b — Flat binary .fdt (float32, MATLAB column-major as EEGLAB expects)
# EEGLAB .fdt layout: (n_ch, n_pnts, n_ep) in Fortran order
# Our data_eeg is (n_ep, 58, n_pnts) → transpose to (58, n_pnts, n_ep)
fdt_matlab = data_eeg.transpose(1, 2, 0)                 # (58, n_pnts, n_ep)
fdt_matlab.flatten(order="F").astype(np.float32).tofile(str(OUT_FDT))
print(f"  Saved  FDT : {OUT_FDT}  ({OUT_FDT.stat().st_size / 1e6:.1f} MB)")

# 5c — JSON metadata
meta_json = {
    "source":        str(PREP_SET.name),
    "reference":     str(REF_SET.name),
    "n_channels":    len(ref_ch),
    "n_epochs":      n_ep,
    "n_pnts":        n_pnts,
    "srate":         srate,
    "ch_names":      ref_ch,
    "removed_channels": to_remove,
    "fdt_layout":    "(n_ch, n_pnts * n_epochs) float32",
    "npz_data_shape": f"({n_ep}, {len(ref_ch)}, {n_pnts})",
    "event_types":   types_found,
    "n_events":      len(events_out),
}
with open(str(OUT_JSON), "w") as jf:
    json.dump(meta_json, jf, indent=2)
print(f"  Saved JSON : {OUT_JSON}")

print("\n" + "=" * 60)
print("DONE")
print("=" * 60)
print(f"  data  : {data_eeg.shape}  float32  (n_epochs, n_ch, n_pnts)")
print(f"  ch    : {ref_ch}")
print(f"  srate : {srate} Hz")
print(f"  events: {len(events_out)}")
print()
print("  Load example:")
print("    import numpy as np")
print(f"    d = np.load('{OUT_NPZ.name}', allow_pickle=True)")
print("    data     = d['data']       # (n_ep, 58, n_pnts)")
print("    ch_names = d['ch_names']   # 58 EEG channel names")
print("    srate    = int(d['srate'])")
