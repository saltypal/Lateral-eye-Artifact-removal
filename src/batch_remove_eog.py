"""
batch_remove_eog.py
===================
Batch version of remove_eog_channels.py.

Recursively scans:
    INPUT_DIR  =  .../Dataset1_osf/
    finds all  *_prep.set / *_prep.fdt  pairs

For each participant, removes the 25 non-EEG channels and saves:
    OUTPUT_DIR / {study}_{participant}_eeg_only.npz
    OUTPUT_DIR / {study}_{participant}_eeg_only.fdt
    OUTPUT_DIR / {study}_{participant}_eeg_only_meta.json

Reference channel list loaded once from:
    DATASET / without_eog_channels.set  (MATLAB v5, 58 EEG channels)

Run from src/ directory:
    conda run -p D:\\Others\\Miniconda\\envs\\BCI --no-capture-output python batch_remove_eog.py
"""

from pathlib import Path
import json
import traceback

import numpy as np
import h5py
import hdf5storage
import scipy.io as sio

# ─────────────────────────────────────────────────────────────
# Configurable paths
# ─────────────────────────────────────────────────────────────
_SRC       = Path(__file__).resolve().parent
DATASET    = _SRC.parent / "dataset"
INPUT_DIR  = DATASET / "complete_dataset" / "Dataset1_osf"
OUTPUT_DIR = DATASET / "complete_dataset" / "complete_eeg"
REF_SET    = DATASET / "without_eog_channels.set"   # 58-channel reference (MATLAB v5)


# ─────────────────────────────────────────────────────────────
# Helper: decode HDF5 uint16 char array to string
# ─────────────────────────────────────────────────────────────
def _u16_to_str(arr) -> str:
    return "".join(chr(int(c)) for c in np.asarray(arr).flatten())


def _deref_str(ref, fh) -> str:
    try:
        val = fh[ref][()]
        if np.asarray(val).dtype.kind in ("u", "i"):
            return _u16_to_str(val)
        return str(np.asarray(val).flat[0])
    except Exception:
        return ""


def _deref_num(ref, fh) -> float:
    try:
        return float(np.asarray(fh[ref][()]).flat[0])
    except Exception:
        return 0.0


# ─────────────────────────────────────────────────────────────
# Step 0 — Load reference channel list (once)
# ─────────────────────────────────────────────────────────────
def load_ref_channels(ref_set: Path) -> list[str]:
    mat = hdf5storage.loadmat(str(ref_set))
    return [str(ch["labels"][0]) for ch in mat["chanlocs"][0]]


# ─────────────────────────────────────────────────────────────
# Step 1 — Read channel names from a MATLAB v7.3 HDF5 .set file
# ─────────────────────────────────────────────────────────────
def read_channels_h5(path: Path) -> list[str]:
    names = []
    with h5py.File(str(path), "r") as f:
        labels_ds = f["EEG"]["chanlocs"]["labels"]   # (n_ch, 1) of obj refs
        for ref in labels_ds[:, 0]:
            names.append(_u16_to_str(f[ref][()]))
    return names


# ─────────────────────────────────────────────────────────────
# Step 2 — Read EEG data
# ─────────────────────────────────────────────────────────────
def read_data(set_path: Path, fdt_path: Path) -> tuple:
    """
    Returns (raw, n_ch, n_pnts, n_ep, srate)
    raw shape: (n_ep, n_ch, n_pnts)  float32
    """
    with h5py.File(str(set_path), "r") as f:
        eeg    = f["EEG"]
        n_ch   = int(np.array(eeg["nbchan"]).item())
        n_pnts = int(np.array(eeg["pnts"]).item())
        n_ep   = int(np.array(eeg["trials"]).item())
        srate  = int(np.array(eeg["srate"]).item())

        # Determine if data is inline or external .fdt
        data_field = eeg["data"][()]
        is_external = (data_field.dtype == np.dtype("uint16")
                       or data_field.size < n_ch * n_pnts)

    if is_external:
        flat     = np.fromfile(str(fdt_path), dtype=np.float32)
        expected = n_ch * n_pnts * n_ep
        if flat.size != expected:
            raise ValueError(
                f"FDT size mismatch in {fdt_path.name}: "
                f"got {flat.size}, expected {expected} ({n_ch}×{n_pnts}×{n_ep})"
            )
        # MATLAB column-major (n_ch, n_pnts, n_ep) → Python (n_ep, n_ch, n_pnts)
        raw = (flat.reshape((n_ch, n_pnts, n_ep), order="F")
                   .transpose(2, 0, 1)
                   .astype(np.float32, copy=False))
    else:
        raw = data_field.astype(np.float32)
        if raw.ndim == 3 and raw.shape != (n_ep, n_ch, n_pnts):
            raw = raw.transpose(2, 1, 0)

    assert raw.shape == (n_ep, n_ch, n_pnts), \
        f"Shape check failed: {raw.shape} ≠ ({n_ep}, {n_ch}, {n_pnts})"

    return raw, n_ch, n_pnts, n_ep, srate


# ─────────────────────────────────────────────────────────────
# Step 3 — Read events
# ─────────────────────────────────────────────────────────────
def read_events(set_path: Path) -> list[dict]:
    events_out = []
    with h5py.File(str(set_path), "r") as fh:
        ev_grp = fh["EEG"]["event"]

        # field may be absent in some files
        def _field(name):
            return ev_grp[name] if name in ev_grp else None

        type_ds    = _field("type")
        latency_ds = _field("latency")
        epoch_ds   = _field("epoch")
        dur_ds     = _field("duration")

        if type_ds is None:
            return []

        n_events = type_ds.shape[0]

        for i in range(n_events):
            ev = {
                "type":     _deref_str(type_ds[i, 0], fh),
                "latency":  _deref_num(latency_ds[i, 0], fh) if latency_ds is not None else 0.0,
                "epoch":    int(_deref_num(epoch_ds[i, 0], fh)) if epoch_ds is not None else 0,
                "duration": _deref_num(dur_ds[i, 0], fh)       if dur_ds    is not None else 0.0,
            }
            events_out.append(ev)
    return events_out


# ─────────────────────────────────────────────────────────────
# Step 4 — Save outputs
# ─────────────────────────────────────────────────────────────
def save_outputs(
    data_eeg:    np.ndarray,     # (n_ep, 58, n_pnts)
    ch_names:    list[str],
    srate:       int,
    events:      list[dict],
    to_remove:   list[str],
    set_path:    Path,
    out_dir:     Path,
    stem:        str,
    zero_padded: list[str] | None = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_npz  = out_dir / f"{stem}_eeg_only.npz"
    out_fdt  = out_dir / f"{stem}_eeg_only.fdt"
    out_set  = out_dir / f"{stem}_eeg_only.set"
    out_json = out_dir / f"{stem}_eeg_only_meta.json"

    n_ep, n_ch, n_pnts = data_eeg.shape

    ev_epochs   = np.array([e["epoch"]    for e in events], dtype=np.int32)
    ev_latency  = np.array([e["latency"]  for e in events], dtype=np.float64)
    ev_duration = np.array([e["duration"] for e in events], dtype=np.float64)
    ev_types    = np.array([e["type"]     for e in events], dtype=object)

    np.savez(
        str(out_npz),
        data        = data_eeg,
        ch_names    = np.array(ch_names, dtype=object),
        srate       = np.array([srate], dtype=np.int32),
        n_epochs    = np.array([n_ep],   dtype=np.int32),
        n_pnts      = np.array([n_pnts], dtype=np.int32),
        ev_epochs   = ev_epochs,
        ev_latency  = ev_latency,
        ev_duration = ev_duration,
        ev_types    = ev_types,
    )

    # EEGLAB-compatible FDT: (n_ch, n_pnts, n_ep) Fortran order
    (data_eeg.transpose(1, 2, 0)              # (n_ch, n_pnts, n_ep)
             .flatten(order="F")
             .astype(np.float32)
             .tofile(str(out_fdt)))

    # EEGLAB-compatible SET (MATLAB v5 struct pointing to companion .fdt)
    fdt_filename = out_fdt.name   # just the filename, not full path

    # chanlocs — minimal struct array with 'labels' field
    chanlocs = np.zeros(n_ch, dtype=np.dtype([
        ("labels",   "O"), ("theta",    "O"), ("radius",   "O"),
        ("X",        "O"), ("Y",        "O"), ("Z",        "O"),
        ("sph_theta","O"), ("sph_phi",  "O"), ("sph_radius","O"),
        ("type",     "O"), ("ref",      "O"), ("urchan",   "O"),
    ]))
    for i, name in enumerate(ch_names):
        chanlocs[i]["labels"]    = name
        chanlocs[i]["theta"]     = np.array([[0.0]])
        chanlocs[i]["radius"]    = np.array([[0.0]])
        chanlocs[i]["X"]         = np.array([[0.0]])
        chanlocs[i]["Y"]         = np.array([[0.0]])
        chanlocs[i]["Z"]         = np.array([[0.0]])
        chanlocs[i]["sph_theta"] = np.array([[0.0]])
        chanlocs[i]["sph_phi"]   = np.array([[0.0]])
        chanlocs[i]["sph_radius"]= np.array([[0.0]])
        chanlocs[i]["type"]      = ""
        chanlocs[i]["ref"]       = ""
        chanlocs[i]["urchan"]    = np.array([[float(i + 1)]])

    # event — struct array
    if events:
        ev_dtype = np.dtype([
            ("type",     "O"), ("latency",  "O"),
            ("epoch",    "O"), ("duration", "O"),
        ])
        ev_arr = np.zeros(len(events), dtype=ev_dtype)
        for i, ev in enumerate(events):
            ev_arr[i]["type"]     = ev["type"]
            ev_arr[i]["latency"]  = np.array([[float(ev["latency"])]])
            ev_arr[i]["epoch"]    = np.array([[float(ev["epoch"])]])
            ev_arr[i]["duration"] = np.array([[float(ev["duration"])]])
    else:
        ev_arr = np.zeros(0, dtype=np.dtype([
            ("type","O"),("latency","O"),("epoch","O"),("duration","O")]))

    eeg_struct = dict(
        setname  = stem,
        filename = fdt_filename,
        filepath = str(out_dir),
        data     = fdt_filename,        # tells EEGLAB to read companion .fdt
        nbchan   = np.array([[float(n_ch)]]),
        pnts     = np.array([[float(n_pnts)]]),
        trials   = np.array([[float(n_ep)]]),
        srate    = np.array([[float(srate)]]),
        xmin     = np.array([[0.0]]),
        xmax     = np.array([[(n_pnts - 1) / srate]]),
        times    = np.arange(n_pnts, dtype=np.float64) / srate,
        chanlocs = chanlocs,
        event    = ev_arr,
        icaweights = np.zeros((0, 0)),
        icasphere  = np.zeros((0, 0)),
        icaact     = np.zeros((0, 0)),
    )
    sio.savemat(str(out_set), {"EEG": eeg_struct}, do_compression=False)

    types_found = sorted({e["type"] for e in events})
    meta_json = {
        "source":            set_path.name,
        "reference":         REF_SET.name,
        "n_channels":        n_ch,
        "n_epochs":          n_ep,
        "n_pnts":            n_pnts,
        "srate":             srate,
        "ch_names":          ch_names,
        "removed_channels":  to_remove,
        "zero_padded_channels": zero_padded or [],
        "fdt_layout":        "(n_ch, n_pnts * n_epochs) float32 Fortran-order",
        "npz_data_shape":    f"({n_ep}, {n_ch}, {n_pnts})",
        "event_types":       types_found,
        "n_events":          len(events),
    }
    with open(str(out_json), "w") as jf:
        json.dump(meta_json, jf, indent=2)


# ─────────────────────────────────────────────────────────────
# Main: process one file
# ─────────────────────────────────────────────────────────────
def process_one(set_path: Path, fdt_path: Path, ref_ch: list[str], out_dir: Path) -> bool:
    stem = set_path.stem.replace("_prep", "")   # e.g. study01_p01
    print(f"\n  Processing  {set_path.parent.name}/{set_path.name}")

    try:
        # Channel cross-check
        prep_ch   = read_channels_h5(set_path)
        to_remove = [ch for ch in prep_ch if ch not in ref_ch]
        missing   = [ch for ch in ref_ch  if ch not in prep_ch]

        if missing:
            print(f"    NOTE: {len(missing)} ref channels absent — will zero-pad: {missing}")

        # Build keep_idx only for ref channels that exist in prep
        keep_idx  = [prep_ch.index(ch) if ch in prep_ch else -1 for ch in ref_ch]

        # Read data
        raw, n_ch, n_pnts, n_ep, srate = read_data(set_path, fdt_path)
        print(f"    shape={raw.shape}  srate={srate}")

        # Read events
        events = read_events(set_path)
        types  = sorted({e["type"] for e in events})
        print(f"    events={len(events)}  types={types}")

        # Build 58-channel output — zero-pad channels that are missing in prep
        n_ref    = len(ref_ch)
        data_eeg = np.zeros((n_ep, n_ref, n_pnts), dtype=np.float32)
        for out_i, src_i in enumerate(keep_idx):
            if src_i >= 0:
                data_eeg[:, out_i, :] = raw[:, src_i, :]
            # else: remains zero

        n_present = sum(1 for i in keep_idx if i >= 0)
        print(f"    Output shape: {data_eeg.shape}  "
              f"({n_present} real channels, {n_ref - n_present} zero-padded)")

        # Save — pass missing list so it appears in metadata
        save_outputs(data_eeg, ref_ch, srate, events,
                     to_remove, set_path, out_dir, stem,
                     zero_padded=missing)

        out_npz = out_dir / f"{stem}_eeg_only.npz"
        out_fdt = out_dir / f"{stem}_eeg_only.fdt"
        out_set = out_dir / f"{stem}_eeg_only.set"
        print(f"    Saved  NPZ: {out_npz.name}  ({out_npz.stat().st_size / 1e6:.1f} MB)")
        print(f"    Saved  FDT: {out_fdt.name}  ({out_fdt.stat().st_size / 1e6:.1f} MB)")
        print(f"    Saved  SET: {out_set.name}  ({out_set.stat().st_size / 1e3:.1f} KB)")
        return True

    except Exception as e:
        print(f"    ERROR: {e}")
        traceback.print_exc()
        return False


# ─────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("BATCH EOG CHANNEL REMOVAL")
    print("=" * 70)
    print(f"  Input dir  : {INPUT_DIR}")
    print(f"  Output dir : {OUTPUT_DIR}")
    print(f"  Reference  : {REF_SET}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load reference channel list once
    ref_ch = load_ref_channels(REF_SET)
    print(f"\n  Reference channels ({len(ref_ch)}): {ref_ch[:5]} … {ref_ch[-3:]}")

    # Find all *_prep.set files recursively
    set_files = sorted(INPUT_DIR.rglob("*_prep.set"))
    print(f"\n  Found {len(set_files)} prep .set files")

    ok_count  = 0
    err_count = 0
    skipped   = 0

    for set_path in set_files:
        stem_id = set_path.stem.replace("_prep", "")
        out_npz = OUTPUT_DIR / f"{stem_id}_eeg_only.npz"
        out_set = OUTPUT_DIR / f"{stem_id}_eeg_only.set"
        if out_npz.exists() and out_set.exists():
            print(f"\n  Skipping {set_path.name}  (output already exists)")
            skipped += 1
            continue

        # Companion .fdt must exist alongside .set
        fdt_path = set_path.with_suffix(".fdt")
        if not fdt_path.exists():
            print(f"\n  SKIP (no .fdt found): {set_path}")
            err_count += 1
            continue

        success = process_one(set_path, fdt_path, ref_ch, OUTPUT_DIR)
        if success:
            ok_count += 1
        else:
            err_count += 1

    print("\n" + "=" * 70)
    print(f"BATCH COMPLETE")
    print(f"  Processed : {ok_count}")
    print(f"  Errors    : {err_count}")
    print(f"  Skipped   : {skipped}  (already done)")
    print(f"  Output dir: {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
