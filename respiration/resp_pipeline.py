"""
resp_pipeline.py  (created by Claude)

SHARED respiration signal-processing helpers used by BOTH experiments' prepare scripts
(valence/prepare_respiration.py and rank/prepare_rank.py). This is the single source of
truth for: recording<->file pairing, the lowpass->downsample->bandpass cleaning, the H5
loader, the BORIS reader, dead-signal detection, and behavior rasterization.

Nothing experiment-specific lives here (no valence_map, no rank_map, no windowing policy) —
those stay in the per-experiment prepare scripts.
"""
import os, glob
import numpy as np
import h5py
import pandas as pd
from scipy.signal import butter, filtfilt, resample_poly, sosfiltfilt
from scipy.ndimage import uniform_filter1d


def rec_key(fname):
    return "_".join(os.path.basename(fname).split("_")[:3])   # e.g. 'RI1_s1_1'


def find_one(directory, stem, ext):
    matches = [h for h in glob.glob(os.path.join(directory, "*" + ext)) if rec_key(h) == stem]
    if not matches:
        raise FileNotFoundError(f"No '{stem}*{ext}' in {directory}")
    return sorted(matches)[0]


def preprocess_resp_to_target_rate(raw_signal, fs, target_rate, lp_order, bp_low, bp_high, bp_order):
    """User's lowpass -> downsample -> bandpass pipeline (nk butterworth == scipy sosfiltfilt)."""
    raw_signal = np.asarray(raw_signal, dtype=np.float64).flatten()
    fs = float(fs); target_rate = float(target_rate)
    # 1) anti-alias lowpass at target_rate/2
    norm_cutoff = (target_rate / 2) / (fs / 2)
    b, a = butter(N=lp_order, Wn=norm_cutoff, btype="low")
    filtered = filtfilt(b, a, raw_signal)
    # 2) downsample
    down = max(1, int(fs // target_rate))
    downsampled = resample_poly(filtered, up=1, down=down)
    # 3) bandpass 0.1-20 Hz (== nk.signal_filter butterworth)
    sos = butter(bp_order, [bp_low, bp_high], btype="bandpass", output="sos", fs=target_rate)
    rsp_cleaned = sosfiltfilt(sos, downsampled)
    return np.asarray(rsp_cleaned, dtype=np.float64)


def load_resp(h5_path, resp_key):
    with h5py.File(h5_path, "r") as h:
        resp = np.asarray(h[resp_key][()]).astype(np.float64).reshape(-1)
        md = dict(h["metadata"].attrs) if "metadata" in h else {}
        em = dict(h["ekg_metadata"].attrs) if "ekg_metadata" in h else {}
    return resp, float(em.get("sampling_frequency", 20000.0)), str(md.get("trial_type", rec_key(h5_path)[:3]))


def read_bouts(csv_path, subject_only):
    df = pd.read_csv(csv_path)
    if subject_only and "Subject" in df.columns:
        df = df[df["Subject"] == "subject"]
    return df


def _remove_short_runs(mask, min_len):
    """Set any True-run shorter than min_len samples back to False."""
    mask = mask.astype(bool).copy()
    padded = np.concatenate(([0], mask.astype(np.int8), [0]))
    edges = np.flatnonzero(np.diff(padded))          # alternating run starts/ends
    for s, e in zip(edges[0::2], edges[1::2]):
        if e - s < min_len:
            mask[s:e] = False
    return mask


def dead_signal_mask(resp, fs, win_sec, rel_thresh, abs_thresh, min_sec, ref_pct=75.0):
    """Boolean mask (len == len(resp)) marking 'dead' respiration: stretches whose local
    amplitude is far below the recording's typical amplitude (flat / low-amplitude noise
    that still carries faint resp-like wiggles). Local amplitude = rolling std over a
    win_sec window. The reference "typical active amplitude" is the ref_pct-th percentile
    of that local std (p75 by default, NOT the median: a recording that is itself heavily
    dead drags its median down, which would shrink the threshold and under-flag the worst
    recordings -- a high percentile is robust to that). A sample is dead if its local std
    falls below rel_thresh * reference, or below abs_thresh outright (z-scored units, if
    set). Dead runs shorter than min_sec are ignored so brief between-breath dips survive.

    NB measured on this dataset the local-std distribution is unimodal (dead signal is a
    ~1-5% low tail), so bimodal auto-thresholds (Otsu/GMM) fail -- they split the active
    distribution in half and flag ~50%. A robust percentile-relative cut is the right tool."""
    w = max(1, int(round(win_sec * fs)))
    local_mean = uniform_filter1d(resp, w, mode="nearest")
    local_var = uniform_filter1d(resp * resp, w, mode="nearest") - local_mean ** 2
    local_std = np.sqrt(np.clip(local_var, 0.0, None))
    ref = np.percentile(local_std, ref_pct)           # robust "typical active" amplitude
    dead = local_std < rel_thresh * ref
    if abs_thresh > 0:
        dead |= local_std < abs_thresh
    return _remove_short_runs(dead, max(1, int(round(min_sec * fs))))


def rasterize(df, n, fs, behavior_states):
    labels = np.zeros(n, dtype=np.int64)
    for _, r in df.iterrows():
        beh = str(r.get("Behavior", "")).strip()
        if beh not in behavior_states:
            continue
        s0 = max(0, min(int(round(float(r["Start (s)"]) * fs)), n))
        s1 = max(0, min(int(round(float(r["Stop (s)"]) * fs)), n))
        if s1 > s0:
            labels[s0:s1] = int(behavior_states[beh])
    return labels
