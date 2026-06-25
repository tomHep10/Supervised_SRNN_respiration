"""
prepare_respiration.py  (created by Claude)

Turn raw respiration .h5 + BORIS behavior .csv into windowed SRNN arrays:
    observations.npy : (n_windows, T, D)  float32   cleaned+z-scored respiration
    labels.npy       : (n_windows, T, 1)  int       per-timepoint behavior state
    meta.npz / label_map.json

Steps per recording:
  1. read respiration (20 kHz) + trial_type (-> valence)
  2. CLEAN with the user's pipeline: lowpass(butter N=4, cutoff=target_fs/2) ->
     downsample(resample_poly down=fs//target_fs) -> bandpass(butter order 2, 0.1-20 Hz).
     The bandpass is the scipy-equivalent of nk.signal_filter(method="butterworth").
  3. z-score (model conditioning)
  4. rasterize BORIS bouts -> per-timepoint state labels (only the 3 sniff types)
Then, PER RECORDING, the signal is tiled into non-overlapping contiguous 30 s windows
and the N richest in sniffing are kept. Each window is a natural continuous segment of
the original recording (no overlap, no stitching, no event-centering).

Run from the repo root:
    conda run -n sleap-new python respiration/prepare_respiration.py --config respiration/config_respiration.yaml
"""
import os, glob, json, argparse
import numpy as np
import h5py
import pandas as pd
import yaml
from scipy.signal import butter, filtfilt, resample_poly, sosfiltfilt


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="respiration/config_respiration.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    P, W, paths = cfg["preprocess"], cfg["windowing"], cfg["paths"]
    fs_t = int(P["target_fs"])
    T = int(round(W["window_sec"] * fs_t))            # non-overlapping window length
    N = int(W["windows_per_recording"])
    os.makedirs(paths["out_dir"], exist_ok=True)

    obs_all, lab_all, val_all, rid_all, names, starts_all = [], [], [], [], [], []
    print(f"window T={T} ({W['window_sec']}s)  keep top {N} sniffing-richest per recording\n" + "=" * 70)
    for rid, stem in enumerate(cfg["recordings"]):
        h5p = find_one(paths["h5_dir"], stem, ".h5")
        csvp = find_one(paths["csv_dir"], stem, ".csv")
        raw, fs, trial = load_resp(h5p, P["resp_key"])
        valence = int(cfg["valence_map"][trial])
        resp = preprocess_resp_to_target_rate(raw, fs, fs_t, P["lowpass_order"],
                                              P["bandpass_low"], P["bandpass_high"], P["bandpass_order"])
        if P.get("zscore", True):
            resp = (resp - resp.mean()) / (resp.std() + 1e-8)
        M = resp.shape[0]
        labels = rasterize(read_bouts(csvp, P.get("subject_only", True)), M, fs_t, P["behavior_states"])
        sniff = (labels > 0).astype(np.float64)        # sniffing = any of the 3 sniff states

        # tile into NON-OVERLAPPING contiguous windows, score each by sniffing fraction
        starts = list(range(0, M - T + 1, T))
        scored = sorted(starts, key=lambda s: sniff[s:s + T].mean(), reverse=True)
        keep = sorted(scored[:N])                      # top-N, restored to chronological order
        for s in keep:
            obs_all.append(resp[s:s + T, None]); lab_all.append(labels[s:s + T, None])
            val_all.append(valence); rid_all.append(rid); starts_all.append(s)
        names.append(stem)
        fr = [f"{100*sniff[s:s+T].mean():.0f}%" for s in keep]
        print(f"{stem}: trial={trial} val={valence} dur={len(raw)/fs:.0f}s -> {M} samp | "
              f"{len(starts)} candidate windows, kept {len(keep)} | sniff% per kept: {fr}")

    observations = np.asarray(obs_all, dtype=np.float32)
    labels_arr = np.asarray(lab_all, dtype=np.int64)
    valence_arr = np.asarray(val_all, dtype=np.int64)
    rid_arr = np.asarray(rid_all, dtype=np.int64)
    np.save(os.path.join(paths["out_dir"], "observations.npy"), observations)
    np.save(os.path.join(paths["out_dir"], "labels.npy"), labels_arr)
    np.savez(os.path.join(paths["out_dir"], "meta.npz"), valence=valence_arr, recording_id=rid_arr,
             recording_names=np.array(names), target_fs=fs_t, T=T, window_starts=np.array(starts_all))
    json.dump(P["behavior_states"], open(os.path.join(paths["out_dir"], "label_map.json"), "w"), indent=2)
    print("=" * 70)
    print(f"observations {observations.shape}  labels {labels_arr.shape}")
    print(f"windows: pos={int((valence_arr==1).sum())} neg={int((valence_arr==0).sum())} -> {paths['out_dir']}")


if __name__ == "__main__":
    main()
