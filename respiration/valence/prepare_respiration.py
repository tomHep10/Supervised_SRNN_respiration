"""
prepare_respiration.py  (created by Claude)  — VALENCE prepare

Turn raw respiration .h5 + BORIS behavior .csv into windowed SRNN arrays:
    observations.npy : (n_windows, T, D)  float32   cleaned+z-scored respiration
    labels.npy       : (n_windows, T, 1)  int       per-timepoint behavior state
    meta.npz / label_map.json

Steps per recording:
  1. read respiration (20 kHz) + trial_type (-> valence)
  2. CLEAN with the user's pipeline: lowpass(butter N=4) -> downsample -> bandpass 0.1-20 Hz
  3. z-score (model conditioning)
  4. rasterize BORIS bouts -> per-timepoint state labels (only the 3 sniff types)
Then, PER RECORDING, the signal is tiled into non-overlapping contiguous 30 s windows
and the N richest in sniffing are kept.

The signal-processing helpers are shared with the rank pipeline and live in
../resp_pipeline.py (the single source of truth). This file holds only the valence-specific
policy: valence comes from the RIx prefix (valence_map), and the prepared set is the explicit
`recordings` list in the config.

Run from the repo root:
    python respiration/valence/prepare_respiration.py --config respiration/valence/config_respiration_hpg.yaml
"""
import os, json, argparse, sys
import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # respiration/
from resp_pipeline import (find_one, load_resp, preprocess_resp_to_target_rate,
                           read_bouts, rasterize, dead_signal_mask)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="respiration/valence/config_respiration.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    P, W, paths = cfg["preprocess"], cfg["windowing"], cfg["paths"]
    fs_t = int(P["target_fs"])
    T = int(round(W["window_sec"] * fs_t))            # non-overlapping window length
    N = int(W["windows_per_recording"])
    os.makedirs(paths["out_dir"], exist_ok=True)

    rm_dead = P.get("remove_dead_signal", True)
    max_dead = float(P.get("max_window_dead_frac", 0.20))

    obs_all, lab_all, val_all, rid_all, names, starts_all, dead_all = [], [], [], [], [], [], []
    print(f"window T={T} ({W['window_sec']}s)  keep top {N} sniffing-richest per recording"
          f"{'  | dead-signal removal ON' if rm_dead else ''}\n" + "=" * 70)
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

        # mark flat/dead respiration BEFORE windowing. Dead stretches are not behavior
        # (whatever BORIS labeled), so zero their labels; and they must not be picked as
        # "sniffing-rich" windows, so forbid windows dominated by dead signal below.
        if rm_dead:
            dead = dead_signal_mask(resp, fs_t, P.get("dead_win_sec", 1.0),
                                    P.get("dead_rel_thresh", 0.25), P.get("dead_abs_thresh", 0.0),
                                    P.get("dead_min_sec", 2.0), P.get("dead_ref_pct", 75.0))
            labels[dead] = 0
        else:
            dead = np.zeros(M, dtype=bool)
        sniff = (labels > 0).astype(np.float64)        # sniffing = any of the 3 sniff states

        # tile into NON-OVERLAPPING contiguous windows; drop windows that are mostly dead,
        # then score the rest by sniffing fraction
        starts = list(range(0, M - T + 1, T))
        clean = [s for s in starts if dead[s:s + T].mean() <= max_dead]
        scored = sorted(clean, key=lambda s: sniff[s:s + T].mean(), reverse=True)
        keep = sorted(scored[:N])                      # top-N, restored to chronological order
        for s in keep:
            obs_all.append(resp[s:s + T, None]); lab_all.append(labels[s:s + T, None])
            val_all.append(valence); rid_all.append(rid); starts_all.append(s)
            dead_all.append(float(dead[s:s + T].mean()))
        names.append(stem)
        fr = [f"{100*sniff[s:s+T].mean():.0f}%" for s in keep]
        print(f"{stem}: trial={trial} val={valence} dur={len(raw)/fs:.0f}s -> {M} samp | "
              f"{len(starts)} cand, {len(starts)-len(clean)} dropped(dead), kept {len(keep)} | "
              f"dead {100*dead.mean():.0f}% of rec | sniff% per kept: {fr}")

    observations = np.asarray(obs_all, dtype=np.float32)
    labels_arr = np.asarray(lab_all, dtype=np.int64)
    valence_arr = np.asarray(val_all, dtype=np.int64)
    rid_arr = np.asarray(rid_all, dtype=np.int64)
    np.save(os.path.join(paths["out_dir"], "observations.npy"), observations)
    np.save(os.path.join(paths["out_dir"], "labels.npy"), labels_arr)
    np.savez(os.path.join(paths["out_dir"], "meta.npz"), valence=valence_arr, recording_id=rid_arr,
             recording_names=np.array(names), target_fs=fs_t, T=T, window_starts=np.array(starts_all),
             window_dead_frac=np.array(dead_all, dtype=np.float32))
    json.dump(P["behavior_states"], open(os.path.join(paths["out_dir"], "label_map.json"), "w"), indent=2)
    print("=" * 70)
    print(f"observations {observations.shape}  labels {labels_arr.shape}")
    print(f"windows: pos={int((valence_arr==1).sum())} neg={int((valence_arr==0).sum())} -> {paths['out_dir']}")


if __name__ == "__main__":
    main()
