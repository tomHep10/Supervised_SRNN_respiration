"""
prepare_rank.py  (created by Claude)

Cagemate / RANK version of prepare_respiration.py. SAME signal pipeline (50 Hz clean +
dead-signal removal + 30 s top-N-sniffing windowing) — it reuses those helpers verbatim
from prepare_respiration.py. The only differences from the valence prep are:

  * data is auto-discovered by globbing CM_*_merged.h5 in paths.h5_dir (the filenames are
    inconsistent, so each H5 is paired to its BORIS csv by the filename STEM, not a fixed
    token count),
  * the per-recording target is RANK (0 = Subordinate, 1 = Dominant) from config.rank_map,
    keyed by the subject token sX_Y parsed out of the filename,
  * meta carries `rank`, plus per-window `subject` and `cage_id` for leave-one-cage-out.

Run from the repo root:
    python respiration/rank/prepare_rank.py --config respiration/rank/config_rank_hpg.yaml
"""
import os, re, glob, json, argparse, sys
import numpy as np

# reuse the EXACT shared signal-processing helpers (single source of truth, ../resp_pipeline.py)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # respiration/
from resp_pipeline import (
    preprocess_resp_to_target_rate, load_resp, read_bouts, rasterize, dead_signal_mask,
)

SUBJ_RE = re.compile(r"s(\d+)_(\d+)")          # first sX_Y in the name = the FOCAL subject


def parse_subject(stem):
    """Return (subject_token, cage) e.g. 'CM_s1_1_d1_2_...' -> ('s1_1', '1'). The partner
    token (d.. / sub..) never starts with 's<digit>', so the leftmost match is the focal."""
    m = SUBJ_RE.search(stem)
    if not m:
        raise ValueError(f"no sX_Y subject token in '{stem}'")
    return f"s{m.group(1)}_{m.group(2)}", m.group(1)


def find_csv(csv_dir, stem):
    """Pair an H5 to its BORIS csv by filename stem. Names vary: '<stem>.csv',
    '<stem>.1.csv', '<stem>.1_VT.csv' -> match anything that starts with the stem."""
    hits = sorted(c for c in glob.glob(os.path.join(csv_dir, "*.csv"))
                  if os.path.basename(c).startswith(stem))
    return hits[0] if hits else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="respiration/rank/config_rank_hpg.yaml")
    args = ap.parse_args()
    import yaml
    cfg = yaml.safe_load(open(args.config))
    P, W, paths = cfg["preprocess"], cfg["windowing"], cfg["paths"]
    disc = cfg.get("discovery", {})
    rank_map = cfg["rank_map"]
    fs_t = int(P["target_fs"])
    T = int(round(W["window_sec"] * fs_t))
    N = int(W["windows_per_recording"])
    os.makedirs(paths["out_dir"], exist_ok=True)

    rm_dead = P.get("remove_dead_signal", True)
    max_dead = float(P.get("max_window_dead_frac", 0.20))
    max_bout = float(P.get("max_bout_sec", 0.0))            # drop BORIS bouts longer than this (artifacts)
    skip_no_csv = bool(disc.get("skip_no_csv", True))
    min_sniff = int(disc.get("min_sniff_bouts", 0))        # drop recordings with too few real sniff bouts
    force_include = set(disc.get("force_include_subjects", []) or [])   # keep these even if below min_sniff
    exclude_cages = set(str(c) for c in (disc.get("exclude_cages", []) or []))  # drop whole cages
    h5s = sorted(glob.glob(os.path.join(paths["h5_dir"], disc.get("glob", "*_merged.h5"))))

    obs_all, lab_all, rank_all, rid_all, subj_all, cage_all = [], [], [], [], [], []
    names, starts_all, dead_all = [], [], []
    print(f"window T={T} ({W['window_sec']}s)  keep top {N} sniffing-richest per recording"
          f"{'  | dead-signal removal ON' if rm_dead else ''}  | TARGET=rank (LOCO)\n" + "=" * 78)

    rid = 0
    for h5p in h5s:
        stem = os.path.basename(h5p)
        for suf in ("_merged.h5", ".h5"):
            if stem.endswith(suf):
                stem = stem[: -len(suf)]; break
        try:
            subject, cage = parse_subject(stem)
        except ValueError as e:
            print(f"SKIP {stem}: {e}"); continue
        if subject not in rank_map:
            print(f"SKIP {stem}: subject {subject} not in rank_map (not an included animal)")
            continue
        if cage in exclude_cages:
            print(f"SKIP {stem}: cage {cage} in exclude_cages")
            continue
        rank = int(rank_map[subject])

        csvp = find_csv(paths["csv_dir"], stem)
        if csvp is None and skip_no_csv:
            print(f"SKIP {stem}: no BORIS csv (skip_no_csv=true)"); continue

        # read + filter behavior, and DECIDE INCLUSION before the expensive signal load
        bouts, n_long, n_sniff = None, 0, 0
        if csvp is not None:
            bouts = read_bouts(csvp, P.get("subject_only", True))
            if max_bout > 0 and {"Start (s)", "Stop (s)"} <= set(bouts.columns):
                dur = bouts["Stop (s)"] - bouts["Start (s)"]
                n_long = int((dur > max_bout).sum())
                bouts = bouts[dur <= max_bout]
            n_sniff = int(bouts["Behavior"].isin(P["behavior_states"]).sum()) if "Behavior" in bouts else 0
        if n_sniff < min_sniff and subject not in force_include:
            print(f"SKIP {stem}: only {n_sniff} sniff bouts (< min_sniff_bouts={min_sniff}; "
                  f"in cage but no usable behavior). Add '{subject}' to force_include_subjects "
                  f"to keep it."); continue
        if n_sniff < min_sniff and subject in force_include:
            print(f"  (force-include {subject}: {n_sniff} sniff bouts, below min — windows "
                  f"will be chronological)")

        raw, fs, _ = load_resp(h5p, P["resp_key"])
        resp = preprocess_resp_to_target_rate(raw, fs, fs_t, P["lowpass_order"],
                                              P["bandpass_low"], P["bandpass_high"], P["bandpass_order"])
        if P.get("zscore", True):
            resp = (resp - resp.mean()) / (resp.std() + 1e-8)
        M = resp.shape[0]

        labels = rasterize(bouts, M, fs_t, P["behavior_states"]) if bouts is not None \
            else np.zeros(M, dtype=np.int64)

        if rm_dead:
            dead = dead_signal_mask(resp, fs_t, P.get("dead_win_sec", 1.0),
                                    P.get("dead_rel_thresh", 0.25), P.get("dead_abs_thresh", 0.0),
                                    P.get("dead_min_sec", 2.0), P.get("dead_ref_pct", 75.0))
            labels[dead] = 0
        else:
            dead = np.zeros(M, dtype=bool)
        sniff = (labels > 0).astype(np.float64)

        starts = list(range(0, M - T + 1, T))
        clean = [s for s in starts if dead[s:s + T].mean() <= max_dead]
        scored = sorted(clean, key=lambda s: sniff[s:s + T].mean(), reverse=True)
        keep = sorted(scored[:N])
        for s in keep:
            obs_all.append(resp[s:s + T, None]); lab_all.append(labels[s:s + T, None])
            rank_all.append(rank); rid_all.append(rid)
            subj_all.append(subject); cage_all.append(cage)
            starts_all.append(s); dead_all.append(float(dead[s:s + T].mean()))
        names.append(stem)
        fr = [f"{100*sniff[s:s+T].mean():.0f}%" for s in keep]
        print(f"{stem}: subj={subject} cage={cage} rank={'Dom' if rank else 'Sub'} "
              f"dur={len(raw)/fs:.0f}s -> {M} samp | {len(starts)} cand, "
              f"{len(starts)-len(clean)} dropped(dead), kept {len(keep)} | "
              f"{f'{n_long} long-bouts cut | ' if n_long else ''}"
              f"{'no-csv ' if csvp is None else ''}dead {100*dead.mean():.0f}% of rec | sniff% per kept: {fr}")
        rid += 1

    if not obs_all:
        print("No recordings kept — check h5_dir / rank_map / skip_no_csv."); return

    observations = np.asarray(obs_all, dtype=np.float32)
    labels_arr = np.asarray(lab_all, dtype=np.int64)
    rank_arr = np.asarray(rank_all, dtype=np.int64)
    rid_arr = np.asarray(rid_all, dtype=np.int64)
    np.save(os.path.join(paths["out_dir"], "observations.npy"), observations)
    np.save(os.path.join(paths["out_dir"], "labels.npy"), labels_arr)
    np.savez(os.path.join(paths["out_dir"], "meta.npz"), rank=rank_arr, recording_id=rid_arr,
             recording_names=np.array(names), subject=np.array(subj_all), cage_id=np.array(cage_all),
             target_fs=fs_t, T=T, window_starts=np.array(starts_all),
             window_dead_frac=np.array(dead_all, dtype=np.float32))
    json.dump(P["behavior_states"], open(os.path.join(paths["out_dir"], "label_map.json"), "w"), indent=2)
    print("=" * 78)
    n_cage = len(np.unique([parse_subject(n)[1] for n in names]))
    print(f"observations {observations.shape}  labels {labels_arr.shape}")
    print(f"windows: Dom={int((rank_arr==1).sum())} Sub={int((rank_arr==0).sum())}  "
          f"recordings={len(names)}  cages={n_cage} (-> leave-one-cage-out folds 0-{n_cage-1}) "
          f"-> {paths['out_dir']}")


if __name__ == "__main__":
    main()
