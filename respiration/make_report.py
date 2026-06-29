"""
make_report.py  (created by Claude)

Consolidate EVERYTHING about one train+eval run into a single plain-text report under
reports/<cohort>/ -- so you never have to dig through logs and CSVs by hand again. Works for
both experiments (valence or rank), auto-detected from the config.

It gathers:
  1. Run config         -- experiment, cohort, split, key preprocessing/model/training params
  2. Data summary       -- window counts, class balance, recordings, cages/subjects, dead %
  3. Training           -- per-fold final held-out reconstruction MSE + final loss (mean/range)
  4. Evaluation         -- the full analysis output (decode / ROC-AUC / permutation / LDA).
                           Either embeds an existing --analysis-log, or runs the analysis fresh.
  5. Artifacts          -- relative paths to every figure, checkpoint dir, log, and this report

Usage (after a run's prepare+train+analyze):
    python respiration/make_report.py --config respiration/rank/config_rank_hpg.yaml
    python respiration/make_report.py --config respiration/valence/config_respiration_hpg.yaml \
        --analysis-log logs/classifier_35946644.log      # embed an existing analysis log (fast)
Output: reports/<cohort>/report_<YYYYmmdd_HHMMSS>.txt  (also copied to report_latest.txt)
"""
import os, sys, glob, re, argparse, subprocess, datetime
import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)


def rel(p):
    """repo-relative path (for portable references in the report)."""
    try:
        return os.path.relpath(os.path.abspath(p), REPO)
    except ValueError:
        return p


def fold_num(p):
    m = re.search(r"fold(\d+)", p)
    return int(m.group(1)) if m else -1


def data_summary(cfg, is_rank, out_dir):
    meta_p = os.path.join(out_dir, "meta.npz")
    obs_p = os.path.join(out_dir, "observations.npy")
    if not os.path.exists(meta_p):
        return [f"  (no prepared data at {rel(out_dir)} -- run prepare first)"]
    meta = np.load(meta_p, allow_pickle=True)
    obs = np.load(obs_p)
    L = [f"  observations: {obs.shape}  (n_windows, T, D)  -> {rel(obs_p)}"]
    names = np.asarray(meta["recording_names"])
    if is_rank:
        rank = np.asarray(meta["rank"])
        cages = sorted(set(np.asarray(meta["cage_id"]).tolist()))
        L.append(f"  windows: {len(rank)}  ({int((rank==1).sum())} Dominant / {int((rank==0).sum())} Subordinate)")
        L.append(f"  recordings: {len(names)}   cages: {len(cages)} {cages}  (leave-one-cage-out)")
    else:
        val = np.asarray(meta["valence"])
        subj = np.array([re.search(r"(s\d+_\d+)", str(s)).group(1) for s in names])
        L.append(f"  windows: {len(val)}  ({int((val==1).sum())} positive(RI1) / {int((val==0).sum())} negative(RI2))")
        L.append(f"  recordings: {len(names)}   subjects: {len(np.unique(subj))}  (leave-one-subject-out)")
    if "window_dead_frac" in meta:
        wdf = np.asarray(meta["window_dead_frac"], dtype=float)
        L.append(f"  window dead-fraction: mean={wdf.mean():.3f}  max={wdf.max():.3f}  "
                 f"({int((wdf>0).sum())}/{len(wdf)} windows contain some dead signal)")
    L.append(f"  recording names: {', '.join(str(n) for n in names)}")
    return L


def training_summary(save_dir, split):
    csvs = sorted(glob.glob(os.path.join(save_dir, f"progress_{split}_fold*.csv")), key=fold_num)
    if not csvs:
        return [f"  (no progress CSVs in {rel(save_dir)} -- run training first)"], None
    L = [f"  per-fold final held-out reconstruction MSE  (from {rel(save_dir)}/progress_{split}_fold*.csv):",
         f"    {'fold':6s} {'final_MSE':>11s} {'final_loss':>13s} {'epochs':>7s}"]
    mses = []
    for c in csvs:
        rows = [r for r in open(c).read().splitlines() if r and not r.startswith("epoch")]
        if not rows:
            continue
        last = rows[-1].split(",")
        ep, loss, mse = last[0], float(last[1]), float(last[2])
        mses.append(mse)
        L.append(f"    {('fold'+str(fold_num(c))):6s} {mse:11.6f} {loss:13.1f} {ep:>7s}")
    if mses:
        mses = np.array(mses)
        L.append(f"    {'MEAN':6s} {mses.mean():11.6f}   (min {mses.min():.6f}, max {mses.max():.6f}, n={len(mses)} folds)")
    return L, (mses if mses is not None else None)


def get_eval_text(cfg, is_rank, split, analysis_log, config_path):
    def _clean(txt):
        return "\n".join(l for l in txt.splitlines() if "nested_tensor" not in l and "warnings.warn" not in l)
    if analysis_log:
        if not os.path.exists(analysis_log):
            return f"  (analysis log not found: {analysis_log})"
        return _clean(open(analysis_log).read())
    # run the analysis fresh
    script = ("respiration/rank/analyze_rank.py" if is_rank else "respiration/valence/analyze_valence.py")
    cmd = [sys.executable, os.path.join(REPO, script), "--config", config_path, "--split", split, "--pca_fold", "0"]
    try:
        out = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, timeout=1800)
        txt = out.stdout
        if out.returncode != 0:
            txt += f"\n[analysis exited {out.returncode}]\n{out.stderr[-2000:]}"
        return _clean(txt)
    except Exception as e:
        return f"  (could not run analysis: {e})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--analysis-log", default=None,
                    help="embed this existing analysis log instead of re-running the analysis")
    ap.add_argument("--stamp", default=None, help="timestamp string (default: now)")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    paths = cfg["paths"]
    is_rank = "rank_map" in cfg
    exp = "RANK (cagemate)" if is_rank else "VALENCE"
    split = cfg["train"].get("split_mode", "cage" if is_rank else "subject")
    cohort = cfg.get("cohort") or ("rank" if is_rank else "valence")
    stamp = args.stamp or datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    P = cfg["preprocess"]; M = cfg["model"]; Tr = cfg["train"]
    lines = []
    w = lines.append
    w("=" * 78)
    w(f"RUN REPORT — {exp}   |   cohort: {cohort}   |   split: leave-one-{split}-out")
    w(f"generated: {stamp}   config: {rel(args.config)}")
    w("=" * 78)

    w("\n## 1. RUN CONFIG")
    w(f"  experiment={exp}  cohort={cohort}  cross-validation=leave-one-{split}-out")
    w(f"  preprocess: target_fs={P['target_fs']}Hz  bandpass {P['bandpass_low']}-{P['bandpass_high']}Hz  "
      f"window={cfg['windowing']['window_sec']}s (T={int(cfg['windowing']['window_sec']*P['target_fs'])})  "
      f"top-{cfg['windowing']['windows_per_recording']} sniffing/recording")
    w(f"  dead-signal removal: {P.get('remove_dead_signal')}  (rel_thresh={P.get('dead_rel_thresh')}, "
      f"p{P.get('dead_ref_pct')}, max_window_dead_frac={P.get('max_window_dead_frac')})")
    if is_rank:
        w(f"  rank filters: min_sniff_bouts={cfg['discovery'].get('min_sniff_bouts')}  "
          f"force_include={cfg['discovery'].get('force_include_subjects')}  "
          f"exclude_cages={cfg['discovery'].get('exclude_cages')}  max_bout_sec={P.get('max_bout_sec')}")
    w(f"  model: num_tv={M['num_tv']}  hidden={M['hidden_shape']}  "
      f"epochs={Tr['epochs']}  lr={Tr['lr']}  coef_cross={Tr['coef_cross']}  seed={cfg['system'].get('seed')}")

    w("\n## 2. DATA")
    for l in data_summary(cfg, is_rank, paths["out_dir"]):
        w(l)

    w("\n## 3. TRAINING (held-out reconstruction)")
    tlines, _ = training_summary(paths["save_dir"], split)
    for l in tlines:
        w(l)

    w("\n## 4. EVALUATION (decode / ROC-AUC / permutation / LDA)")
    src = f"embedded log {rel(args.analysis_log)}" if args.analysis_log else "freshly run analysis"
    w(f"  [source: {src}]")
    w(get_eval_text(cfg, is_rank, split, args.analysis_log, args.config))

    w("\n## 5. ARTIFACTS (relative to repo root)")
    figs = sorted(glob.glob(os.path.join(paths["plot_dir"], "*.png")))
    w(f"  figures ({len(figs)}) in {rel(paths['plot_dir'])}/:")
    for f in figs:
        w(f"    - {rel(f)}")
    w(f"  checkpoints: {rel(paths['save_dir'])}/  (resp_srnn_{split}_h{M['hidden_shape']}_fold*.pt)")
    w(f"  prepared data: {rel(paths['out_dir'])}/")
    w(f"  config: {rel(args.config)}")

    out_dir = os.path.join(REPO, "reports", cohort)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"report_{stamp}.txt")
    text = "\n".join(lines) + "\n"
    open(out_path, "w").write(text)
    open(os.path.join(out_dir, "report_latest.txt"), "w").write(text)
    print(f"wrote {rel(out_path)}  (and reports/{cohort}/report_latest.txt)")


if __name__ == "__main__":
    main()
