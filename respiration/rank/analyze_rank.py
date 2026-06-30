"""
analyze_rank.py  (created by Claude)  — RANK / cagemate analysis

The rank analogue of valence/analyze_valence.py. Same leakage-aware analyses, but the
target is RANK (Dominant/Subordinate) and the leave-one-group-out unit is the CAGE
(animals interact only with cagemates, so holding out a whole cage is the leakage-free
test). Reuses the shared, target-agnostic helpers in ../srnn_analysis.py.

  (A) RECORDING-LEVEL breathing rate -> rank (leakage-free: each recording is scored by
      the fold that HELD ITS CAGE OUT).  breathing_Hz = switch_rate * fs / 2.
      For valence breathing rate was a perfect confound; for rank we simply report whether
      it carries any rank signal.
  (B) POOLED LATENT PCA through ONE model, colored by rank (descriptive geometry).
  (C) RANK SIGNAL BEYOND BREATHING RATE: leave-one-CAGE-out decode of rate / full latent /
      rate-removed latent.
  (D) PERMUTATION TEST (LOCO-by-cage), labels shuffled at the recording level.
  (E) LDA projection (leave-one-cage-out, leakage-aware).

NOTE (per-timestep): (B)-(F) no longer average each window's 1500 latents to a single
  vector. They use EVERY timestep's latent (~1500 per window), repeating each window's
  rank / cage / recording id across its timesteps so rows stay aligned. --latent_stride>1
  subsamples timesteps if the per-timestep decode/permutation is too slow.

CPU only, inference only. Run via hipergator/analyze_rank.slurm, or:
    python respiration/rank/analyze_rank.py --config respiration/rank/config_rank_hpg.yaml
"""
import os, sys, glob, re, argparse
import numpy as np
import yaml
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # respiration/ (for srnn_analysis)
from srnn_analysis import (load_ckpt, per_window_switch_rate, per_window_mean_latent,
                           per_timestep_latent, expand_to_timesteps,
                           logo_decode, permutation_test, logo_lda_scores)
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from joblib import Parallel, delayed

DOM, SUB = "#b2182b", "#2166ac"          # Dominant / Subordinate colors
RANK_HI, RANK_LO = "Dominant", "Subordinate"   # label 1 / label 0


def n_jobs():
    """Cores for the permutation fan-out; honors the SLURM allocation (-1 = all off-cluster)."""
    return int(os.environ.get("SLURM_CPUS_PER_TASK", "0")) or -1


def cage_of(names):
    """Per-recording cage = the leading index of the sCAGE_ANIMAL token (e.g. 's1_2' -> '1')."""
    return np.array([re.search(r"s(\d+)_\d+", str(s)).group(1) for s in names])


def rec_lda_auc(lat, ts_rank, cage_ts, rid_ts, recs_u):
    """ROC-AUC of per-recording mean (leave-one-cage-out) LDA score vs rank. Top-level (not a
    closure) so joblib can memmap the big constant arrays across permutation workers instead
    of re-pickling them per task."""
    s = logo_lda_scores(lat, ts_rank, cage_ts)            # per-timestep LDA scores
    rm = np.array([s[rid_ts == r].mean() for r in recs_u])
    rr = np.array([ts_rank[rid_ts == r][0] for r in recs_u])
    return roc_auc_score(rr, rm)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="respiration/rank/config_rank_hpg.yaml")
    ap.add_argument("--split", choices=["cage"], default="cage",
                    help="only 'cage' (leave-one-cage-out) is supported: cagemates interact "
                         "only with each other, so a held-out recording whose cagemate is in "
                         "training leaks; holding out a whole cage is the leakage-free test.")
    ap.add_argument("--pca_fold", type=int, default=0,
                    help="which fold's trained model to use for the pooled PCA")
    ap.add_argument("--n_perm", type=int, default=1000,
                    help="label shuffles for the permutation test (part D); 0 to skip")
    ap.add_argument("--latent_stride", type=int, default=1,
                    help="keep every Nth timestep of the per-window latent trajectory "
                         "(1 = every latent, ~1500/window). Raise it to subsample timesteps "
                         "if the per-timestep decode/permutation is too slow.")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    paths = cfg["paths"]; h = int(cfg["model"]["hidden_shape"])
    fs = int(cfg["preprocess"]["target_fs"])
    device = torch.device("cpu")

    ckpts = sorted(glob.glob(os.path.join(paths["save_dir"], f"resp_srnn_{args.split}_h{h}_fold*.pt")),
                   key=lambda p: int(re.search(r"fold(\d+)", p).group(1)))
    print(f"analyzing split='{args.split}': found {len(ckpts)} fold checkpoints\n")
    if not ckpts:
        print(f"No 'resp_srnn_{args.split}_h{h}_fold*.pt' checkpoints in {paths['save_dir']}.")
        print("Train them first: sbatch hipergator/rank_job_loco.slurm")
        return

    # ----------------- (A) leakage-free recording-level breathing rate -> rank -----------------
    print("=" * 72)
    print(f"(A) RECORDING-LEVEL breathing rate  (leakage-free: each recording scored by the")
    print(f"    fold that held ITS CAGE out; split='{args.split}').  breathing_Hz = switch_rate*fs/2  (fs={fs})")
    print("=" * 72)
    rows = []  # (name, rank, n_windows, mean_switch_rate, breathing_hz)
    for ckpt in ckpts:
        ck, model, rnninfer = load_ckpt(ckpt, h, device)
        y = torch.tensor(ck["y_test"], dtype=torch.float32, device=device)
        sr = per_window_switch_rate(model, rnninfer, y, device)        # per-window
        names = np.asarray(ck["recording_names"]); rid = np.asarray(ck["recording_id_test"])
        rk = np.asarray(ck.get("rank_test", ck.get("target_test")))
        for r in np.unique(rid):                                       # group held-out windows by recording
            m = rid == r
            rows.append((str(names[r]), int(rk[m][0]), int(m.sum()),
                         float(sr[m].mean()), float(sr[m].mean() * fs / 2)))

    rows.sort(key=lambda r: (-r[1], r[0]))                # Dominant first, then by name
    print(f"\n  {'recording':30s} {'rank':12s} {'n_win':>5s} {'switch_rate':>12s} {'breathing_Hz':>13s}")
    for name, rk, nw, srm, hz in rows:
        print(f"  {name:30s} {RANK_HI if rk==1 else RANK_LO:12s} {nw:5d} {srm:12.3f} {hz:13.2f}")

    rk_arr = np.array([r[1] for r in rows]); hz_arr = np.array([r[4] for r in rows])
    if len(np.unique(rk_arr)) == 2:
        dom, sub = hz_arr[rk_arr == 1], hz_arr[rk_arr == 0]
        auc = roc_auc_score(rk_arr, hz_arr)
        print(f"\n  Dominant    breathing_Hz: mean={dom.mean():.2f}  range=[{dom.min():.2f}, {dom.max():.2f}]")
        print(f"  Subordinate breathing_Hz: mean={sub.mean():.2f}  range=[{sub.min():.2f}, {sub.max():.2f}]")
        print(f"  ROC-AUC (breathing rate -> rank, n={len(rows)} recordings) = {auc:.3f}")
        print("  (0.5 = rate carries no rank signal; n is small -> suggestive pilot, not significance)")

    # ----------------- (B) pooled latent PCA through ONE model, colored by rank -----------------
    print("\n" + "=" * 72)
    print(f"(B) POOLED LATENT PCA -- all recordings through the fold-{args.pca_fold} model")
    print("=" * 72)
    obs = np.load(os.path.join(paths["out_dir"], "observations.npy"))
    meta = np.load(os.path.join(paths["out_dir"], "meta.npz"), allow_pickle=True)
    rank = np.asarray(meta["rank"]); rid_all = np.asarray(meta["recording_id"])
    rec_names = np.asarray(meta["recording_names"])
    cage_all = cage_of(rec_names)[rid_all]                 # per-window cage id
    y_all = torch.tensor(obs, dtype=torch.float32, device=device)

    pca_ckpt = os.path.join(paths["save_dir"], f"resp_srnn_{args.split}_h{h}_fold{args.pca_fold}.pt")
    _, model, rnninfer = load_ckpt(pca_ckpt, h, device)
    # EVERY timestep's latent (not the per-window mean): ~1500 latents per window. The
    # per-window rank / cage groups / recording ids are repeated per timestep so the rows
    # stay aligned. All of (B)-(F) below run on these per-timestep rows.
    lat, T_kept = per_timestep_latent(rnninfer, y_all, stride=args.latent_stride)  # (n_win*T_kept, h)
    rank_ts = expand_to_timesteps(rank, T_kept)
    cage_ts = expand_to_timesteps(cage_all, T_kept)
    rid_ts = expand_to_timesteps(rid_all, T_kept)
    print(f"  per-timestep latents: {lat.shape[0]} rows "
          f"({len(rank)} windows x {T_kept} kept timesteps, stride={args.latent_stride})")
    pcs = PCA(n_components=2).fit_transform(lat)

    os.makedirs(paths["plot_dir"], exist_ok=True)
    # 220k points would make an unreadable, huge PNG -> fit PCA on all, scatter a random subset
    rng_plot = np.random.RandomState(0)
    fig, ax = plt.subplots(figsize=(6, 5))
    for v, c, lbl in [(1, DOM, RANK_HI), (0, SUB, RANK_LO)]:
        m = np.where(rank_ts == v)[0]
        if len(m) > 15000:
            m = rng_plot.choice(m, 15000, replace=False)
        ax.scatter(pcs[m, 0], pcs[m, 1], s=4, alpha=0.25, color=c, label=lbl, linewidth=0)
    ax.set_title(f"Pooled latent PCA (per-timestep h, fold-{args.pca_fold} model)\n"
                 "one coordinate system; color = rank")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    leg = ax.legend()
    for lh in getattr(leg, "legend_handles", getattr(leg, "legendHandles", [])):
        lh.set_alpha(1)
    out_png = os.path.join(paths["plot_dir"], "pooled_latent_pca_by_rank.png")
    plt.tight_layout(); plt.savefig(out_png, dpi=150); plt.close()
    print(f"  saved -> {out_png}")
    print(f"  {len(rank)} windows: {int((rank==1).sum())} Dominant / {int((rank==0).sum())} Subordinate")

    # ----------------- (C) is there rank signal BEYOND breathing rate? -----------------
    if len(np.unique(rank)) == 2:
        print("\n" + "=" * 72)
        print("(C) RANK SIGNAL BEYOND BREATHING RATE")
        print(f"    leave-one-cage-out decoding; PCA/latent model = fold-{args.pca_fold}")
        print("=" * 72)
        swr_all = per_window_switch_rate(model, rnninfer, y_all, device)   # per-window rate proxy
        swr_ts = expand_to_timesteps(swr_all, T_kept)                      # broadcast to timesteps
        n_cage = len(np.unique(cage_all))
        print(f"  {'feature':34s} {'LOCO(cage)':>12s}")
        for label, X, resid in [("1. rate only (switch rate)", swr_ts, None),
                                 ("2. full latent h", lat, None),
                                 ("3. latent h, RATE regressed out", lat, swr_ts)]:
            a = logo_decode(X, rank_ts, cage_ts, residualize=resid)
            print(f"  {label:34s} {a:12.3f}")
        print(f"  (chance=0.5; LOCO uses only n={n_cage} cage groups -> coarse, treat as pilot)")
        print("  if the rate-removed latent (row 3) collapses to ~chance, any apparent signal")
        print("  was rate/identity leakage, not rank that generalizes across cages.")

        # ----------------- (D) permutation test (LOCO-by-cage) -----------------
        if args.n_perm > 0:
            print("\n" + "=" * 72)
            print(f"(D) PERMUTATION TEST  (LOCO-by-cage; n_perm={args.n_perm}, "
                  "labels shuffled at recording level)")
            print("=" * 72)
            tests = [("full latent h", lat, None),
                     ("latent h, rate removed", lat, swr_ts)]
            fig, axes = plt.subplots(1, len(tests), figsize=(5 * len(tests), 4), squeeze=False)
            for ax, (name, X, resid) in zip(axes[0], tests):
                obs_acc, null, p = permutation_test(X, rank_ts, rid_ts, cage_ts,
                                                    args.n_perm, seed=131, residualize=resid)
                ax.hist(null, bins=30, color="#bbbbbb", edgecolor="white")
                ax.axvline(0.5, color="k", ls=":", lw=1, label="chance = 0.5")
                ax.axvline(obs_acc, color="#d62728", lw=2.2,
                           label=f"observed = {obs_acc:.3f}\np = {p:.4f}")
                ax.set_title(f"{name}\n(LOCO-by-cage)")
                ax.set_xlabel("balanced accuracy"); ax.set_ylabel("# permutations")
                ax.legend(fontsize=8, loc="upper right")
                print(f"  {name:24s} observed={obs_acc:.3f}  null mean={null.mean():.3f} "
                      f"sd={null.std():.3f}  p={p:.4f}")
            out_png = os.path.join(paths["plot_dir"], f"permutation_test_{args.split}.png")
            plt.tight_layout(); plt.savefig(out_png, dpi=150); plt.close()
            print(f"  saved -> {out_png}")
            print(f"  p = fraction of {args.n_perm} label-shuffles with balanced-acc >= observed")
            print(f"  (n={len(np.unique(cage_all))} cage groups -> coarse; p is honest about that)")

        # ----------------- (E) supervised LDA projection (leave-one-cage-out) -----------------
        print("\n" + "=" * 72)
        print("(E) LDA PROJECTION  (supervised separating axis, LOCO-by-cage, leakage-aware)")
        print("=" * 72)
        ld = logo_lda_scores(lat, rank_ts, cage_ts)           # one score per TIMESTEP
        fig, (axh, axr) = plt.subplots(1, 2, figsize=(11, 4))
        for v, c, lbl in [(1, DOM, RANK_HI), (0, SUB, RANK_LO)]:
            axh.hist(ld[rank_ts == v], bins=40, alpha=0.6, color=c, label=lbl,
                     edgecolor="none", density=True)
        axh.set_title("Per-timestep LDA score by rank\n(held-out projection)")
        axh.set_xlabel("LDA discriminant score"); axh.set_ylabel("density"); axh.legend(fontsize=8)
        recs = np.unique(rid_ts)
        rec_mean = np.array([ld[rid_ts == r].mean() for r in recs])
        rec_rank = np.array([rank_ts[rid_ts == r][0] for r in recs])
        rng = np.random.RandomState(0)
        for v, c in [(1, DOM), (0, SUB)]:
            m = rec_rank == v
            x = v + (rng.rand(m.sum()) - 0.5) * 0.25
            axr.scatter(x, rec_mean[m], s=55, color=c, edgecolor="k", linewidth=0.4, alpha=0.85)
        axr.set_xticks([0, 1]); axr.set_xticklabels(["Subordinate", "Dominant"])
        axr.set_title("Per-recording mean LDA score\n(one point per recording)")
        axr.set_ylabel("mean LDA discriminant score")
        out_png = os.path.join(paths["plot_dir"], f"lda_projection_{args.split}.png")
        plt.tight_layout(); plt.savefig(out_png, dpi=150); plt.close()
        print(f"  saved -> {out_png}")
        print("  left = per-window score by rank; right = per-recording means (the unit that")
        print("  matters). Projection is leave-one-cage-out, so separation here is not circular.")

        # ----------------- (F) permutation test on the per-recording LDA separation -----------------
        # (E)'s right panel is the eyeball test; this asks whether that separation beats chance.
        # Statistic = ROC-AUC of per-recording mean LDA score vs rank. Null: shuffle rank at the
        # RECORDING level, REFIT the leave-one-cage-out LDA on the shuffled labels (so the whole
        # supervised pipeline is re-run -> not circular), recompute the per-recording AUC.
        if args.n_perm > 0:
            recs_u = np.unique(rid_ts)
            rec_rank_true = np.array([rank_ts[rid_ts == r][0] for r in recs_u])

            obs_auc = rec_lda_auc(lat, rank_ts, cage_ts, rid_ts, recs_u)
            rng = np.random.RandomState(131)
            # pre-shuffle serially (deterministic), then fan the LDA refits out across cores
            perms = []
            for _ in range(args.n_perm):
                mp = dict(zip(recs_u, rng.permutation(rec_rank_true)))
                perms.append(np.array([mp[r] for r in rid_ts]))
            null = np.asarray(Parallel(n_jobs=n_jobs())(
                delayed(rec_lda_auc)(lat, yp, cage_ts, rid_ts, recs_u) for yp in perms))
            p = (1.0 + np.sum(null >= obs_auc)) / (1.0 + args.n_perm)
            print("\n" + "=" * 72)
            print("(F) PERMUTATION TEST on the per-recording LDA separation (LOCO-by-cage)")
            print("=" * 72)
            print(f"  statistic = ROC-AUC(per-recording mean LDA score, rank), n={len(recs_u)} recordings")
            print(f"  observed AUC = {obs_auc:.3f}   null mean = {null.mean():.3f} sd = {null.std():.3f}   p = {p:.4f}")
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.hist(null, bins=30, color="#bbbbbb", edgecolor="white")
            ax.axvline(0.5, color="k", ls=":", lw=1, label="chance = 0.5")
            ax.axvline(obs_auc, color="#d62728", lw=2.2, label=f"observed = {obs_auc:.3f}\np = {p:.4f}")
            ax.set_title("Per-recording LDA separation\n(LOCO-by-cage; rank shuffled at recording level)")
            ax.set_xlabel("ROC-AUC (per-recording mean LDA vs rank)"); ax.set_ylabel("# permutations")
            ax.legend(fontsize=8, loc="upper left")
            out_png = os.path.join(paths["plot_dir"], f"lda_permutation_{args.split}.png")
            plt.tight_layout(); plt.savefig(out_png, dpi=150); plt.close()
            print(f"  saved -> {out_png}")
            print(f"  (n={len(recs_u)} recordings -> coarse; the AUC takes few distinct values, so p is granular)")


if __name__ == "__main__":
    main()
