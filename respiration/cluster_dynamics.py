"""
cluster_dynamics.py  (created by Claude)

WITHIN-WINDOW LATENT MICRO-STATE CLUSTERING -- a separate, descriptive companion to
analyze_valence.py / analyze_rank.py.

Those analyses (now) use EVERY timestep's latent (~1500 per 30 s window) instead of the
per-window mean. This script asks the next question: *within* a window, does the latent
trajectory visit a small number of recurring states, and what are the dynamics between
them? It pools every timestep's latent from all windows (through ONE trained fold model),
clusters them with K-Means (an unsupervised partition of the latent cloud into "micro-
states"), and then characterizes the within-window dynamics of those micro-states:

  * cluster geometry        -- centroids in latent space + a PCA view colored by cluster
  * how many states (k)     -- optional silhouette / inertia sweep to pick k
  * occupancy               -- fraction of time spent in each micro-state, OVERALL and
                               split by the target label (does positive vs negative /
                               Dominant vs Subordinate use the states differently?)
  * dwell time              -- mean consecutive run length per micro-state (within windows)
  * transitions             -- micro-state -> micro-state transition matrix (within windows;
                               window boundaries are NOT counted as transitions)
  * vs the SRNN's own state -- contingency of K-Means micro-states against the model's
                               discrete inhale/exhale states (do the clusters subdivide the
                               breathing cycle, or cut across it?)

This is DESCRIPTIVE geometry/dynamics through one model, not a leakage-free decoding claim
(the clean tests live in the analyze_* scripts). K-Means is the natural first choice here:
the latent cloud is low-dimensional (h=8), we want a hard partition into a handful of
recurring states to read dwell/transition structure off of, and it is cheap on ~220k
points. (A sweep + silhouette guards against forcing structure that isn't there.)

CPU only, inference only. Examples:
    # valence experiment, auto-pick k by silhouette over 2..8
    python respiration/cluster_dynamics.py \
        --config respiration/valence/config_respiration_hpg.yaml --target valence --k 0
    # rank experiment, fixed k=4
    python respiration/cluster_dynamics.py \
        --config respiration/rank/config_rank_hpg.yaml --target rank --split cage --k 4
"""
import os, sys, glob, re, argparse
import numpy as np
import yaml
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # respiration/ (for srnn_analysis)
from srnn_analysis import (load_ckpt, per_timestep_latent, per_timestep_state,
                           expand_to_timesteps)
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# per-target display: meta key, {label_value: (color, name)}
TARGETS = {
    "valence": ("valence", {1: ("#1b7837", "positive (RI1)"), 0: ("#762a83", "negative (RI2)")}),
    "rank":    ("rank",    {1: ("#b2182b", "Dominant"),       0: ("#2166ac", "Subordinate")}),
}


def pick_k(Xz, k_min, k_max, seed, sample_size=5000):
    """Sweep k and score each by silhouette (on a subsample -- silhouette is O(n^2)) and
    inertia. Returns (best_k, list_of (k, silhouette, inertia)). Best = max silhouette."""
    rng = np.random.RandomState(seed)
    sub = rng.choice(len(Xz), min(sample_size, len(Xz)), replace=False)
    rows = []
    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, n_init=5, random_state=seed).fit(Xz)
        sil = silhouette_score(Xz[sub], km.labels_[sub])
        rows.append((k, float(sil), float(km.inertia_)))
        print(f"    k={k:2d}  silhouette={sil:+.3f}  inertia={km.inertia_:.0f}")
    best_k = max(rows, key=lambda r: r[1])[0]
    return best_k, rows


def dwell_and_transitions(traj, k):
    """traj: (n_windows, T) integer micro-state per timestep. Returns:
      dwell  -- mean consecutive run length (in timesteps) per micro-state, length k
      trans  -- (k,k) row-normalized within-window transition matrix (boundaries excluded)
    Window boundaries are never counted as a transition (each window handled separately)."""
    run_lens = [[] for _ in range(k)]
    trans = np.zeros((k, k), dtype=float)
    for row in traj:
        # transitions within this window
        a, b = row[:-1], row[1:]
        for i, j in zip(a, b):
            trans[i, j] += 1
        # consecutive-run lengths within this window
        change = np.where(np.diff(row) != 0)[0]
        bounds = np.concatenate(([-1], change, [len(row) - 1]))
        for s, e in zip(bounds[:-1], bounds[1:]):
            run_lens[row[e]].append(e - s)
    dwell = np.array([np.mean(r) if r else 0.0 for r in run_lens])
    rowsum = trans.sum(1, keepdims=True)
    trans_norm = np.divide(trans, rowsum, out=np.zeros_like(trans), where=rowsum > 0)
    return dwell, trans_norm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="respiration/valence/config_respiration_hpg.yaml")
    ap.add_argument("--target", choices=["valence", "rank"], default="valence",
                    help="which per-window label to split occupancy by (meta key)")
    ap.add_argument("--split", default="subject",
                    help="checkpoint split tag to load the fold model from "
                         "(valence: 'subject'; rank: 'cage')")
    ap.add_argument("--pca_fold", type=int, default=0,
                    help="which fold's trained model provides the latents")
    ap.add_argument("--k", type=int, default=0,
                    help="number of micro-states (K-Means clusters). 0 = auto-pick by "
                         "silhouette over [--k_min, --k_max].")
    ap.add_argument("--k_min", type=int, default=2)
    ap.add_argument("--k_max", type=int, default=8)
    ap.add_argument("--latent_stride", type=int, default=1,
                    help="keep every Nth timestep (1 = every latent, ~1500/window)")
    ap.add_argument("--seed", type=int, default=131)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    paths = cfg["paths"]; h = int(cfg["model"]["hidden_shape"])
    fs = int(cfg["preprocess"]["target_fs"])
    device = torch.device("cpu")
    meta_key, label_style = TARGETS[args.target]

    # ----------------- load model + data -----------------
    pca_ckpt = os.path.join(paths["save_dir"], f"resp_srnn_{args.split}_h{h}_fold{args.pca_fold}.pt")
    if not os.path.exists(pca_ckpt):
        print(f"No checkpoint {pca_ckpt}. Train the folds first, or pass --split / --pca_fold.")
        cands = sorted(glob.glob(os.path.join(paths["save_dir"], f"resp_srnn_*_h{h}_fold*.pt")))
        if cands:
            print("  available checkpoints:")
            for c in cands:
                print("   ", os.path.basename(c))
        return
    obs = np.load(os.path.join(paths["out_dir"], "observations.npy"))
    meta = np.load(os.path.join(paths["out_dir"], "meta.npz"), allow_pickle=True)
    label = np.asarray(meta[meta_key]); rid_all = np.asarray(meta["recording_id"])
    y_all = torch.tensor(obs, dtype=torch.float32, device=device)
    _, model, rnninfer = load_ckpt(pca_ckpt, h, device)

    print("=" * 72)
    print(f"WITHIN-WINDOW MICRO-STATE CLUSTERING  (target={args.target}, "
          f"fold-{args.pca_fold} {args.split} model)")
    print("=" * 72)

    # ----------------- every timestep's latent (NOT the per-window mean) -----------------
    lat, T_kept = per_timestep_latent(rnninfer, y_all, stride=args.latent_stride)  # (n_win*T, h)
    n_win = len(label)
    label_ts = expand_to_timesteps(label, T_kept)
    srnn_state = per_timestep_state(model, rnninfer, y_all, device, stride=args.latent_stride).reshape(-1)
    print(f"  {lat.shape[0]} per-timestep latents  ({n_win} windows x {T_kept} timesteps, "
          f"stride={args.latent_stride}); h={h}")

    Xz = StandardScaler().fit_transform(lat)   # cluster in standardized latent space

    # ----------------- choose k -----------------
    os.makedirs(paths["plot_dir"], exist_ok=True)
    if args.k and args.k > 0:
        k = args.k
        print(f"\n  using fixed k = {k}")
    else:
        print(f"\n  sweeping k in [{args.k_min}, {args.k_max}] (silhouette on a 5k subsample):")
        k, sweep = pick_k(Xz, args.k_min, args.k_max, args.seed)
        print(f"  -> best k by silhouette = {k}")
        ks = [r[0] for r in sweep]
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4))
        a1.plot(ks, [r[1] for r in sweep], "o-", color="#1b7837")
        a1.axvline(k, color="k", ls=":"); a1.set_xlabel("k"); a1.set_ylabel("silhouette")
        a1.set_title("silhouette vs k (higher = better)")
        a2.plot(ks, [r[2] for r in sweep], "o-", color="#762a83")
        a2.axvline(k, color="k", ls=":"); a2.set_xlabel("k"); a2.set_ylabel("inertia")
        a2.set_title("inertia vs k (elbow)")
        out = os.path.join(paths["plot_dir"], f"cluster_k_selection_{args.target}.png")
        plt.tight_layout(); plt.savefig(out, dpi=150); plt.close()
        print(f"  saved -> {out}")

    # ----------------- final K-Means -----------------
    km = KMeans(n_clusters=k, n_init=10, random_state=args.seed).fit(Xz)
    cl = km.labels_                                   # (n_win*T,) micro-state per timestep
    traj = cl.reshape(n_win, T_kept)                  # within-window micro-state trajectory

    # ----------------- occupancy: overall + by target label -----------------
    print("\n" + "-" * 72)
    print("OCCUPANCY  (fraction of timesteps in each micro-state)")
    print("-" * 72)
    overall = np.array([(cl == c).mean() for c in range(k)])
    vals = sorted(label_style.keys(), reverse=True)   # [1, 0]
    occ_by = {v: np.array([(cl[label_ts == v] == c).mean() for c in range(k)]) for v in vals}
    hdr = "  cluster   overall " + " ".join(f"{label_style[v][1][:10]:>11s}" for v in vals)
    print(hdr)
    for c in range(k):
        line = f"  {c:^7d}  {overall[c]:7.3f} " + " ".join(f"{occ_by[v][c]:11.3f}" for v in vals)
        print(line)

    # ----------------- dwell time + transitions -----------------
    dwell, trans = dwell_and_transitions(traj, k)
    print("\n" + "-" * 72)
    print("DWELL TIME  (mean consecutive run length within a window)")
    print("-" * 72)
    for c in range(k):
        print(f"  cluster {c}: {dwell[c]:7.1f} timesteps  = {dwell[c]/fs:6.3f} s")

    # ----------------- micro-state vs SRNN inhale/exhale state -----------------
    n_srnn = int(srnn_state.max()) + 1
    cont = np.zeros((k, n_srnn), dtype=float)
    for c in range(k):
        m = cl == c
        for s in range(n_srnn):
            cont[c, s] = (srnn_state[m] == s).mean() if m.any() else 0.0
    print("\n" + "-" * 72)
    print("MICRO-STATE vs SRNN DISCRETE STATE  (row = P(SRNN state | micro-state))")
    print("-" * 72)
    print("  cluster " + " ".join(f"{'srnn'+str(s):>8s}" for s in range(n_srnn)))
    for c in range(k):
        print(f"  {c:^7d} " + " ".join(f"{cont[c, s]:8.3f}" for s in range(n_srnn)))

    # ======================= FIGURES =======================
    cmap = plt.get_cmap("tab10")
    cluster_colors = [cmap(c % 10) for c in range(k)]

    # (1) PCA of latents colored by micro-state
    pcs = PCA(n_components=2).fit_transform(lat)
    rng_plot = np.random.RandomState(0)
    fig, ax = plt.subplots(figsize=(6, 5))
    for c in range(k):
        idx = np.where(cl == c)[0]
        if len(idx) > 8000:
            idx = rng_plot.choice(idx, 8000, replace=False)
        ax.scatter(pcs[idx, 0], pcs[idx, 1], s=4, alpha=0.3, color=cluster_colors[c],
                   label=f"state {c}", linewidth=0)
    ax.set_title(f"Per-timestep latent PCA, colored by micro-state (k={k})")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    leg = ax.legend(markerscale=3, fontsize=8)
    for lh in getattr(leg, "legend_handles", getattr(leg, "legendHandles", [])):
        lh.set_alpha(1)
    out = os.path.join(paths["plot_dir"], f"cluster_pca_{args.target}.png")
    plt.tight_layout(); plt.savefig(out, dpi=150); plt.close()
    print(f"\n  saved -> {out}")

    # (2) centroids heatmap (k x h, standardized space) + occupancy-by-target + transitions
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    im0 = axes[0].imshow(km.cluster_centers_, aspect="auto", cmap="coolwarm")
    axes[0].set_title("micro-state centroids\n(standardized latent dims)")
    axes[0].set_xlabel("latent dim"); axes[0].set_ylabel("micro-state")
    axes[0].set_yticks(range(k))
    fig.colorbar(im0, ax=axes[0], fraction=0.046)

    width = 0.8 / len(vals)
    xc = np.arange(k)
    for i, v in enumerate(vals):
        axes[1].bar(xc + i * width, occ_by[v], width, color=label_style[v][0],
                    label=label_style[v][1], edgecolor="white")
    axes[1].set_xticks(xc + width * (len(vals) - 1) / 2); axes[1].set_xticklabels(range(k))
    axes[1].set_xlabel("micro-state"); axes[1].set_ylabel("occupancy fraction")
    axes[1].set_title(f"occupancy by {args.target}"); axes[1].legend(fontsize=8)

    im2 = axes[2].imshow(trans, cmap="viridis", vmin=0, vmax=1)
    axes[2].set_title("within-window transition matrix\nP(next | current)")
    axes[2].set_xlabel("to micro-state"); axes[2].set_ylabel("from micro-state")
    axes[2].set_xticks(range(k)); axes[2].set_yticks(range(k))
    for i in range(k):
        for j in range(k):
            axes[2].text(j, i, f"{trans[i, j]:.2f}", ha="center", va="center",
                         color="white" if trans[i, j] < 0.6 else "black", fontsize=7)
    fig.colorbar(im2, ax=axes[2], fraction=0.046)
    out = os.path.join(paths["plot_dir"], f"cluster_summary_{args.target}.png")
    plt.tight_layout(); plt.savefig(out, dpi=150); plt.close()
    print(f"  saved -> {out}")

    # (3) example within-window micro-state trajectories (a few windows of each label)
    fig, axes = plt.subplots(len(vals), 1, figsize=(11, 2.2 * len(vals)), squeeze=False)
    t_axis = np.arange(T_kept) * args.latent_stride / fs
    rng_ex = np.random.RandomState(1)
    for ax, v in zip(axes[:, 0], vals):
        wins = np.where(label == v)[0]
        pick = rng_ex.choice(wins, min(4, len(wins)), replace=False)
        for off, w in enumerate(pick):
            ax.step(t_axis, traj[w] + off * (k + 1), where="post", lw=0.8,
                    color=label_style[v][0])
        ax.set_title(f"{label_style[v][1]}: example within-window micro-state trajectories",
                     fontsize=9)
        ax.set_ylabel("state (offset/window)"); ax.set_yticks([])
    axes[-1, 0].set_xlabel("time within window (s)")
    out = os.path.join(paths["plot_dir"], f"cluster_trajectories_{args.target}.png")
    plt.tight_layout(); plt.savefig(out, dpi=150); plt.close()
    print(f"  saved -> {out}")

    print("\n  DESCRIPTIVE only (one model, pooled timesteps) -- the leakage-free decoding")
    print("  tests live in analyze_valence.py / analyze_rank.py.")


if __name__ == "__main__":
    main()
