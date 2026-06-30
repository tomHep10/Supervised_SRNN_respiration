"""
analyze_valence.py  (created by Claude)

Two leakage-aware analyses the per-fold plots / the old cross-fold decoder can't give:

  (A) RECORDING-LEVEL breathing-rate / switch-rate test (leakage-free).
      Each recording is scored by the fold that HELD IT OUT, so the model never
      trained on it. The two discovered states are the inhale/exhale phases of the
      breathing cycle, so one breath ~= 2 state switches and:
            breathing_Hz = switch_rate * fs / 2
      Per-window switch rate is aggregated to ONE value per recording (the
      independent unit -- no window pseudo-replication) and positive (RI1) vs
      negative (RI2) are compared with ROC-AUC.

  (B) POOLED LATENT PCA in ONE coordinate system.
      Per-fold PCAs only show the single held-out recording (one valence), and
      latents are NOT comparable across separately-trained folds (rotation/
      permutation/sign/scale non-identifiability). So we run ALL windows of ALL
      recordings through a SINGLE model and PCA the latents, colored by valence.
      This is descriptive geometry (the one model trained on most of these
      recordings) -- it answers "do valences separate in the latent?", it is NOT a
      leakage-free decoding claim. The clean leakage-free number is (A).

  NOTE (per-timestep): (B)-(E) no longer average each window's 1500 latents to a single
      vector. They use EVERY timestep's latent (~1500 per window), repeating each window's
      valence / subject / recording id across its timesteps so rows stay aligned. This
      keeps the within-window trajectory instead of collapsing it. --latent_stride>1
      subsamples timesteps if the per-timestep decode/permutation is too slow.

CPU only, inference only. Run via hipergator/analyze_job.slurm, or:
    python respiration/analyze_valence.py --config respiration/config_respiration_hpg.yaml
"""
import os, sys, glob, re, argparse
import numpy as np
import yaml
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # repo root (file now in respiration/valence/)
from SRNN import model_srnn, inference_network, train as srnn_train
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from joblib import Parallel, delayed


def n_jobs():
    """Cores for the permutation fan-out; honors the SLURM allocation so we don't
    oversubscribe (-1 = all cores off-cluster). Raise --cpus-per-task to go faster."""
    return int(os.environ.get("SLURM_CPUS_PER_TASK", "0")) or -1


def load_ckpt(path, h, device):
    """
    Loads a previously saved model checkpoint and prepares the model and inference network for use.

    What is a checkpoint?
    ---------------------
    In deep learning, a "checkpoint" is a file that saves all the important information about a model at a certain point during or after training. This allows you to later reload the model exactly as it was, without retraining from scratch. The checkpoint usually stores things like learned weights, model configuration, and sometimes the optimizer state.

    Line-by-line explanation:
    -------------------------
    - ck = torch.load(path, weights_only=False, map_location=device)
        # Loads the checkpoint file from disk ("path" specifies the file). This file contains saved data, typically a Python dictionary.
        # "weights_only=False" means it loads everything saved (not just the model's raw numbers/weights).
        # "map_location=device" will place the loaded data on the specified device (e.g., 'cpu' or 'cuda' for GPU).
        # After this, "ck" is a dictionary with keys and values for the model and training state.

    - model = model_srnn.Model(ck["D"], ck["num_tv"], h, ck["neural_private_shape"]).to(device)
        # Here, we create a new instance of the model architecture (called "model_srnn.Model").
        # ck["D"]: The input dimension of the data (number of features per timepoint).
        # ck["num_tv"]: How many "states" the model has for switching (often discrete behavioral or neural states).
        # h: The size of the hidden layer we want (passed as an argument to this function).
        # ck["neural_private_shape"]: Additional model parameter for internal representation (specific to architecture).
        # .to(device) moves the model to the specified hardware device (CPU or GPU).

    - rnninfer = inference_network.RNNInfer(ck["D"], h).to(device)
        # Builds a related "inference network" that helps the model make predictions about the hidden states.
        # Takes the input dimension and hidden size as arguments, then moves it to the same hardware device.

    - model.load_state_dict(ck["model_state_dict"])
        # Loads the saved weights/parameters into the model -- these are the numbers the model has learned, 
        # so it behaves identically to when it was trained.

    - rnninfer.load_state_dict(ck["rnninfer_state_dict"])
        # Loads the saved weights/parameters for the inference network.

    - return ck, model, rnninfer
        # Returns:
        #   ck:        The full checkpoint dictionary, in case you need other saved info.
        #   model:     The reconstituted model, ready for inference or evaluation.
        #   rnninfer:  The reconstituted inference network, also ready for use.
    """
    ck = torch.load(path, weights_only=False, map_location=device)
    model = model_srnn.Model(ck["D"], ck["num_tv"], h, ck["neural_private_shape"]).to(device)
    rnninfer = inference_network.RNNInfer(ck["D"], h).to(device)
    model.load_state_dict(ck["model_state_dict"])
    rnninfer.load_state_dict(ck["rnninfer_state_dict"])
    return ck, model, rnninfer


def per_window_switch_rate(model, rnninfer, y, device):
    X = torch.zeros_like(y)
    _, _, pos, *_ = srnn_train.eval_(model, rnninfer, X, y, device)
    states = np.exp(pos).argmax(-1)                       # (n,T)
    return (np.diff(states, axis=1) != 0).mean(1)         # (n,) per-window switch rate


def per_window_mean_latent(rnninfer, y):
    rnninfer.eval()
    with torch.no_grad():
        _, _, mean_out = rnninfer(y)
    return mean_out.cpu().numpy().mean(axis=1)            # (n,h)


def per_timestep_latent(rnninfer, y, stride=1):
    """EVERY inferred latent, NOT averaged over the window. Where per_window_mean_latent
    collapses a window's T (=1500) timesteps to one mean vector, this keeps each timestep,
    returning latents flattened to (n_windows * T_kept, h) -- ~1500 latents per window. Pair
    it with expand_to_timesteps() on any per-window metadata so rows stay aligned.
    stride>1 keeps every stride-th timestep to cut compute. Returns (latent_flat, T_kept)."""
    rnninfer.eval()
    with torch.no_grad():
        _, _, mean_out = rnninfer(y)                       # (n, T, h)
    lat = mean_out.cpu().numpy()
    if stride > 1:
        lat = lat[:, ::stride, :]
    n, T_kept, h = lat.shape
    return lat.reshape(n * T_kept, h), T_kept             # row order: win0 t0..tT, win1 t0..


def expand_to_timesteps(arr, T_kept):
    """Repeat each per-window value T_kept times so per-window metadata (labels, groups,
    recording ids, a per-window covariate) lines up with the flattened per-timestep latents
    from per_timestep_latent(). Row order matches that flattening."""
    return np.repeat(np.asarray(arr), T_kept, axis=0)


def logo_decode(X, y, groups, residualize=None):
    """Leave-one-group-out balanced accuracy for valence.
    If `residualize` (a 1-D per-window covariate, e.g. switch/breathing rate) is
    given, it is linearly regressed OUT of X within each split -- fit on the train
    groups, applied to the held-out one -- so the decode uses only the part of
    the latent NOT explained by rate. Standardization is likewise fit on train only.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    yt, yp = [], []
    for tr, te in LeaveOneGroupOut().split(X, y, groups):
        Xtr, Xte = X[tr].copy(), X[te].copy()
        if residualize is not None:
            r = np.asarray(residualize, dtype=float).reshape(-1, 1)
            lr = LinearRegression().fit(r[tr], Xtr)       # X ~ rate, train only
            Xtr = Xtr - lr.predict(r[tr])
            Xte = Xte - lr.predict(r[te])
        sc = StandardScaler().fit(Xtr)
        clf = LogisticRegression(max_iter=2000).fit(sc.transform(Xtr), y[tr])
        yt.append(y[te]); yp.append(clf.predict(sc.transform(Xte)))
    return balanced_accuracy_score(np.concatenate(yt), np.concatenate(yp))


def permutation_test(X, val_window, rid_window, groups, n_perm, seed, residualize=None):
    """Significance of the LOSO decode against a structure-respecting null.

    Valence is a RECORDING-level property (each recording is all-positive or
    all-negative; a subject spans both). So the null shuffles valence labels at the
    RECORDING level (preserving class balance and the window/subject grouping) and
    recomputes the same LOSO balanced accuracy. Shuffling per-window instead would
    give a falsely narrow null. Returns (observed, null_array, p_value).
    """
    obs = logo_decode(X, val_window, groups, residualize=residualize)
    recs = np.unique(rid_window)
    rec_val = np.array([val_window[rid_window == r][0] for r in recs])   # one label per recording
    rng = np.random.RandomState(seed)
    # Pre-generate every shuffled label vector SERIALLY (cheap; keeps the RNG deterministic
    # regardless of core count), then fan the expensive LOGO decodes out across cores.
    # joblib auto-memmaps the big constant arrays so they're shared, not re-pickled.
    perms = []
    for _ in range(n_perm):
        mapping = dict(zip(recs, rng.permutation(rec_val)))
        perms.append(np.array([mapping[r] for r in rid_window]))
    null = np.asarray(Parallel(n_jobs=n_jobs())(
        delayed(logo_decode)(X, y_perm, groups, residualize=residualize) for y_perm in perms))
    p = (1.0 + np.sum(null >= obs)) / (1.0 + n_perm)                     # one-sided, +1 smoothing
    return obs, null, p


def loso_lda_scores(X, y, groups):
    """Honest 1-D LDA discriminant score per window. For 2 classes LDA has exactly one
    discriminant axis. Fitting and plotting on the same data is circular (LDA finds a
    separating direction even in noise), so for each held-out group we fit LDA (after
    standardizing) on the OTHER groups and project only the held-out windows."""
    scores = np.full(len(y), np.nan)
    for tr, te in LeaveOneGroupOut().split(X, y, groups):
        sc = StandardScaler().fit(X[tr])
        lda = LinearDiscriminantAnalysis(n_components=1).fit(sc.transform(X[tr]), y[tr])
        scores[te] = lda.transform(sc.transform(X[te]))[:, 0]
    return scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="respiration/valence/config_respiration_hpg.yaml")
    ap.add_argument("--split", choices=["subject"], default="subject",
                    help="which trained folds to analyze. Only 'subject' (leave-one-subject-out) "
                         "is supported: every subject spans both valences, so holding out a whole "
                         "subject is leakage-free across individuals. Leave-one-recording-out was "
                         "removed -- it leaks (a held-out recording's subject is still in training).")
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
        if args.split == "subject":
            print("Train them first: sbatch hipergator/respiration_job_loso.slurm")
        return

    # ----------------- (A) leakage-free recording-level breathing rate -----------------
    print("=" * 72)
    print(f"(A) RECORDING-LEVEL breathing rate  (leakage-free: each recording is scored by")
    print(f"    the fold that held it out; split='{args.split}').  breathing_Hz = switch_rate*fs/2  (fs={fs})")
    print("=" * 72)
    rows = []  # (name, valence, n_windows, mean_switch_rate, breathing_hz)
    for ckpt in ckpts:
        f = int(re.search(r"fold(\d+)", ckpt).group(1))
        ck, model, rnninfer = load_ckpt(ckpt, h, device)
        y = torch.tensor(ck["y_test"], dtype=torch.float32, device=device)
        sr = per_window_switch_rate(model, rnninfer, y, device)        # per-window
        names = np.asarray(ck["recording_names"]); rid = np.asarray(ck["recording_id_test"])
        val = np.asarray(ck["valence_test"])
        # group this fold's held-out windows BY RECORDING (a subject fold holds out
        # several recordings of BOTH valences).
        for r in np.unique(rid):
            m = rid == r
            rows.append((str(names[r]), int(val[m][0]), int(m.sum()),
                         float(sr[m].mean()), float(sr[m].mean() * fs / 2)))

    rows.sort(key=lambda r: (-r[1], r[0]))                # positives first, then by name
    print(f"\n  {'recording':12s} {'valence':9s} {'n_win':>5s} {'switch_rate':>12s} {'breathing_Hz':>13s}")
    for name, val, nw, srm, hz in rows:
        print(f"  {name:12s} {'pos(RI1)' if val==1 else 'neg(RI2)':9s} {nw:5d} {srm:12.3f} {hz:13.2f}")

    val_arr = np.array([r[1] for r in rows]); hz_arr = np.array([r[4] for r in rows])
    if len(np.unique(val_arr)) == 2:
        pos, neg = hz_arr[val_arr == 1], hz_arr[val_arr == 0]
        auc = roc_auc_score(val_arr, hz_arr)
        print(f"\n  positive breathing_Hz: mean={pos.mean():.2f}  range=[{pos.min():.2f}, {pos.max():.2f}]")
        print(f"  negative breathing_Hz: mean={neg.mean():.2f}  range=[{neg.min():.2f}, {neg.max():.2f}]")
        print(f"  separation gap (min_pos - max_neg) = {pos.min() - neg.max():+.2f} Hz  "
              f"(>0 => perfectly separable)")
        print(f"  ROC-AUC (breathing rate -> valence, n={len(rows)} recordings) = {auc:.3f}")
        print("  (n is small -> treat as a suggestive pilot, not significance)")

    # ----------------- (B) pooled latent PCA through ONE model -----------------
    print("\n" + "=" * 72)
    print(f"(B) POOLED LATENT PCA -- all recordings through the fold-{args.pca_fold} model")
    print("=" * 72)
    obs = np.load(os.path.join(paths["out_dir"], "observations.npy"))
    meta = np.load(os.path.join(paths["out_dir"], "meta.npz"), allow_pickle=True)
    valence = np.asarray(meta["valence"]); rid_all = np.asarray(meta["recording_id"])
    rec_names = np.asarray(meta["recording_names"])
    # subject = FULL sX_Y token (trailing number = global individual id); same token in
    # RI1 and RI2 is the same animal. (Grouping by leading sX alone = session/pair, wrong.)
    subj_of_rec = np.array([re.search(r"(s\d+_\d+)", str(s)).group(1) for s in rec_names])
    subj_all = subj_of_rec[rid_all]                       # per-window subject id
    y_all = torch.tensor(obs, dtype=torch.float32, device=device)

    pca_ckpt = os.path.join(paths["save_dir"], f"resp_srnn_{args.split}_h{h}_fold{args.pca_fold}.pt")
    _, model, rnninfer = load_ckpt(pca_ckpt, h, device)
    # EVERY timestep's latent (not the per-window mean): ~1500 latents per window. The
    # per-window labels / subject groups / recording ids are repeated per timestep so the
    # rows stay aligned. All of (B)-(E) below run on these per-timestep rows.
    lat, T_kept = per_timestep_latent(rnninfer, y_all, stride=args.latent_stride)  # (n_win*T_kept, h)
    valence_ts = expand_to_timesteps(valence, T_kept)
    subj_ts = expand_to_timesteps(subj_all, T_kept)
    rid_ts = expand_to_timesteps(rid_all, T_kept)
    print(f"  per-timestep latents: {lat.shape[0]} rows "
          f"({len(valence)} windows x {T_kept} kept timesteps, stride={args.latent_stride})")
    pcs = PCA(n_components=2).fit_transform(lat)

    os.makedirs(paths["plot_dir"], exist_ok=True)
    # 220k points would make an unreadable, huge PNG -> fit PCA on all, scatter a random subset
    rng_plot = np.random.RandomState(0)
    fig, ax = plt.subplots(figsize=(6, 5))
    for v, c, lbl in [(1, "#1b7837", "positive (RI1)"), (0, "#762a83", "negative (RI2)")]:
        m = np.where(valence_ts == v)[0]
        if len(m) > 15000:
            m = rng_plot.choice(m, 15000, replace=False)
        ax.scatter(pcs[m, 0], pcs[m, 1], s=4, alpha=0.25, color=c, label=lbl, linewidth=0)
    ax.set_title(f"Pooled latent PCA (per-timestep h, fold-{args.pca_fold} model)\n"
                 "one coordinate system; color = valence")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    leg = ax.legend()
    for lh in getattr(leg, "legend_handles", getattr(leg, "legendHandles", [])):
        lh.set_alpha(1)
    out_png = os.path.join(paths["plot_dir"], "pooled_latent_pca_by_valence.png")
    plt.tight_layout(); plt.savefig(out_png, dpi=150); plt.close()
    print(f"  saved -> {out_png}")
    print(f"  {len(valence)} windows: {int((valence==1).sum())} pos / {int((valence==0).sum())} neg")

    # ----------------- (C) is there valence signal BEYOND breathing rate? -----------------
    # Everything clean so far is rate (states = inhale/exhale -> switch rate = rate).
    # The NEXT experiment will be rate-matched, so the real question is whether the
    # latent carries valence AFTER rate is removed. Three decodes (rate only / full
    # latent / latent with rate regressed out), all under leave-one-SUBJECT-out (LOSO)
    # grouping: every subject spans both valences, so holding out a whole subject tests
    # whether valence generalizes ACROSS individuals -> the real, leakage-free test.
    # (Leave-one-recording-out was removed: a held-out recording's subject is still in
    # train, so individual respiration signatures leak and inflate the number.)
    if len(np.unique(valence)) == 2:
        print("\n" + "=" * 72)
        print("(C) VALENCE SIGNAL BEYOND BREATHING RATE")
        print(f"    leave-one-subject-out decoding; PCA/latent model = fold-{args.pca_fold}")
        print("=" * 72)
        swr_all = per_window_switch_rate(model, rnninfer, y_all, device)   # per-window rate proxy
        swr_ts = expand_to_timesteps(swr_all, T_kept)                      # broadcast to timesteps
        n_subj = len(np.unique(subj_all))
        print(f"  {'feature':34s} {'LOSO(subject)':>15s}")
        for label, X, resid in [("1. rate only (switch rate)", swr_ts, None),
                                 ("2. full latent h", lat, None),
                                 ("3. latent h, RATE regressed out", lat, swr_ts)]:
            a_subj = logo_decode(X, valence_ts, subj_ts, residualize=resid)
            print(f"  {label:34s} {a_subj:15.3f}")
        print(f"  (chance=0.5; LOSO uses only n={n_subj} subject groups -> coarse, treat as pilot)")
        print("  if the rate-removed latent (row 3) collapses to ~chance, the apparent signal")
        print("  was individual-respiration leakage, not valence that generalizes across animals.")

        # ----------------- (D) permutation test + plot (LOSO-by-subject) -----------------
        if args.n_perm > 0:
            print("\n" + "=" * 72)
            print(f"(D) PERMUTATION TEST  (LOSO-by-subject; n_perm={args.n_perm}, "
                  "labels shuffled at recording level)")
            print("=" * 72)
            tests = [("full latent h", lat, None),
                     ("latent h, rate removed", lat, swr_ts)]
            fig, axes = plt.subplots(1, len(tests), figsize=(5 * len(tests), 4), squeeze=False)
            for ax, (name, X, resid) in zip(axes[0], tests):
                obs, null, p = permutation_test(X, valence_ts, rid_ts, subj_ts,
                                                args.n_perm, seed=131, residualize=resid)
                ax.hist(null, bins=30, color="#bbbbbb", edgecolor="white")
                ax.axvline(0.5, color="k", ls=":", lw=1, label="chance = 0.5")
                ax.axvline(obs, color="#d62728", lw=2.2,
                           label=f"observed = {obs:.3f}\np = {p:.4f}")
                ax.set_title(f"{name}\n(LOSO-by-subject)")
                ax.set_xlabel("balanced accuracy"); ax.set_ylabel("# permutations")
                ax.legend(fontsize=8, loc="upper right")
                print(f"  {name:24s} observed={obs:.3f}  null mean={null.mean():.3f} "
                      f"sd={null.std():.3f}  p={p:.4f}")
            out_png = os.path.join(paths["plot_dir"], f"permutation_test_{args.split}.png")
            plt.tight_layout(); plt.savefig(out_png, dpi=150); plt.close()
            print(f"  saved -> {out_png}")
            print(f"  p = fraction of {args.n_perm} label-shuffles with balanced-acc >= observed")
            print(f"  (n={len(np.unique(subj_all))} subject groups -> coarse; p is honest about that)")

        # ----------------- (E) supervised LDA projection (the view PCA can't give) -----------------
        # PCA shows max-variance axes (valence isn't one). LDA finds the axis that best
        # separates valence. Projection is leave-one-subject-out so it is NOT circular.
        print("\n" + "=" * 72)
        print("(E) LDA PROJECTION  (supervised separating axis, LOSO-by-subject, leakage-aware)")
        print("=" * 72)
        ld = loso_lda_scores(lat, valence_ts, subj_ts)        # one score per TIMESTEP
        green, purple = "#1b7837", "#762a83"
        fig, (axh, axr) = plt.subplots(1, 2, figsize=(11, 4))
        # left: per-timestep LDA score distribution by valence
        for v, c, lbl in [(1, green, "positive (RI1)"), (0, purple, "negative (RI2)")]:
            axh.hist(ld[valence_ts == v], bins=40, alpha=0.6, color=c, label=lbl,
                     edgecolor="none", density=True)
        axh.set_title("Per-timestep LDA score by valence\n(held-out projection)")
        axh.set_xlabel("LDA discriminant score"); axh.set_ylabel("density"); axh.legend(fontsize=8)
        # right: per-recording mean LDA score (the independent unit), jittered by valence
        recs = np.unique(rid_ts)
        rec_mean = np.array([ld[rid_ts == r].mean() for r in recs])
        rec_val = np.array([valence_ts[rid_ts == r][0] for r in recs])
        rng = np.random.RandomState(0)
        for v, c in [(1, green), (0, purple)]:
            m = rec_val == v
            x = v + (rng.rand(m.sum()) - 0.5) * 0.25
            axr.scatter(x, rec_mean[m], s=55, color=c, edgecolor="k", linewidth=0.4, alpha=0.85)
        axr.set_xticks([0, 1]); axr.set_xticklabels(["negative\n(RI2)", "positive\n(RI1)"])
        axr.set_title("Per-recording mean LDA score\n(one point per recording)")
        axr.set_ylabel("mean LDA discriminant score")
        out_png = os.path.join(paths["plot_dir"], f"lda_projection_{args.split}.png")
        plt.tight_layout(); plt.savefig(out_png, dpi=150); plt.close()
        print(f"  saved -> {out_png}")
        print("  left = per-window score by valence; right = per-recording means (the unit that")
        print("  matters). Projection is leave-one-subject-out, so separation here is not circular.")


if __name__ == "__main__":
    main()
