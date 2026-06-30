"""
srnn_analysis.py  (created by Claude)

SHARED, target-agnostic analysis helpers used by BOTH analyze_valence.py and
analyze_rank.py. None of these care whether the 2-class label is valence or rank or
how the held-out groups are defined (subject for LOSO, cage for LOCO) -- the caller
passes the label array and the grouping array. Single source of truth so the two
experiment analyses can't drift.
"""
import os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root
from SRNN import model_srnn, inference_network, train as srnn_train
from sklearn.metrics import balanced_accuracy_score
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from joblib import Parallel, delayed


def n_jobs():
    """How many cores to fan the permutation decodes across. Honors the SLURM allocation
    (SLURM_CPUS_PER_TASK) so we never oversubscribe a shared node; -1 (all cores joblib
    sees) off-cluster. Bump --cpus-per-task in the .slurm file to make the perm test faster."""
    return int(os.environ.get("SLURM_CPUS_PER_TASK", "0")) or -1


def load_ckpt(path, h, device):
    """Reload a saved SRNN fold checkpoint: returns (ckpt_dict, model, rnninfer) ready for
    inference. The dict carries the held-out test data + labels saved at train time."""
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
    """EVERY inferred latent, NOT averaged over the window.

    `per_window_mean_latent` collapses the T (=1500) timesteps of each window to a single
    mean vector. This instead keeps each timestep, returning the latents flattened to
    (n_windows * T_kept, h) -- ~1500 latents per window -- so the analyses see the within-
    window trajectory rather than its average. Pair the flattened latent with
    `expand_to_timesteps()` on any per-window metadata (labels, groups, recording id) so
    rows stay aligned. `stride > 1` keeps every `stride`-th timestep to cut compute while
    preserving coverage. Returns (latent_flat (n*T_kept, h), T_kept)."""
    rnninfer.eval()
    with torch.no_grad():
        _, _, mean_out = rnninfer(y)                       # (n, T, h)
    lat = mean_out.cpu().numpy()
    if stride > 1:
        lat = lat[:, ::stride, :]
    n, T_kept, h = lat.shape
    return lat.reshape(n * T_kept, h), T_kept              # row order: win0 t0..tT, win1 t0..


def per_timestep_state(model, rnninfer, y, device, stride=1):
    """Per-timestep discrete SRNN state (argmax of the posterior), shape (n, T_kept).
    These are the model's own inhale/exhale phases -- used to check whether the latent
    micro-state clusters subdivide the breathing cycle."""
    X = torch.zeros_like(y)
    _, _, pos, *_ = srnn_train.eval_(model, rnninfer, X, y, device)
    states = np.exp(pos).argmax(-1)                        # (n, T)
    return states[:, ::stride] if stride > 1 else states


def expand_to_timesteps(arr, T_kept):
    """Repeat each per-window value T_kept times so per-window metadata (labels, groups,
    recording ids, a per-window covariate) lines up with the flattened per-timestep
    latents from `per_timestep_latent`. Row order matches that flattening."""
    return np.repeat(np.asarray(arr), T_kept, axis=0)


def logo_decode(X, y, groups, residualize=None):
    """Leave-one-group-out balanced accuracy for a 2-class target.
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


def permutation_test(X, lab_window, rid_window, groups, n_perm, seed, residualize=None):
    """Significance of the leave-one-group-out decode against a structure-respecting null.

    The 2-class label is a RECORDING-level property (each recording is all one class). So
    the null shuffles labels at the RECORDING level (preserving class balance and the
    window/group structure) and recomputes the same grouped balanced accuracy. Shuffling
    per-window instead would give a falsely narrow null. Returns (observed, null_array, p).
    """
    obs = logo_decode(X, lab_window, groups, residualize=residualize)
    recs = np.unique(rid_window)
    rec_lab = np.array([lab_window[rid_window == r][0] for r in recs])   # one label per recording
    rng = np.random.RandomState(seed)
    # Pre-generate every shuffled label vector SERIALLY (cheap; keeps the RNG stream
    # deterministic no matter how many cores run), then fan the expensive LOGO decodes out
    # across cores. joblib auto-memmaps the big constant arrays (X, groups) so they're
    # shared, not re-pickled per task. Reproducible AND parallel.
    perms = []
    for _ in range(n_perm):
        mapping = dict(zip(recs, rng.permutation(rec_lab)))
        perms.append(np.array([mapping[r] for r in rid_window]))
    null = np.asarray(Parallel(n_jobs=n_jobs())(
        delayed(logo_decode)(X, y_perm, groups, residualize=residualize) for y_perm in perms))
    p = (1.0 + np.sum(null >= obs)) / (1.0 + n_perm)                     # one-sided, +1 smoothing
    return obs, null, p


def logo_lda_scores(X, y, groups):
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
