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
    null = np.empty(n_perm)
    for i in range(n_perm):
        mapping = dict(zip(recs, rng.permutation(rec_lab)))
        y_perm = np.array([mapping[r] for r in rid_window])
        null[i] = logo_decode(X, y_perm, groups, residualize=residualize)
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
