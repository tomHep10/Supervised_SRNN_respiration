"""
plot_respiration.py  (created by Claude)

Respiration-appropriate analysis of a trained checkpoint:
  1. resp_recon.png  : true vs reconstructed respiration trace (line overlay),
                       background shaded by inferred discrete state.
  2. states.png      : true vs inferred state sequence across test windows.
  3. latent_pca.png  : PCA of the inferred latent h, colored by VALENCE (the
                       hypothesis test) and by inferred state.
  4. prints a quick valence-decoding accuracy from the per-window mean latent.

Run from the repo root:
    conda run -n sleap-new python respiration/plot_respiration.py \
        --config respiration/config_respiration.yaml
"""
import os, sys, argparse
import numpy as np
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from SRNN import model_srnn, inference_network, train as srnn_train
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="respiration/config_respiration.yaml")
    ap.add_argument("--fold", type=int, default=None)
    ap.add_argument("--split", choices=["subject", "window"], default=None)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    paths = cfg["paths"]
    fold = args.fold if args.fold is not None else int(cfg["train"]["fold"])
    split_mode = args.split if args.split is not None else cfg["train"].get("split_mode", "subject")
    h = int(cfg["model"]["hidden_shape"])
    ckpt_path = os.path.join(paths["save_dir"], f"resp_srnn_{split_mode}_h{h}_fold{fold}.pt")

    device = torch.device("cpu")
    ck = torch.load(ckpt_path, weights_only=False, map_location=device)
    num_tv, D = ck["num_tv"], ck["D"]

    y_test = torch.tensor(ck["y_test"], dtype=torch.float32, device=device)
    X_test = torch.zeros_like(y_test)
    label_test = ck["label_test"][:, :, 0].astype(int)     # (n,T)
    valence = ck["valence_test"]                            # (n,)

    model = model_srnn.Model(D, num_tv, h, ck["neural_private_shape"]).to(device)
    rnninfer = inference_network.RNNInfer(D, h).to(device)
    model.load_state_dict(ck["model_state_dict"]); rnninfer.load_state_dict(ck["rnninfer_state_dict"])

    # eval: reconstruction + state posterior + latent
    y_pred, _, pos_test, *_ = srnn_train.eval_(model, rnninfer, X_test, y_test, device)
    inferred_states = np.exp(pos_test).argmax(-1)          # (n,T)
    rnninfer.eval()
    with torch.no_grad():
        _, _, mean_out = rnninfer(y_test)
    latent = mean_out.cpu().numpy()                        # (n,T,h)

    os.makedirs(paths["plot_dir"], exist_ok=True)
    n, T = label_test.shape
    state_colors = ['#999999', '#0072B2', '#D55E00', '#009E73', '#CC79A7', '#F0E442']

    # ── 1) respiration reconstruction (best window), shaded by inferred state ──
    mse_w = ((y_pred - ck["y_test"]) ** 2).mean(axis=(1, 2))
    w = int(np.argmin(mse_w))
    t = np.arange(T)
    fig, ax = plt.subplots(figsize=(8, 2.6))
    for k in range(num_tv):                                # state shading
        ax.fill_between(t, -3, 3, where=(inferred_states[w] == k),
                        color=state_colors[k % len(state_colors)], alpha=0.18, step="mid")
    ax.plot(t, ck["y_test"][w, :, 0], 'k', lw=1.2, label="true resp")
    ax.plot(t, y_pred[w, :, 0], 'r', lw=1.0, alpha=0.8, label="reconstructed")
    ax.set_title(f"Respiration reconstruction (test window {w}, shading = inferred state)")
    ax.set_xlabel("time (samples)"); ax.set_ylabel("z-resp"); ax.set_ylim(-3, 3); ax.legend(loc="upper right", fontsize=8)
    plt.tight_layout(); plt.savefig(os.path.join(paths["plot_dir"], "resp_recon.png"), dpi=150); plt.close()

    # ── 2) true vs inferred states ──
    cmap = mcolors.ListedColormap(state_colors[:num_tv])
    fig, axes = plt.subplots(2, 1, figsize=(7, 3), sharex=True)
    axes[0].imshow(label_test, aspect="auto", cmap=cmap, vmin=0, vmax=num_tv-1, interpolation="nearest")
    axes[0].set_title("True behavior states"); axes[0].set_ylabel("window"); axes[0].set_xticks([])
    axes[1].imshow(inferred_states, aspect="auto", cmap=cmap, vmin=0, vmax=num_tv-1, interpolation="nearest")
    axes[1].set_title("Inferred states"); axes[1].set_ylabel("window"); axes[1].set_xlabel("time (samples)")
    plt.tight_layout(); plt.savefig(os.path.join(paths["plot_dir"], "states.png"), dpi=150); plt.close()

    # ── 3) PCA of latent, colored by valence and by inferred state ──
    H = latent.reshape(-1, latent.shape[-1])               # (n*T, h)
    val_pt = np.repeat(valence, T)                         # (n*T,)
    st_pt  = inferred_states.reshape(-1)
    if H.shape[0] > 20000:                                 # subsample for the scatter
        sel = np.random.RandomState(0).choice(H.shape[0], 20000, replace=False)
        H, val_pt, st_pt = H[sel], val_pt[sel], st_pt[sel]
    pcs = PCA(n_components=2).fit_transform(H)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for v, c, lbl in [(1, "#1b7837", "positive (RI1)"), (0, "#762a83", "negative (RI2)")]:
        m = val_pt == v
        axes[0].scatter(pcs[m, 0], pcs[m, 1], s=3, alpha=0.3, color=c, label=lbl)
    axes[0].set_title("Latent PCA — colored by VALENCE"); axes[0].legend(markerscale=3, fontsize=8)
    axes[0].set_xlabel("PC1"); axes[0].set_ylabel("PC2")
    for k in range(num_tv):
        m = st_pt == k
        axes[1].scatter(pcs[m, 0], pcs[m, 1], s=3, alpha=0.3, color=state_colors[k % len(state_colors)], label=f"state {k}")
    axes[1].set_title("Latent PCA — colored by inferred state"); axes[1].legend(markerscale=3, fontsize=8)
    axes[1].set_xlabel("PC1"); axes[1].set_ylabel("PC2")
    plt.tight_layout(); plt.savefig(os.path.join(paths["plot_dir"], "latent_pca.png"), dpi=150); plt.close()

    # ── 4) valence decoding from per-window mean latent ──
    Xw = latent.mean(axis=1)                               # (n, h)  per-window mean latent
    msg = "valence decoding: need both classes in the test set (skipped)"
    if len(np.unique(valence)) == 2 and n >= 10:
        clf = LogisticRegression(max_iter=1000)
        try:
            acc = cross_val_score(clf, Xw, valence, cv=min(5, n // 2), scoring="balanced_accuracy")
            msg = f"valence decoding (logreg on mean latent, CV bal-acc): {acc.mean():.3f} +/- {acc.std():.3f}"
        except Exception as e:
            msg = f"valence decoding skipped ({e})"

    print(f"saved figures to {paths['plot_dir']}")
    print(f"test windows={n}  T={T}  num_tv={num_tv}  valence(pos/neg)={int((valence==1).sum())}/{int((valence==0).sum())}")
    print(msg)
    print("NOTE: with leave-one-subject-out a held-out subject contributes recordings of "
          "BOTH valences, but per-fold valence decoding is still noisy; the leakage-free "
          "valence numbers come from pooling latents across folds in analyze_valence.py.")


if __name__ == "__main__":
    main()
