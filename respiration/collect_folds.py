"""
collect_folds.py  (created by Claude)

Leakage-free valence analysis across the 4 leave-one-recording-out folds. For each
fold the held-out recording was never in that model's training set, so its latent /
switch features are unbiased. We pool all held-out windows and test whether
POSITIVE (RI1) vs NEGATIVE (RI2) recordings differ in:
  (a) the continuous latent h_t   (per-window mean latent)
  (b) inferred switch statistics  (state occupancy fractions + switch rate)
Decoding uses Leave-One-Recording-Out CV on the decoder too (group = recording), so
neither the SRNN nor the decoder ever sees the held-out recording.

NOTE: with only 4 recordings (2 pos / 2 neg) this is a PILOT signal, not a p-value.

Run after all 4 folds have trained:
    python respiration/collect_folds.py --config respiration/config_respiration_hpg.yaml
"""
import os, sys, argparse
import numpy as np
import yaml
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from SRNN import model_srnn, inference_network, train as srnn_train
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict
from sklearn.metrics import balanced_accuracy_score, accuracy_score


def window_features(model, rnninfer, y, device, num_tv):
    """Return per-window (mean_latent[h], state_occupancy[num_tv], switch_rate[1])."""
    X = torch.zeros_like(y)
    _, _, pos, *_ = srnn_train.eval_(model, rnninfer, X, y, device)      # pos: (n,T,K) log-post
    states = np.exp(pos).argmax(-1)                                       # (n,T)
    rnninfer.eval()
    with torch.no_grad():
        _, _, mean_out = rnninfer(y)
    lat = mean_out.cpu().numpy().mean(axis=1)                            # (n,h)
    occ = np.stack([(states == k).mean(1) for k in range(num_tv)], axis=1)  # (n,K)
    swr = (np.diff(states, axis=1) != 0).mean(1, keepdims=True)          # (n,1) switch rate
    return lat, occ, swr, states


def decode(name, X, val, grp):
    if len(np.unique(val)) < 2:
        print(f"  [{name}] only one valence present — skip"); return
    preds = cross_val_predict(LogisticRegression(max_iter=2000), X, val, groups=grp, cv=LeaveOneGroupOut())
    print(f"  [{name:16s}] LORO balanced-acc={balanced_accuracy_score(val,preds):.3f} "
          f"acc={accuracy_score(val,preds):.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="respiration/config_respiration_hpg.yaml")
    ap.add_argument("--folds", type=int, default=4)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    paths = cfg["paths"]; h = int(cfg["model"]["hidden_shape"])
    device = torch.device("cpu")

    LAT, OCC, SWR, VAL, GRP = [], [], [], [], []
    for f in range(args.folds):
        ckpt = os.path.join(paths["save_dir"], f"resp_srnn_recording_h{h}_fold{f}.pt")
        if not os.path.exists(ckpt):
            print(f"[skip] missing {ckpt}"); continue
        ck = torch.load(ckpt, weights_only=False, map_location=device)
        K = ck["num_tv"]
        model = model_srnn.Model(ck["D"], K, h, ck["neural_private_shape"]).to(device)
        rnninfer = inference_network.RNNInfer(ck["D"], h).to(device)
        model.load_state_dict(ck["model_state_dict"]); rnninfer.load_state_dict(ck["rnninfer_state_dict"])
        y = torch.tensor(ck["y_test"], dtype=torch.float32, device=device)
        lat, occ, swr, _ = window_features(model, rnninfer, y, device, K)
        LAT.append(lat); OCC.append(occ); SWR.append(swr)
        VAL.append(np.asarray(ck["valence_test"])); GRP.append(np.asarray(ck["recording_id_test"]))
        print(f"fold {f}: held-out windows={len(swr)} valence={int(np.asarray(ck['valence_test'])[0])} "
              f"mean switch-rate={swr.mean():.3f}")

    if not LAT:
        print("No checkpoints found — train the 4 folds first."); return
    lat = np.concatenate(LAT); occ = np.concatenate(OCC); swr = np.concatenate(SWR)
    val = np.concatenate(VAL); grp = np.concatenate(GRP)
    switch_feats = np.concatenate([occ, swr], axis=1)

    print("\n" + "=" * 64)
    print(f"pooled windows={len(val)}  pos={int((val==1).sum())} neg={int((val==0).sum())}")
    # descriptive: switch statistics by valence
    for v, lbl in [(1, "positive(RI1)"), (0, "negative(RI2)")]:
        m = val == v
        print(f"  {lbl}: switch-rate={swr[m].mean():.3f}  "
              f"state-occupancy={np.round(occ[m].mean(0), 3).tolist()}")
    print("\nLEAKAGE-FREE valence decoding (leave-one-recording-out on the decoder):")
    decode("latent h", lat, val, grp)
    decode("switch stats", switch_feats, val, grp)
    decode("latent+switch", np.concatenate([lat, switch_feats], axis=1), val, grp)
    print("(chance=0.5; n=4 recordings -> treat as a pilot signal, not significance)")


if __name__ == "__main__":
    main()
