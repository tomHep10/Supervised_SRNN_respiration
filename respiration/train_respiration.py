"""
train_respiration.py  (created by Claude)

Driver that trains the SRNN on the prepared respiration windows. It REUSES the
SRNN/ package unchanged (model, inference network, loss, baum-welch) but uses a
lean training loop instead of SRNN.train.train_ , because train_ stores the test
posterior for *every* epoch -> fine for the 10-trial sim, but multiple GB with
hundreds of real windows. We import SRNN.train.eval_ for evaluation only.

Run from the repo root:
    conda run -n sleap-new python respiration/train_respiration.py \
        --config respiration/config_respiration.yaml
CLI overrides: --fold --epochs --num_tv --hidden_shape --coef_cross
"""
import os, sys, json, time, argparse
import numpy as np
import yaml
import torch
from torch.utils.data import TensorDataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root
from SRNN import model_srnn, inference_network, train as srnn_train
from SRNN import loss_function, initialization


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="respiration/config_respiration.yaml")
    ap.add_argument("--fold", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--num_tv", type=int, default=None)
    ap.add_argument("--hidden_shape", type=int, default=None)
    ap.add_argument("--coef_cross", type=float, default=None)
    ap.add_argument("--split", choices=["recording", "window"], default=None)
    return ap.parse_args()


def split_indices(meta, mode, fold):
    """Return (train_idx, test_idx) over windows."""
    rid = meta["recording_id"]
    n = rid.shape[0]
    if mode == "recording":
        rec_ids = np.unique(rid)
        test_rec = rec_ids[fold % len(rec_ids)]
        test_idx = np.where(rid == test_rec)[0]
        train_idx = np.where(rid != test_rec)[0]
    else:  # random window KFold
        from sklearn.model_selection import KFold
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        train_idx, test_idx = list(kf.split(np.arange(n)))[fold % 5]
    return train_idx, test_idx


def main():
    args = parse_args()
    cfg = yaml.safe_load(open(args.config))
    paths, M, Tr = cfg["paths"], cfg["model"], cfg["train"]

    fold        = args.fold        if args.fold        is not None else int(Tr["fold"])
    epochs      = args.epochs      if args.epochs      is not None else int(Tr["epochs"])
    num_tv      = args.num_tv      if args.num_tv      is not None else int(M["num_tv"])
    hidden_shape= args.hidden_shape if args.hidden_shape is not None else int(M["hidden_shape"])
    coef_cross  = args.coef_cross  if args.coef_cross  is not None else float(Tr["coef_cross"])
    batch_size  = int(Tr["batch_size"]); eval_every = int(Tr.get("eval_every", 25))
    lr = float(Tr["lr"]); split_mode = args.split if args.split is not None else Tr.get("split_mode", "recording")

    seed = int(cfg["system"].get("seed", 131))
    np.random.seed(seed); torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}  fold={fold}  epochs={epochs}  num_tv={num_tv} "
          f"hidden={hidden_shape}  coef_cross={coef_cross}  split={split_mode}")

    # --- load prepared data ---
    obs = np.load(os.path.join(paths["out_dir"], "observations.npy"))   # (n,T,D)
    lab = np.load(os.path.join(paths["out_dir"], "labels.npy"))          # (n,T,1)
    meta = dict(np.load(os.path.join(paths["out_dir"], "meta.npz"), allow_pickle=True))
    D = obs.shape[2]
    train_idx, test_idx = split_indices(meta, split_mode, fold)
    print(f"data {obs.shape}  D={D}  train_windows={len(train_idx)} test_windows={len(test_idx)}")

    dtype = torch.float32
    y_train = torch.tensor(obs[train_idx], dtype=dtype, device=device)
    y_test  = torch.tensor(obs[test_idx],  dtype=dtype, device=device)
    X_train = torch.zeros_like(y_train); X_test = torch.zeros_like(y_test)  # input-free model
    lab_train = lab[train_idx, :, 0]      # (ntr, T)

    pp = torch.tensor(initialization.one_hot(lab_train, num_tv), dtype=dtype, device=device)

    # --- build model (SRNN/ untouched) ---
    model = model_srnn.Model(D, num_tv, hidden_shape, int(M["neural_private_shape"])).to(device)
    rnninfer = inference_network.RNNInfer(D, hidden_shape).to(device)
    opt  = torch.optim.Adam(model.parameters(), lr=lr)
    opt_r= torch.optim.Adam(rnninfer.parameters(), lr=lr)
    sch  = torch.optim.lr_scheduler.StepLR(opt,  step_size=2000, gamma=0.8)
    sch_r= torch.optim.lr_scheduler.StepLR(opt_r, step_size=2000, gamma=0.8)

    loader = DataLoader(TensorDataset(X_train, y_train, pp),
                        batch_size=batch_size, shuffle=True, drop_last=False)

    os.makedirs(paths["save_dir"], exist_ok=True)
    ckpt_path = os.path.join(paths["save_dir"], f"resp_srnn_{split_mode}_h{hidden_shape}_fold{fold}.pt")
    prog_path = os.path.join(paths["save_dir"], f"progress_{split_mode}_fold{fold}.csv")
    with open(prog_path, "w") as pf:
        pf.write("epoch,loss,test_mse,elapsed_s,eta_s\n")
    loss_curve = np.full(epochs, np.nan)
    start = time.time()

    for epoch in range(epochs):
        model.train(); rnninfer.train()
        running = 0.0
        for Xb, yb, ppb in loader:
            opt.zero_grad(); opt_r.zero_grad()
            infer_dist, h, _ = rnninfer(yb)
            prob_ini, ps, ph, py, gamma, delta, fwp, bwp = model(Xb, yb, h, h, device)
            t1, t2 = loss_function.get_loss(gamma, delta, prob_ini, ps, ph, py)
            post = model.get_posterior_lk(fwp, bwp)
            ce = loss_function.get_cross_entropy(post, ppb)
            loss = -(t1.mean() + t2.mean() + coef_cross * ce + infer_dist.entropy().mean())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(rnninfer.parameters(), 1.0)
            opt.step(); opt_r.step()
            running += float(loss.detach())
        sch.step(); sch_r.step()
        loss_curve[epoch] = running

        if epoch % eval_every == 0 or epoch == epochs - 1:
            y_pred, _, pos_test, *_ = srnn_train.eval_(model, rnninfer, X_test, y_test, device)
            mse = float(np.mean((y_pred - obs[test_idx]) ** 2))
            el = time.time() - start
            eta = el / (epoch + 1) * (epochs - epoch - 1)
            print(f"epoch {epoch:4d}/{epochs} | loss {running:12.1f} | test MSE {mse:.4f} "
                  f"| {el:5.0f}s elapsed, ~{eta:5.0f}s left", flush=True)
            with open(prog_path, "a") as pf:
                pf.write(f"{epoch},{running:.1f},{mse:.6f},{el:.0f},{eta:.0f}\n")
            torch.save({
                "config": cfg, "fold": fold, "num_tv": num_tv, "hidden_shape": hidden_shape,
                "neural_private_shape": int(M["neural_private_shape"]), "D": D,
                "coef_cross": coef_cross, "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "rnninfer_state_dict": rnninfer.state_dict(),
                "y_test": obs[test_idx], "label_test": lab[test_idx],
                "valence_test": meta["valence"][test_idx],
                "recording_id_test": meta["recording_id"][test_idx],
                "recording_names": meta["recording_names"],
                "test_idx": test_idx, "train_idx": train_idx,
                "loss_curve": loss_curve,
            }, ckpt_path)

    print(f"done. checkpoint -> {ckpt_path}")


if __name__ == "__main__":
    main()
