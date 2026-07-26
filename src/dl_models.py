"""Deep-learning ET regression (PyTorch) vs the best classical model.

With only ~227 samples, deep nets overfit easily, so we use strong regularisation
(dropout, weight decay, early stopping) and a 5-net ENSEMBLE (averaging reduces
variance, which is the main failure mode of small-data DL). Evaluated on the same
three schemes: random-CV, leave-year-out, leave-tower-out.

Architectures:
  MLP-shallow   14 -> 32 -> 1
  MLP-deep      14 -> 64 -> 32 -> 16 -> 1   (dropout 0.3, BatchNorm)
  MLP-ensemble  mean of 5 MLP-deep with different seeds
"""
import os
import sys
sys.path.insert(0, "/anvil/scratch/x-jwang120/coastal-et/src")
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score
import train_indices_model as T

torch.set_num_threads(8)
SITES = ["US-Esm", "US-TaS", "US-Skr", "US-Elm", "US-EvM"]
FEATS = T.FEATS
DEV = "cpu"


class MLP(nn.Module):
    def __init__(self, nin, hidden, p=0.3):
        super().__init__()
        layers = []
        d = nin
        for h in hidden:
            layers += [nn.Linear(d, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(p)]
            d = h
        layers += [nn.Linear(d, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_one(Xtr, ytr, hidden, seed, epochs=400, p=0.3):
    torch.manual_seed(seed)
    np.random.seed(seed)
    n = len(Xtr)
    idx = np.random.permutation(n)
    nval = max(8, int(0.15 * n))
    vi, ti = idx[:nval], idx[nval:]
    Xt = torch.tensor(Xtr[ti], dtype=torch.float32)
    yt = torch.tensor(ytr[ti], dtype=torch.float32)
    Xv = torch.tensor(Xtr[vi], dtype=torch.float32)
    yv = torch.tensor(ytr[vi], dtype=torch.float32)
    m = MLP(Xtr.shape[1], hidden, p)
    opt = torch.optim.Adam(m.parameters(), lr=3e-3, weight_decay=1e-3)
    lossf = nn.SmoothL1Loss()
    best, best_state, patience, bad = 1e9, None, 40, 0
    for ep in range(epochs):
        m.train()
        opt.zero_grad()
        loss = lossf(m(Xt), yt)
        loss.backward()
        opt.step()
        m.eval()
        with torch.no_grad():
            vl = lossf(m(Xv), yv).item()
        if vl < best - 1e-4:
            best, best_state, bad = vl, {k: v.clone() for k, v in m.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state:
        m.load_state_dict(best_state)
    m.eval()
    return m


def predict(m, X):
    with torch.no_grad():
        return m(torch.tensor(X, dtype=torch.float32)).numpy()


def dl_fit_pred(Xtr, ytr, Xte, hidden, n_ens=1):
    sc = StandardScaler().fit(Xtr)
    Xtr_s, Xte_s = sc.transform(Xtr), sc.transform(Xte)
    preds = [predict(train_one(Xtr_s, ytr, hidden, seed=s), Xte_s)
             for s in range(n_ens)]
    return np.mean(preds, axis=0)


def evaluate(d, hidden, n_ens):
    # random-CV
    kf = KFold(10, shuffle=True, random_state=0)
    yt, yp = [], []
    for tri, tei in kf.split(d):
        tr, te = d.iloc[tri], d.iloc[tei]
        yp.append(dl_fit_pred(tr[FEATS].values, tr.ET_closed_mm.values,
                              te[FEATS].values, hidden, n_ens))
        yt.append(te.ET_closed_mm.values)
    r_rand = r2_score(np.concatenate(yt), np.concatenate(yp))
    # leave-year
    yt, yp = [], []
    for s in SITES:
        ds = d[d.SITE_ID == s]
        for y in sorted(ds.year.unique()):
            tr, te = ds[ds.year != y], ds[ds.year == y]
            if len(te) < 5 or len(tr) < 20:
                continue
            yp.append(dl_fit_pred(tr[FEATS].values, tr.ET_closed_mm.values,
                                  te[FEATS].values, hidden, n_ens))
            yt.append(te.ET_closed_mm.values)
    r_year = r2_score(np.concatenate(yt), np.concatenate(yp))
    # leave-tower
    yt, yp = [], []
    for s in SITES:
        tr, te = d[d.SITE_ID != s], d[d.SITE_ID == s]
        if len(te) < 5:
            continue
        yp.append(dl_fit_pred(tr[FEATS].values, tr.ET_closed_mm.values,
                              te[FEATS].values, hidden, n_ens))
        yt.append(te.ET_closed_mm.values)
    r_tower = r2_score(np.concatenate(yt), np.concatenate(yp))
    return r_rand, r_year, r_tower


def main():
    d = T.build().dropna(subset=["ET_closed_mm"] + FEATS).reset_index(drop=True)
    print(f"{len(d)} samples, {len(FEATS)} features (PyTorch {torch.__version__}, CPU)\n")

    rows = []
    for name, hidden, ens in [("MLP-shallow", [32], 1),
                              ("MLP-deep", [64, 32, 16], 1),
                              ("MLP-ensemble(5)", [64, 32, 16], 5)]:
        r = evaluate(d, hidden, ens)
        rows.append((name,) + r)
        print(f"{name:<18} random-CV={r[0]:.3f}  leave-year={r[1]:.3f}  leave-tower={r[2]:.3f}",
              flush=True)

    # classical GP reference on the same splits
    gp = make_pipeline(StandardScaler(), GaussianProcessRegressor(
        kernel=ConstantKernel() * RBF() + WhiteKernel(), normalize_y=True,
        alpha=1e-3, random_state=0))
    from sklearn.base import clone

    def gp_eval():
        kf = KFold(10, shuffle=True, random_state=0)
        yt, yp = [], []
        for tri, tei in kf.split(d):
            tr, te = d.iloc[tri], d.iloc[tei]
            m = clone(gp).fit(tr[FEATS].values, tr.ET_closed_mm.values)
            yp.append(m.predict(te[FEATS].values)); yt.append(te.ET_closed_mm.values)
        rr = r2_score(np.concatenate(yt), np.concatenate(yp))
        yt, yp = [], []
        for s in SITES:
            ds = d[d.SITE_ID == s]
            for y in sorted(ds.year.unique()):
                tr, te = ds[ds.year != y], ds[ds.year == y]
                if len(te) < 5 or len(tr) < 20:
                    continue
                m = clone(gp).fit(tr[FEATS].values, tr.ET_closed_mm.values)
                yp.append(m.predict(te[FEATS].values)); yt.append(te.ET_closed_mm.values)
        ry = r2_score(np.concatenate(yt), np.concatenate(yp))
        return rr, ry
    rr, ry = gp_eval()
    print(f"\n{'GaussianProcess (ref)':<18} random-CV={rr:.3f}  leave-year={ry:.3f}")
    print("\nNote: 227 samples is far below the DL regime; regularised trees / GP are"
          "\nexpected to match or beat deep nets here.")


if __name__ == "__main__":
    main()
