"""Energy-balance-constrained and PINN models on the 13-site coastal network.

Does embedding physics improve spatial upscaling (leave-site-out) beyond plain ML?
All models predict daily ET on overpass days; physics supplies the available energy.

  Plain-ET        ExtraTrees predicts ET directly                (baseline, R2~0.72)
  EF-tree         ExtraTrees predicts EF=ET/(Rn.W2MM); ET=EF.Rn.W2MM   (energy balance)
  PT-tree         ExtraTrees predicts alpha=ET/ET_eq; ET=alpha.ET_eq   (Priestley-Taylor)
  PINN-PT         NN predicts alpha in [0,1.35]; ET=alpha.ET_eq        (physics-informed NN)
  PINN-EB         NN predicts (LE,H) with a soft energy-balance loss   (Rn-G=H+LE)

Rn: tower NETRAD where available (11 sites); estimated from gridMET solar radiation
for US-Skr and US-StJ via a fitted Rn~SRAD relationship.
"""
import os
import sys
sys.path.insert(0, "/anvil/scratch/x-jwang120/coastal-et/src")
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.base import clone
from sklearn.metrics import r2_score
import torch
import torch.nn as nn
import more_sites as M
torch.set_num_threads(8)

R = "/anvil/scratch/x-jwang120/coastal-et"
SITES = M.EVERGLADES + M.ADDED
FE = M.FEATS
LAMBDA = 2.45e6
W2MM = 86400.0 / LAMBDA
GAMMA = 0.066


def delta_svp(Tc):
    es = 0.6108 * np.exp(17.27 * Tc / (Tc + 237.3))
    return 4098 * es / (Tc + 237.3) ** 2


def build():
    et = pd.read_parquet(f"{R}/data/processed/daily_closed_et.parquet")
    et.index = pd.to_datetime(et.index, utc=True)
    # fit Rn ~ SRAD on sites that have both, to fill Skr/StJ
    fitx, fity = [], []
    per = {}
    for s in SITES:
        e = et[(et.SITE_ID == s)].copy(); e.index = e.index.normalize()
        rn = e["NETRAD"]
        gm = f"{R}/data/interim/gridmet/{s}_gridmet.parquet"
        srad = None
        if os.path.exists(gm):
            g = pd.read_parquet(gm); g.index = pd.to_datetime(g.index, utc=True).normalize()
            srad = g["SRAD_Wm2"] if "SRAD_Wm2" in g else None
        per[s] = (rn, srad, e["TA_C"], e["G"])
        if srad is not None and rn.notna().sum() > 30:
            j = pd.concat([rn.rename("rn"), srad.rename("sr")], axis=1).dropna()
            fitx.append(j.sr.values); fity.append(j.rn.values)
    fx, fy = np.concatenate(fitx), np.concatenate(fity)
    b1, b0 = np.polyfit(fx, fy, 1)
    print(f"  Rn ~ {b1:.2f}*SRAD + {b0:.0f}  (fallback for sites without NETRAD)")

    frames = []
    for s in SITES:
        d = M.build_site(s)                       # features + ET + met, date index
        if d is None or len(d) < 10:
            continue
        rn, srad, ta, g = per[s]
        r = rn.reindex(d.index)
        if r.notna().sum() == 0 and srad is not None:      # Skr / StJ
            r = (b0 + b1 * srad.reindex(d.index))
        d["Rn"] = r.values
        d["G"] = g.reindex(d.index).fillna(0).values
        d["ET_avail"] = d["Rn"] * W2MM             # mm/day if EF=1
        dl = delta_svp(d["TA_ERA"].values)
        d["ET_eq"] = (dl / (dl + GAMMA)) * d["Rn"].values * W2MM
        frames.append(d)
    data = pd.concat(frames, ignore_index=True)
    data = data[(data.ET_avail > 0.5) & (data.ET_eq > 0.2)].copy()
    data["EF"] = (data.ET_closed_mm / data.ET_avail).clip(0, 1.2)
    data["alpha"] = (data.ET_closed_mm / data.ET_eq).clip(0, 2.0)
    data["LE_obs"] = data.ET_closed_mm / W2MM      # W/m2 daily-mean
    data["H_obs"] = (data.Rn - data.G) - data.LE_obs
    return data.dropna(subset=["ET_closed_mm", "Rn"] + FE).reset_index(drop=True)


# ---------- tree predictors returning ET ----------
def et_tree():
    return ExtraTreesRegressor(500, min_samples_leaf=2, random_state=0, n_jobs=-1)


def p_plain(tr, te):
    m = clone(et_tree()).fit(tr[FE].values, tr.ET_closed_mm.values)
    return m.predict(te[FE].values)


def p_ef(tr, te):
    m = clone(et_tree()).fit(tr[FE].values, tr.EF.values)
    return np.clip(m.predict(te[FE].values), 0, 1.2) * te.ET_avail.values


def p_pt(tr, te):
    m = clone(et_tree()).fit(tr[FE].values, tr.alpha.values)
    return np.clip(m.predict(te[FE].values), 0, 1.6) * te.ET_eq.values


# ---------- PINNs ----------
class Net(nn.Module):
    def __init__(self, nin, nout=1):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(nin, 64), nn.BatchNorm1d(64), nn.ReLU(),
                                 nn.Dropout(0.3), nn.Linear(64, 32), nn.ReLU(),
                                 nn.Dropout(0.2), nn.Linear(32, nout))

    def forward(self, x):
        return self.net(x)


def _train(Xtr, extra, targets, loss_fn, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    n = len(Xtr); idx = np.random.permutation(n); nv = max(20, int(0.15 * n))
    vi, ti = idx[:nv], idx[nv:]
    X = torch.tensor(Xtr, dtype=torch.float32)
    ex = {k: torch.tensor(v, dtype=torch.float32) for k, v in extra.items()}
    tg = {k: torch.tensor(v, dtype=torch.float32) for k, v in targets.items()}
    nout = 2 if "H_obs" in targets else 1
    m = Net(Xtr.shape[1], nout)
    opt = torch.optim.Adam(m.parameters(), lr=3e-3, weight_decay=1e-3)
    best, bs, bad = 1e9, None, 0
    for ep in range(500):
        m.train(); opt.zero_grad()
        out = m(X[ti])
        loss = loss_fn(out, {k: v[ti] for k, v in ex.items()}, {k: v[ti] for k, v in tg.items()})
        loss.backward(); opt.step()
        m.eval()
        with torch.no_grad():
            vout = m(X[vi])
            vl = loss_fn(vout, {k: v[vi] for k, v in ex.items()},
                         {k: v[vi] for k, v in tg.items()}).item()
        if vl < best - 1e-4:
            best, bs, bad = vl, {k: v.clone() for k, v in m.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= 40:
                break
    if bs:
        m.load_state_dict(bs)
    m.eval()
    return m


def p_pinn_pt(tr, te, n_ens=5):
    sc = StandardScaler().fit(tr[FE].values)
    Xtr, Xte = sc.transform(tr[FE].values), sc.transform(te[FE].values)
    hub = nn.SmoothL1Loss()

    def loss_fn(out, ex, tg):
        alpha = torch.sigmoid(out[:, 0]) * 1.35
        et = alpha * ex["ET_eq"]
        return hub(et, tg["ET"])
    preds = []
    for s in range(n_ens):
        m = _train(Xtr, {"ET_eq": tr.ET_eq.values}, {"ET": tr.ET_closed_mm.values}, loss_fn, s)
        with torch.no_grad():
            a = torch.sigmoid(m(torch.tensor(Xte, dtype=torch.float32))[:, 0]) * 1.35
        preds.append((a.numpy()) * te.ET_eq.values)
    return np.mean(preds, axis=0)


def p_pinn_eb(tr, te, n_ens=5, lam=0.3):
    """NN predicts (LE, H); soft loss enforces Rn-G = H+LE. Returns ET (mm/day)."""
    sc = StandardScaler().fit(tr[FE].values)
    Xtr, Xte = sc.transform(tr[FE].values), sc.transform(te[FE].values)
    hub = nn.SmoothL1Loss()

    def loss_fn(out, ex, tg):
        le, h = out[:, 0], out[:, 1]
        data = hub(le, tg["LE_obs"]) + hub(h, tg["H_obs"])
        phys = ((ex["avail"] - h - le) ** 2).mean()
        return data + lam * phys
    preds = []
    for s in range(n_ens):
        m = _train(Xtr, {"avail": (tr.Rn - tr.G).values},
                   {"LE_obs": tr.LE_obs.values, "H_obs": tr.H_obs.values}, loss_fn, s)
        with torch.no_grad():
            le = m(torch.tensor(Xte, dtype=torch.float32))[:, 0].numpy()
        preds.append(le * W2MM)
    return np.mean(preds, axis=0)


def evalR2(d, fn, scheme):
    yt, yp = [], []
    if scheme == "kfold":
        for tri, tei in KFold(10, shuffle=True, random_state=0).split(d):
            yp.append(fn(d.iloc[tri], d.iloc[tei])); yt.append(d.iloc[tei].ET_closed_mm.values)
    elif scheme == "year":
        for s in SITES:
            ds = d[d.SITE_ID == s]
            for y in sorted(ds.year.dropna().unique()):
                tr, te = ds[ds.year != y], ds[ds.year == y]
                if len(te) < 5 or len(tr) < 15:
                    continue
                yp.append(fn(tr, te)); yt.append(te.ET_closed_mm.values)
    else:
        for s in SITES:
            tr, te = d[d.SITE_ID != s], d[d.SITE_ID == s]
            if len(te) < 5:
                continue
            yp.append(fn(tr, te)); yt.append(te.ET_closed_mm.values)
    return r2_score(np.concatenate(yt), np.concatenate(yp))


def main():
    d = build()
    print(f"{len(d)} samples, {d.SITE_ID.nunique()} sites")
    print(f"  observed EF median {d.EF.median():.2f}, alpha median {d.alpha.median():.2f}\n")
    models = [("Plain-ET (tree)", p_plain), ("EF-constrained (tree)", p_ef),
              ("PT-constrained (tree)", p_pt), ("PINN-PT (alpha)", p_pinn_pt),
              ("PINN-EB (LE,H soft)", p_pinn_eb)]
    print(f"{'model':<24}{'K-fold':>9}{'leave-year':>12}{'leave-site':>12}")
    print("-" * 57)
    rows = []
    for name, fn in models:
        rr = evalR2(d, fn, "kfold"); ry = evalR2(d, fn, "year"); rs = evalR2(d, fn, "site")
        rows.append((name, rr, ry, rs))
        print(f"{name:<24}{rr:>9.3f}{ry:>12.3f}{rs:>12.3f}", flush=True)
    pd.DataFrame(rows, columns=["model", "kfold", "leave_year", "leave_site"]).to_csv(
        f"{R}/data/processed/physics_models.csv", index=False)
    print(f"\nwrote physics_models.csv")


if __name__ == "__main__":
    main()
