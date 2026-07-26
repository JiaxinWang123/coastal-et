"""Physics-informed ET: embed the surface energy balance + Priestley-Taylor.

TSEB's canopy source initialises transpiration with Priestley-Taylor:
    LE = alpha * [Delta/(Delta+gamma)] * (Rn - G)
where Delta = slope of the saturation-vapour-pressure curve (f of T), gamma =
psychrometric constant, and alpha is the PT coefficient (~1.26 well-watered,
lower under moisture/salinity stress). alpha is a DIMENSIONLESS SURFACE property
-- far more transferable between sites than absolute ET, because the energy
magnitude (Rn) is supplied by physics, not learned.

Models compared (all on the same footprint-weighted features, 3 CV schemes):
  PT-fixed        alpha = 1.26 everywhere            pure physics, no ML
  PT-calibrated   alpha = mean(training alpha)       physics + 1 constant
  PINN            NN predicts alpha in [0,1.35]      PHYSICS-INFORMED
  MLP (ET)        NN predicts ET directly            pure ML reference
  GP  (ET)        Gaussian process on ET             best classical reference

Hypothesis: predicting alpha (surface control) x physics-supplied Rn generalises
to an UNSEEN site better than learning ET directly -- the leave-tower-out case
that pure ML could not do.
"""
import os
import sys
import glob
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
from sklearn.metrics import r2_score, mean_absolute_error
import train_indices_model as T

torch.set_num_threads(8)
R = "/anvil/scratch/x-jwang120/coastal-et"
SITES = ["US-Esm", "US-TaS", "US-Skr", "US-Elm", "US-EvM"]
FEATS = T.FEATS
LAMBDA = 2.45e6            # J/kg
W2MM = 86400.0 / LAMBDA   # W/m2 (daily mean) -> mm/day  (~0.03526)
GAMMA = 0.066             # kPa/degC psychrometric constant (~sea level)


def delta_svp(Tc):
    """Slope of saturation vapour pressure curve, kPa/degC (Tetens)."""
    es = 0.6108 * np.exp(17.27 * Tc / (Tc + 237.3))
    return 4098 * es / (Tc + 237.3) ** 2


def rn_for_skr():
    """Daily Rn for US-Skr from its ERA5 net-radiation components (no tower NETRAD)."""
    files = glob.glob(f"{R}/data/raw/ameriflux/fluxnet/../../../projects", recursive=True)
    fs = glob.glob("/anvil/projects/x-ees260113/team2/datasets/ERA5/**/*skr*.csv",
                   recursive=True)
    if not fs:
        return None
    fr = []
    for f in fs:
        c = pd.read_csv(f)
        if "date" not in c:
            continue
        c["date"] = pd.to_datetime(c["date"], utc=True, errors="coerce")
        fr.append(c)
    if not fr:
        return None
    c = pd.concat(fr).dropna(subset=["date"]).set_index("date").sort_index()
    ns = pd.to_numeric(c.get("surface_net_solar_radiation_hourly"), errors="coerce")
    nt = pd.to_numeric(c.get("surface_net_thermal_radiation_hourly"), errors="coerce")
    rn = (ns + nt) / 3600.0                        # J/m2/h -> W/m2
    return rn.resample("D").mean().rename("NETRAD")


def build():
    d = T.build()
    et = pd.read_parquet(f"{R}/data/processed/daily_closed_et.parquet")
    et.index = pd.to_datetime(et.index, utc=True)
    rn_map = {}
    skr = rn_for_skr()
    for s in SITES:
        e = et[et.SITE_ID == s][["NETRAD"]].copy()
        e.index = e.index.normalize()
        r = e["NETRAD"]
        if r.notna().sum() == 0 and skr is not None:
            r = skr.reindex(e.index)
        rn_map[s] = r
    d["date"] = pd.to_datetime(d["date"], utc=True).dt.normalize()
    d["Rn"] = [rn_map.get(s, pd.Series()).get(dt, np.nan)
               for s, dt in zip(d.SITE_ID, d.date)]
    # equilibrium (Priestley-Taylor) ET, and observed alpha
    Tc = d["TA_ERA"].values
    dl = delta_svp(Tc)
    d["ET_eq"] = (dl / (dl + GAMMA)) * d["Rn"].values * W2MM   # mm/day, G~0 daily
    d = d[(d.ET_eq > 0.2) & d.Rn.notna()].copy()
    d["alpha_obs"] = (d.ET_closed_mm / d.ET_eq).clip(0, 2.0)
    return d.dropna(subset=["ET_closed_mm"] + FEATS + ["ET_eq"]).reset_index(drop=True)


class Net(nn.Module):
    def __init__(self, nin, out_act):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(nin, 48), nn.BatchNorm1d(48), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(48, 24), nn.ReLU(), nn.Dropout(0.2), nn.Linear(24, 1))
        self.out_act = out_act

    def forward(self, x):
        return self.out_act(self.net(x).squeeze(-1))


def train_net(Xtr, ytr, out_act, seed=0, epochs=400):
    torch.manual_seed(seed); np.random.seed(seed)
    n = len(Xtr); idx = np.random.permutation(n); nv = max(8, int(0.15 * n))
    vi, ti = idx[:nv], idx[nv:]
    Xt = torch.tensor(Xtr[ti], dtype=torch.float32); yt = torch.tensor(ytr[ti], dtype=torch.float32)
    Xv = torch.tensor(Xtr[vi], dtype=torch.float32); yv = torch.tensor(ytr[vi], dtype=torch.float32)
    m = Net(Xtr.shape[1], out_act)
    opt = torch.optim.Adam(m.parameters(), lr=3e-3, weight_decay=1e-3)
    lf = nn.SmoothL1Loss(); best, bs, bad = 1e9, None, 0
    for ep in range(epochs):
        m.train(); opt.zero_grad(); lf(m(Xt), yt).backward(); opt.step()
        m.eval()
        with torch.no_grad():
            vl = lf(m(Xv), yv).item()
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


def predict_net(m, X):
    with torch.no_grad():
        return m(torch.tensor(X, dtype=torch.float32)).numpy()


# --- predictors returning ET on the test rows ---
def pinn(tr, te):
    """NN predicts alpha in [0,1.35]; ET = alpha * ET_eq (ensemble of 3)."""
    sc = StandardScaler().fit(tr[FEATS].values)
    Xtr, Xte = sc.transform(tr[FEATS].values), sc.transform(te[FEATS].values)
    act = lambda z: torch.sigmoid(z) * 1.35
    a = np.mean([predict_net(train_net(Xtr, tr.alpha_obs.values, act, s), Xte)
                 for s in range(3)], axis=0)
    return a * te.ET_eq.values


def mlp_et(tr, te):
    sc = StandardScaler().fit(tr[FEATS].values)
    Xtr, Xte = sc.transform(tr[FEATS].values), sc.transform(te[FEATS].values)
    act = lambda z: torch.relu(z)
    return np.mean([predict_net(train_net(Xtr, tr.ET_closed_mm.values, act, s), Xte)
                    for s in range(3)], axis=0)


def gp_et(tr, te):
    from sklearn.pipeline import make_pipeline
    m = make_pipeline(StandardScaler(), GaussianProcessRegressor(
        kernel=ConstantKernel() * RBF() + WhiteKernel(), normalize_y=True,
        alpha=1e-3, random_state=0)).fit(tr[FEATS].values, tr.ET_closed_mm.values)
    return m.predict(te[FEATS].values)


def pt_fixed(tr, te):
    return 1.26 * te.ET_eq.values


def pt_cal(tr, te):
    return tr.alpha_obs.mean() * te.ET_eq.values


PREDICTORS = {"PT-fixed(1.26)": pt_fixed, "PT-calibrated": pt_cal,
              "PINN (predict alpha)": pinn, "MLP (predict ET)": mlp_et,
              "GP (predict ET)": gp_et}


def eval_scheme(d, fn, scheme):
    yt, yp = [], []
    if scheme == "random":
        for tri, tei in KFold(10, shuffle=True, random_state=0).split(d):
            tr, te = d.iloc[tri], d.iloc[tei]
            yp.append(fn(tr, te)); yt.append(te.ET_closed_mm.values)
    elif scheme == "year":
        for s in SITES:
            ds = d[d.SITE_ID == s]
            for y in sorted(ds.date.dt.year.unique()):
                tr, te = ds[ds.date.dt.year != y], ds[ds.date.dt.year == y]
                if len(te) < 5 or len(tr) < 20:
                    continue
                yp.append(fn(tr, te)); yt.append(te.ET_closed_mm.values)
    else:  # tower
        for s in SITES:
            tr, te = d[d.SITE_ID != s], d[d.SITE_ID == s]
            if len(te) < 5:
                continue
            yp.append(fn(tr, te)); yt.append(te.ET_closed_mm.values)
    yt, yp = np.concatenate(yt), np.concatenate(yp)
    return r2_score(yt, yp), mean_absolute_error(yt, yp)


def main():
    d = build()
    print(f"{len(d)} tower-date samples with Rn+ET_eq, {d.SITE_ID.nunique()} sites")
    print(f"  observed PT coefficient alpha: median {d.alpha_obs.median():.2f} "
          f"(IQR {d.alpha_obs.quantile(.25):.2f}-{d.alpha_obs.quantile(.75):.2f})\n")
    print(f"{'model':<22}{'random-CV':>11}{'leave-year':>12}{'leave-tower':>13}")
    print("-" * 58)
    for name, fn in PREDICTORS.items():
        rr = eval_scheme(d, fn, "random")[0]
        ry = eval_scheme(d, fn, "year")[0]
        rt = eval_scheme(d, fn, "tower")[0]
        print(f"{name:<22}{rr:>11.3f}{ry:>12.3f}{rt:>13.3f}", flush=True)


if __name__ == "__main__":
    main()
