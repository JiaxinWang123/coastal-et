"""Does the training SUPPORT (window size) change spatial-transfer skill?

We currently train on 500 m window-mean satellite features but predict per 30 m pixel.
Here we rebuild the 13-site feature table at 90 / 250 / 500 m windows (all already
stored in the tower-point parquets) and compare leave-site-out R2, to choose a support
that is more consistent with 30 m prediction.
"""
import sys
sys.path.insert(0, "/anvil/scratch/x-jwang120/coastal-et/src")
import numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
import more_sites as M
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import r2_score, mean_absolute_error

FEATS = M.FEATS
SITES = M.EVERGLADES + M.ADDED


def build_table(win):
    """Rebuild the 13-site table using the given window (monkeypatch more_sites.W)."""
    M.W = str(win)
    frames = []
    for s in SITES:
        try:
            d = M.build_site(s)
        except Exception:
            d = None
        if d is not None and len(d):
            frames.append(d)
    return pd.concat(frames, ignore_index=True) if frames else None


def leave_site(d):
    yt, yp = [], []
    for s in sorted(d.SITE_ID.unique()):
        tr, te = d[d.SITE_ID != s], d[d.SITE_ID == s]
        if len(te) < 5:
            continue
        m = ExtraTreesRegressor(400, min_samples_leaf=2, random_state=0, n_jobs=-1)
        m.fit(tr[FEATS].values, tr.ET_closed_mm.values)
        yp.append(m.predict(te[FEATS].values)); yt.append(te.ET_closed_mm.values)
    yt, yp = np.concatenate(yt), np.concatenate(yp)
    return r2_score(yt, yp), mean_absolute_error(yt, yp)


print(f"sites: {len(SITES)}  features: {FEATS}\n")
print(f"{'window':>8}{'n':>7}{'sites':>7}{'leave-site R2':>15}{'MAE':>8}")
print("-" * 45)
for win in [90, 250, 500]:
    d = build_table(win)
    if d is None:
        print(f"{win:>6}m   no data"); continue
    r2, mae = leave_site(d)
    print(f"{win:>6}m{len(d):>7}{d.SITE_ID.nunique():>7}{r2:>15.3f}{mae:>8.2f}", flush=True)
print("\n90 m (~3x3 Landsat pixels) is the closest to 30 m prediction support and to the")
print("marsh flux footprints (~0.3-0.6 ha); 500 m is the current (over-smoothed) default.")
