"""Generate 00_environment_setup.ipynb (team first-person voice).

Team edition: everyone works from the shared project folder
    /anvil/projects/x-ees260113/team2/coastal-et
and uses ONE shared conda env that lives inside it, so nobody has to rebuild.
The notebook gives (1) a one-command fast path to register the shared kernel and
(2) a from-scratch fallback that builds a private env from environment.yml.
"""
import os
import nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

PROJ = "/anvil/projects/x-ees260113/team2/coastal-et"
# The env lives OUTSIDE the synced project folder (team2/coastal-et gets re-synced and
# would delete it). team2/conda-envs is a stable sibling location.
ENVDIR = "/anvil/projects/x-ees260113/team2/conda-envs/coastal-et"
REGISTER = "/anvil/projects/x-ees260113/team2/conda-envs/register_kernel.sh"

nb = new_notebook()
c = []


def md(t):
    c.append(new_markdown_cell(t))


def code(t):
    c.append(new_code_cell(t))


md(f"""# 00 · Environment setup (team)

*The first notebook to run. It connects you to our shared analysis environment so the
rest of the workflow runs the same way for everyone on the team.*

We all work from one shared project folder on Anvil:

```
{PROJ}
```

and we all use **one shared conda environment** that lives inside it
(`env/coastal-et`, Python 3.11). Because the environment is prebuilt and shared, you
do **not** need to install anything — you only register it as a Jupyter kernel once.

This notebook has three short steps:
1. **Make the project reachable in JupyterLab** (a one-time symlink from your home).
2. **Register the shared kernel** (one command — no installing).
3. **Smoke-test** that everything imports and runs.

A from-scratch fallback (build your own env from `environment.yml`) is at the end, in
case you ever need it.""")

md(f"""## 1. Make the shared folder reachable in JupyterLab (one-time)

The JupyterLab file browser is rooted at your **home** directory and cannot navigate
up into `/anvil/projects/...`. The fix is a one-time symlink so the project shows up
in your file browser. Run this once (it is safe to re-run):

> This creates `~/coastal-et` pointing at the shared folder. After it runs, reload the
> browser tab and you will see **coastal-et** in the file list; open
> `coastal-et/notebooks/` from there.""")

code(f'''import os
link = os.path.expanduser("~/coastal-et")
target = "{PROJ}"
if os.path.islink(link) or os.path.exists(link):
    print("already present:", link, "->", os.path.realpath(link))
else:
    os.symlink(target, link)
    print("created symlink:", link, "->", target)''')

md(f"""## 2. Register the shared kernel (one command, no install)

Our shared environment is already built in a **stable** location (deliberately outside
the periodically re-synced project folder):

```
{ENVDIR}
```

Jupyter kernels are **per-user** on Anvil, so **every teammate runs this once** — it does
not install anything, it just points your Jupyter at the shared env. The `!` sends the
line to the shell.""")

code(f'''!bash {REGISTER}
!jupyter kernelspec list | grep -i coastal || echo "not registered yet"''')

md("""Now switch this notebook to that kernel: **Kernel → Change Kernel →
Python (coastal-et)** (top-right). Every other notebook in `notebooks/` expects this
same kernel.""")

md("""## 3. Smoke test

With the kernel set to **Python (coastal-et)**, run this. It imports every library the
workflow uses and does a tiny end-to-end check (a scikit-learn fit and a torch op), so
we know the environment is functional — not just importable.""")

code('''# core stack used by the analysis notebooks 01-04
import numpy as np, pandas as pd, scipy, sklearn, matplotlib
import pyarrow, joblib, xarray, dask, rasterio, rioxarray, pyproj, geopandas
import torch, xgboost, lightgbm
import stackstac, pystac_client, planetary_computer, cdsapi
from sklearn.ensemble import ExtraTreesRegressor

# offline-only extras (footprint precompute, LST sharpening) — NOT needed to run 01-04,
# so a missing one is just a note, not a failure
for _pkg in ("fluxfootprints", "pyDMS"):
    try:
        __import__(_pkg)
    except ImportError:
        print(f"  note: optional package '{_pkg}' not installed "
              f"(only used for offline preprocessing; 01-04 run fine without it)")

X = np.random.RandomState(0).rand(200, 5); y = X @ np.arange(5) + 0.1
r2 = ExtraTreesRegressor(50, random_state=0).fit(X, y).score(X, y)
t = (torch.ones(3) * 2).sum().item()

print("Python", __import__("sys").version.split()[0], "| core imports OK")
print(f"  sklearn ExtraTrees R2 (in-sample): {r2:.3f}")
print(f"  torch sanity (should be 6.0): {t}")
print("\\nEnvironment ready — open 01_data_overview.ipynb next.")''')

md(f"""## 4. Confirm you can read the shared data and figures

Everyone reads inputs and writes nothing destructive here. This just confirms the
shared products the analysis notebooks load are visible from your session.""")

code(f'''ROOT = "{PROJ}"
import os
proc = f"{{ROOT}}/data/processed"
figs = f"{{ROOT}}/figures"
print("processed tables:", len([f for f in os.listdir(proc) if f.endswith(".parquet")]),
      "parquet files")
print("figures:", len([f for f in os.listdir(figs) if f.endswith(".png")]), "PNGs")
print("\\nShared workspace is readable. You are set up.")''')

md(f"""---
## Appendix · Build your own environment (fallback)

You should not need this — the shared env above is the supported path. But if you ever
want a private copy, build one from our pinned spec. Run this in a **terminal on a
compute node** (not the login node); it takes ~10–20 min.

```bash
module load anaconda/2024.02-py311
# build into your OWN scratch (fast, big quota, but auto-purges when idle):
conda env create -p /anvil/scratch/$USER/conda-envs/coastal-et \\
    -f /anvil/projects/x-ees260113/team2/conda-envs/environment.yml

# then register it (same kernel name so notebooks still find it):
source activate /anvil/scratch/$USER/conda-envs/coastal-et
python -m ipykernel install --user --name coastal-et \\
    --display-name "Python (coastal-et)"
```

The exact versions everyone is running are also frozen in
`/anvil/projects/x-ees260113/team2/conda-envs/pip-freeze-lock.txt`.""")

nb["cells"] = c
nb["metadata"] = {
    "kernelspec": {"display_name": "Python (coastal-et)", "language": "python",
                   "name": "coastal-et"},
    "language_info": {"name": "python"},
}

OUTS = [
    "/anvil/scratch/x-jwang120/coastal-et/notebooks/00_environment_setup.ipynb",  # sync source
    f"{PROJ}/notebooks/00_environment_setup.ipynb",
    "/home/x-jwang120/coastal-et/notebooks/00_environment_setup.ipynb",
]
for out in OUTS:
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        nbf.write(nb, f)
    print("wrote", out, "-", len(c), "cells")
