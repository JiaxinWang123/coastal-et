"""Prepend a 'run 00 first' pointer cell to 01_flux_ET_and_predictors.ipynb."""
import nbformat as nbf
from nbformat.v4 import new_markdown_cell

TAG = "<!-- setup-pointer -->"
POINTER = TAG + """
> **New here? Run [`00_environment_setup.ipynb`](00_environment_setup.ipynb) first.**
> It builds (or verifies) the `coastal-et` environment and registers the
> **Python (coastal-et)** kernel that every notebook in this folder uses. Once that
> is done, set this notebook's kernel to *Python (coastal-et)* and continue below.
"""

PATHS = [
    "/anvil/scratch/x-jwang120/coastal-et/notebooks/01_flux_ET_and_predictors.ipynb",
    "/home/x-jwang120/coastal-et/notebooks/01_flux_ET_and_predictors.ipynb",
    "/anvil/projects/x-ees260113/team2/coastal-et-results/notebooks/01_flux_ET_and_predictors.ipynb",
]

import os
for p in PATHS:
    if not os.path.exists(p):
        print("absent (skip):", p)
        continue
    nb = nbf.read(p, as_version=4)
    if nb.cells and TAG in nb.cells[0].get("source", ""):
        print("skip (already present):", p)
        continue
    nb.cells.insert(0, new_markdown_cell(POINTER))
    nbf.write(nb, p)
    print("added pointer:", p)
