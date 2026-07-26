# `src/` — analysis & pipeline code

These scripts are the full "how we built it" code (feature engineering, model
selection, footprint analysis, figure generation, spatial prediction). Several
contain **absolute Anvil HPC paths** (`/anvil/...`) from the environment they were
run in — they are provided for transparency and reference, not turnkey portability.

**The portable, reproducible entry point is the `notebooks/` folder** (paths are
derived from the notebook location). `src/reserve_et.py` is the one module the
notebooks import directly and it *is* path-portable.
