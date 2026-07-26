"""Batch driver: predict ET for every reserve polygon in team2/shp_predict and write
GeoTIFFs, .npz, a summary CSV, and a panel figure. All logic lives in reserve_et.py
(portable). Open water is masked out.
"""
import os
import sys
import glob
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import reserve_et as RE

OUT = f"{RE.PROC}/reserve_maps"
FIGD = os.path.join(RE.ROOT, "figures", "reserve_maps")


def main():
    model, feats = RE.load_production_model()   # ExtraTrees on the 11 VIF-pruned features
    print(f"production model loaded ({len(feats)} features); shapefiles in {RE.SHP_DIR}")
    cat = RE.open_catalog()
    das = RE.open_gridmet()
    rows = []
    for shp in sorted(glob.glob(f"{RE.SHP_DIR}/*/*.shp")):
        name = os.path.basename(os.path.dirname(shp))
        try:
            r = RE.predict_reserve(shp, model, cat, das, mask_water=True, feats=feats)
            if r is None:
                print(f"  {name}: no clear scene"); continue
            RE.save_outputs(r, OUT)
            wf = 100 * r["water_px"] / max(r["inside_px"], 1)
            print(f"  {name}: {r['date']} cloud {r['cloud']}% | {r['pixels']:,} land px | "
                  f"water {wf:.0f}% masked | mean ET {r['mean_ET']} mm/d", flush=True)
            rows.append(r)
        except Exception as e:
            import traceback
            print(f"  {name}: ERROR {type(e).__name__}: {e}"); traceback.print_exc()

    cols = ["reserve", "date", "cloud", "mean_ET", "min_ET", "max_ET",
            "pixels", "water_px", "inside_px", "epsg"]
    summ = pd.DataFrame([{k: r[k] for k in cols} for r in rows])
    os.makedirs(OUT, exist_ok=True)
    summ.to_csv(f"{OUT}/reserve_ET_summary.csv", index=False)
    print("\n=== SUMMARY ===\n" + summ.to_string(index=False))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import geopandas as gpd
    from matplotlib.colors import LinearSegmentedColormap
    PAL = ["#DEC29B", "#EDD9A6", "#FFF4AD", "#C3E683", "#6BCC5C", "#3BB369",
           "#20998F", "#16678A", "#114982"]
    cmap = LinearSegmentedColormap.from_list("et", PAL, N=256); cmap.set_bad("#eeeeea")
    n = len(rows); ncol = 3; nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 3.4, nrow * 3.2))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.axis("off")
    for ax, r in zip(axes, rows):
        ext = [r["x"].min(), r["x"].max(), r["y"].min(), r["y"].max()]
        im = ax.imshow(np.ma.masked_invalid(r["et"]), origin="upper", extent=ext,
                       cmap=cmap, vmin=1, vmax=6)
        gpd.GeoSeries([r["poly"]], crs=r["epsg"]).boundary.plot(ax=ax, color="#333", lw=0.6)
        ax.set_title(f"{r['reserve']}  {r['date']}\nmean {r['mean_ET']} mm/d",
                     fontsize=8.5, fontweight="bold")
        ax.set_xticks([]); ax.set_yticks([]); ax.axis("on")
        for sp in ax.spines.values():
            sp.set_visible(False)
    cb = fig.colorbar(im, ax=axes.tolist(), fraction=0.02, pad=0.02, extend="both")
    cb.set_label("predicted daily ET (mm/day)")
    fig.suptitle("Predicted ET across NERR coastal-wetland reserves (open water masked)",
                 fontsize=12, fontweight="bold")
    os.makedirs(FIGD, exist_ok=True)
    fig.savefig(f"{FIGD}/reserve_ET_panel.png", dpi=170, bbox_inches="tight", facecolor="white")
    print(f"\nwrote {FIGD}/reserve_ET_panel.png and {len(rows)} GeoTIFFs in {OUT}")


if __name__ == "__main__":
    main()
