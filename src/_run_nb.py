import sys, nbformat, traceback, time, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.show = lambda *a, **k: None
nb = nbformat.read(sys.argv[1], as_version=4)
g = {"__name__":"__main__"}
for i, c in enumerate([x for x in nb.cells if x.cell_type=="code"]):
    t0=time.time()
    try:
        exec(compile(c.source, f"<cell {i}>", "exec"), g)
        print(f"[cell {i}] OK ({time.time()-t0:.1f}s) | {c.source.splitlines()[0][:55]}", flush=True)
    except Exception:
        print(f"\n[cell {i}] *** FAILED ***\n{c.source[:500]}\n"); traceback.print_exc(); sys.exit(1)
print("\nNOTEBOOK RAN CLEAN END-TO-END")
