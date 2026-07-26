"""Make all coastal-et paths relocatable.

- src/*.py and src/probe/*.py: compute the project root from __file__ instead of a
  hardcoded personal scratch path; sys.path.insert(.../src) becomes __file__-relative;
  any remaining embedded scratch literals are repointed to the shared project path.
- scripts/*.sh: repoint the scratch path to the shared project path.
Idempotent: rerunning finds nothing to change.
"""
import os
import re
import glob

PROJ = "/anvil/projects/x-ees260113/team2/coastal-et"
SCR = "/anvil/scratch/x-jwang120/coastal-et"


def dirname_expr(n):
    e = "os.path.abspath(__file__)"
    for _ in range(n):
        e = f"os.path.dirname({e})"
    return e


def has_import_os(txt):
    return re.search(r"^\s*(?:import\s+os\b|import\s+[^\n]*\bos\b|from\s+os\b)", txt, re.M) is not None


def ensure_import_os(txt):
    if has_import_os(txt):
        return txt
    lines = txt.split("\n")
    ins = 1 if lines and lines[0].startswith("#!") else 0
    rest = "\n".join(lines[ins:])
    m = re.match(r'\s*(?:"""(?:.|\n)*?"""|\'\'\'(?:.|\n)*?\'\'\')', rest)
    if m:
        ins += m.group(0).count("\n") + 1
    lines.insert(ins, "import os")
    return "\n".join(lines)


def fix_py(path):
    rel = os.path.relpath(path, PROJ)          # src/foo.py  or  src/probe/foo.py
    # dirname() applications to go from the FILE up to PROJ = path components
    # from PROJ to the file = number of separators + 1.
    depth = rel.count(os.sep) + 1
    proj_e = dirname_expr(depth)
    src_e = f'os.path.join({proj_e}, "src")'
    txt = open(path).read()
    orig = txt
    # 1) sys.path.insert(..., "<SCR>/src")  ->  __file__-relative src dir
    txt = txt.replace(f'"{SCR}/src"', src_e).replace(f"'{SCR}/src'", src_e)
    # 2) bare root literal "<SCR>"  ->  __file__-relative project root
    txt = txt.replace(f'"{SCR}"', proj_e).replace(f"'{SCR}'", proj_e)
    # 3) any leftover embedded scratch path -> shared project path (string-safe)
    txt = txt.replace(SCR, PROJ)
    if txt != orig:
        if "os.path" in txt or "os.environ" in txt:
            txt = ensure_import_os(txt)
        open(path, "w").write(txt)
        return True
    return False


def fix_sh(path):
    txt = open(path).read()
    if SCR not in txt:
        return False
    open(path, "w").write(txt.replace(SCR, PROJ))
    return True


changed_py, changed_sh = [], []
for f in glob.glob(f"{PROJ}/src/**/*.py", recursive=True):
    if fix_py(f):
        changed_py.append(os.path.relpath(f, PROJ))
for f in glob.glob(f"{PROJ}/scripts/**/*.sh", recursive=True):
    if fix_sh(f):
        changed_sh.append(os.path.relpath(f, PROJ))

print(f"python files changed: {len(changed_py)}")
print(f"shell files changed:  {len(changed_sh)}")
# report anything still referencing the personal path
leftover = []
for f in glob.glob(f"{PROJ}/**/*.py", recursive=True) + glob.glob(f"{PROJ}/scripts/**/*.sh", recursive=True):
    if SCR in open(f).read():
        leftover.append(os.path.relpath(f, PROJ))
print(f"files still referencing scratch/x-jwang120: {len(leftover)}")
for x in leftover:
    print("   LEFTOVER:", x)
