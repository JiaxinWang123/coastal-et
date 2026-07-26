"""Idempotent I/O helpers for the download + preprocessing scripts.

Behaviour when an output already exists:
  * batch / non-interactive run  -> SKIP (keep existing), print a note;
                                    pass --force (or COASTAL_FORCE=1) to overwrite.
  * interactive run (a TTY)       -> ASK the user whether to re-download/overwrite.
Missing output, or --force / COASTAL_FORCE=1  -> proceed (create/overwrite).

Usage in a script:
    import io_utils
    argv = io_utils.clean_argv()          # positional args with --force stripped out
    ...
    if io_utils.should_write(dest, label="gridMET for US-Skr"):
        ... do the download/processing and write `dest` ...
"""
import os
import sys


def _interactive():
    try:
        return bool(sys.stdin) and sys.stdin.isatty()
    except Exception:
        return False


def force_requested(argv=None):
    """True if the user asked to overwrite (via --force or COASTAL_FORCE=1)."""
    argv = sys.argv if argv is None else argv
    return ("--force" in argv) or (os.environ.get("COASTAL_FORCE") == "1")


def clean_argv(argv=None):
    """argv with the --force flag removed, so positional parsing is unaffected."""
    argv = sys.argv if argv is None else argv
    return [a for a in argv if a != "--force"]


def should_write(path, force=None, label=None, min_bytes=1):
    """Decide whether to (re)create `path`. See module docstring for the rules."""
    if force is None:
        force = force_requested()
    label = label or os.path.basename(path)
    exists = os.path.exists(path) and os.path.getsize(path) >= min_bytes
    if not exists:
        return True
    if force:
        print(f"  [overwrite] --force set, redoing {label}: {path}", flush=True)
        return True
    if _interactive():
        try:
            ans = input(f"  [exists] {label} already present:\n"
                        f"      {path}\n"
                        f"    re-download / re-process and OVERWRITE? [y/N] ").strip().lower()
        except EOFError:
            ans = ""
        if ans in ("y", "yes"):
            return True
        print(f"  [skip] keeping existing {label}", flush=True)
        return False
    print(f"  [skip] {label} already exists; pass --force (or COASTAL_FORCE=1) to redo: {path}",
          flush=True)
    return False
