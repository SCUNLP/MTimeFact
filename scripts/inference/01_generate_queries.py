
"""Run temporal fact-checking inference; see docs/INFERENCE.md."""
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

from mtimefact.inference.runner import stage_cli

if __name__ == "__main__":
    raise SystemExit(stage_cli('queries'))
