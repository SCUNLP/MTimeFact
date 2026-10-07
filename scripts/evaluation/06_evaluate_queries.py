
"""Standalone queries evaluation (paper Stage 1)."""


if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

from mtimefact.evaluation.runner import stage_cli

if __name__ == "__main__":
    raise SystemExit(stage_cli("queries"))
