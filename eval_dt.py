"""
Compatibility entrypoint for DT evaluation.

Canonical entrypoint: scripts/eval_dt.py
"""

from __future__ import annotations

import warnings

from scripts.eval_dt import main


if __name__ == "__main__":
    warnings.warn(
        "eval_dt.py at repo root is deprecated. Use scripts/eval_dt.py instead.",
        DeprecationWarning,
    )
    main()
