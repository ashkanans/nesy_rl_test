"""
Compatibility entrypoint for DT training.

Canonical entrypoint: scripts/train_dt.py
"""

from __future__ import annotations

import warnings

from scripts.train_dt import main


if __name__ == "__main__":
    warnings.warn(
        "train_dt.py at repo root is deprecated. Use scripts/train_dt.py instead.",
        DeprecationWarning,
    )
    main()
