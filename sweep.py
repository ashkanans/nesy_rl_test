"""
Compatibility entrypoint for sweep runner.

Canonical entrypoint: scripts/sweep.py
"""

from __future__ import annotations

import warnings

from scripts.sweep import main


if __name__ == "__main__":
    warnings.warn(
        "sweep.py at repo root is deprecated. Use scripts/sweep.py instead.",
        DeprecationWarning,
    )
    main()
