"""
Compatibility entrypoint for baseline sweeps.

Canonical entrypoint: scripts/run_baselines.py
"""

from __future__ import annotations

import warnings

from scripts.run_baselines import main


if __name__ == "__main__":
    warnings.warn(
        "run_baselines.py at repo root is deprecated. Use scripts/run_baselines.py instead.",
        DeprecationWarning,
    )
    main()
