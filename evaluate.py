"""
Compatibility entrypoint for evaluation.

Canonical entrypoint: scripts/evaluate.py
"""

from __future__ import annotations

import warnings

from scripts.evaluate import main


if __name__ == "__main__":
    warnings.warn(
        "evaluate.py at repo root is deprecated. Use scripts/evaluate.py instead.",
        DeprecationWarning,
    )
    main()
