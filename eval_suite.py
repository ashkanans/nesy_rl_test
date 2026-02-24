"""
Compatibility entrypoint for suite evaluation runner.

Canonical entrypoint: scripts/eval_suite.py
"""

from __future__ import annotations

import warnings

from scripts.eval_suite import main


if __name__ == "__main__":
    warnings.warn(
        "eval_suite.py at repo root is deprecated. Use scripts/eval_suite.py instead.",
        DeprecationWarning,
    )
    main()
