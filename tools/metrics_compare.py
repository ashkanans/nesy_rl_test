from __future__ import annotations

import math


DEFAULT_EXCLUDE_KEYS = {"runtime_sec", "run_id", "timestamp_utc", "checkpoint_path"}


def compare_metrics(
    lhs: dict,
    rhs: dict,
    float_atol: float = 1e-8,
    exclude_keys: set[str] | None = None,
) -> list[str]:
    """
    Compare two metrics dictionaries with strict non-float checks and float tolerance.
    Returns a list of mismatch descriptions (empty means match).
    """
    errors: list[str] = []
    excludes = set(DEFAULT_EXCLUDE_KEYS)
    if exclude_keys:
        excludes.update(exclude_keys)

    keys = sorted(set(lhs.keys()) | set(rhs.keys()))
    for key in keys:
        if key in excludes:
            continue
        if key not in lhs:
            errors.append(f"Missing key in lhs: {key}")
            continue
        if key not in rhs:
            errors.append(f"Missing key in rhs: {key}")
            continue
        a = lhs[key]
        b = rhs[key]

        if isinstance(a, bool) or isinstance(b, bool):
            if bool(a) != bool(b):
                errors.append(f"{key}: {a!r} != {b!r}")
            continue

        if a is None or b is None:
            if a is not b:
                errors.append(f"{key}: {a!r} != {b!r}")
            continue

        if isinstance(a, (int, str)) and isinstance(b, type(a)):
            if a != b:
                errors.append(f"{key}: {a!r} != {b!r}")
            continue

        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            if math.isfinite(float(a)) and math.isfinite(float(b)):
                if abs(float(a) - float(b)) > float_atol:
                    errors.append(f"{key}: {a!r} != {b!r} (atol={float_atol})")
            elif float(a) != float(b):
                errors.append(f"{key}: {a!r} != {b!r}")
            continue

        if a != b:
            errors.append(f"{key}: {a!r} != {b!r}")

    return errors
