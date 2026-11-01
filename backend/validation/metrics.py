"""Onset-detection metrics: one-to-one nearest matching, P/R/F1 per tolerance, median/p95 error."""
import numpy as np

TOLERANCES_S = (0.010, 0.025, 0.050, 0.100)


def match(detected, reference, tol):
    """Greedy nearest one-to-one matching. Returns the list of absolute errors of matched pairs."""
    det, ref = sorted(detected), sorted(reference)
    pairs = sorted(
        ((abs(d - r), i, j) for i, d in enumerate(det) for j, r in enumerate(ref) if abs(d - r) <= tol)
    )
    used_d, used_r, errs = set(), set(), []
    for err, i, j in pairs:
        if i not in used_d and j not in used_r:
            used_d.add(i), used_r.add(j), errs.append(err)
    return errs


def score(detected, reference, tolerances=TOLERANCES_S):
    out = {"n_detected": len(detected), "n_reference": len(reference), "by_tolerance_ms": {}}
    for tol in tolerances:
        errs = match(detected, reference, tol)
        tp = len(errs)
        p = tp / len(detected) if detected else 0.0
        r = tp / len(reference) if reference else 0.0
        out["by_tolerance_ms"][round(tol * 1000)] = {
            "tp": tp, "fp": len(detected) - tp, "fn": len(reference) - tp,
            "precision": round(p, 3), "recall": round(r, 3),
            "f1": round(2 * p * r / (p + r), 3) if p + r else 0.0,
        }
    errs = match(detected, reference, max(tolerances))
    out["median_error_ms"] = round(float(np.median(errs)) * 1000, 2) if errs else None
    out["p95_error_ms"] = round(float(np.percentile(errs, 95)) * 1000, 2) if errs else None
    return out
