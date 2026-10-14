"""Named numerical and population checks written beside each result's claims."""
import numpy as np
import pandas as pd


class Checks:
    def __init__(self):
        self.rows = []

    def close(self, name, actual, expected, *, atol=1e-10, rtol=0., source="", scope=""):
        a, b = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
        if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError(f"{name}: mismatched shapes or nonfinite values")
        delta = float(np.max(np.abs(a-b))) if a.size else 0.
        passed = bool(np.allclose(a, b, atol=atol, rtol=rtol))
        self.rows.append(dict(check=name, passed=passed, elements=a.size,
                              max_abs_error=delta, atol=atol, rtol=rtol, source=source, scope=scope))
        if not passed:
            raise ValueError(f"{name}: error {delta} exceeds declared tolerance")

    def require(self, name, condition, *, source="", scope=""):
        self.rows.append(dict(check=name, passed=bool(condition), elements=1,
                              max_abs_error=0 if condition else 1, atol=0, rtol=0,
                              source=source, scope=scope))
        if not condition:
            raise ValueError(f"{name}: scientific population/contract check failed")

    def save(self, output):
        if not self.rows:
            raise ValueError("a current paper result must declare validation checks")
        pd.DataFrame(self.rows).to_csv(output / "checks.csv", index=False)
        return len(self.rows)
