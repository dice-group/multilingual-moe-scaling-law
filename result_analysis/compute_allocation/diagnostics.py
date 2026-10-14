"""Decision margins within observed compute support."""
import numpy as np
import pandas as pd


def allocation_margins(frame):
    rows = frame.query("cap == 400000000 and method == 'smooth'")
    records = []
    for k, group in rows.groupby("K"):
        for budget in (2., 4.):
            values = []
            for recipe, part in group.groupby("config"):
                part = part.sort_values("F")
                if not part.F.min() <= budget <= part.F.max():
                    raise ValueError("margin query outside support")
                values.append((recipe, float(np.interp(np.log(budget), np.log(part.F), part.loss))))
            best = min(loss for _, loss in values)
            for recipe, loss in values:
                records.append(dict(K=k, F=budget, config=recipe, loss=loss, regret=loss-best,
                                    near_005=loss-best <= .005, near_010=loss-best <= .01))
    return pd.DataFrame(records)
