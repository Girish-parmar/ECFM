"""Create SYNTHETIC offline fixtures so the Part 1 notebooks can run without internet (tests, CI).

The numbers are made up. They have realistic shapes but are NOT real economic data.
"""
from pathlib import Path

import numpy as np
import pandas as pd

MONTHLY = ["CPIAUCSL", "PCEPILFE", "INDPRO", "UNRATE", "PAYEMS", "USREC"]
DAILY = ["T10Y2Y", "T10Y3M", "BAMLH0A0HYM2", "DTWEXBGS", "SP500", "DGS10", "DGS2"]


def make(folder: str | Path, seed: int = 7) -> Path:
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    m = pd.date_range("2000-01-01", "2025-06-01", freq="MS")
    d = pd.bdate_range("2015-01-02", "2025-06-30")
    n = len(m)
    rec = np.zeros(n); rec[(m >= "2001-04-01") & (m <= "2001-11-01")] = 1
    rec[(m >= "2008-01-01") & (m <= "2009-06-01")] = 1; rec[(m >= "2020-03-01") & (m <= "2020-04-01")] = 1
    infl = 0.2 + 0.15 * np.sin(np.arange(n) / 30) + rng.normal(0, 0.05, n)
    infl[(m >= "2021-03-01") & (m <= "2022-09-01")] += 0.5
    series = {
        "CPIAUCSL": 170 * np.exp(np.cumsum(infl / 100)),
        "PCEPILFE": 80 * np.exp(np.cumsum((infl * 0.8) / 100)),
        "INDPRO": 90 * np.exp(np.cumsum(0.15 / 100 - rec * 0.012 + rng.normal(0, 0.004, n))),
        "UNRATE": np.clip(4.5 + np.cumsum(rec * 0.35 - 0.03 + rng.normal(0, 0.08, n)), 3.4, 14.8),
        "PAYEMS": 130000 + np.cumsum(120 - rec * 450 + rng.normal(0, 60, n)),
        "USREC": rec,
    }
    for k, v in series.items():
        pd.DataFrame({k: np.round(v, 3)}, index=pd.Index(m, name="observation_date")).to_csv(folder / f"{k}.csv")
    nd = len(d)
    t10 = 2.2 + np.cumsum(rng.normal(0, 0.05, nd)); t2 = t10 - 0.8 + np.cumsum(rng.normal(0, 0.03, nd)) * 0.3
    daily = {
        "DGS10": t10, "DGS2": t2, "T10Y2Y": t10 - t2, "T10Y3M": t10 - t2 - 0.2,
        "BAMLH0A0HYM2": np.clip(4 + np.cumsum(rng.normal(0, 0.05, nd)), 2.5, 11),
        "DTWEXBGS": 110 * np.exp(np.cumsum(rng.normal(0, 0.003, nd))),
        "SP500": 2000 * np.exp(np.cumsum(rng.normal(0.0004, 0.011, nd))),
    }
    for k, v in daily.items():
        pd.DataFrame({k: np.round(v, 4)}, index=pd.Index(d, name="observation_date")).to_csv(folder / f"{k}.csv")
    # payrolls vintages: first release + two monthly revisions, then a benchmark revision
    rows, pay = [], series["PAYEMS"]
    for i, obs in enumerate(m[m >= "2019-01-01"]):
        j = list(m).index(obs)
        first_pub = obs + pd.offsets.MonthBegin(1) + pd.Timedelta(days=6)
        vals = [pay[j] + rng.normal(0, 90), pay[j] + rng.normal(0, 40), pay[j]]
        starts = [first_pub, first_pub + pd.DateOffset(months=1), first_pub + pd.DateOffset(months=2)]
        for k in range(3):
            end = starts[k + 1] - pd.Timedelta(days=1) if k < 2 else None
            rows.append({"date": obs.date().isoformat(), "value": f"{vals[k]:.0f}",
                         "realtime_start": starts[k].date().isoformat(),
                         "realtime_end": end.date().isoformat() if end is not None else "9999-12-31"})
    pd.DataFrame(rows).to_csv(folder / "PAYEMS_vintages.csv", index=False)
    return folder


if __name__ == "__main__":
    import sys
    print(make(sys.argv[1] if len(sys.argv) > 1 else "fixtures"))
