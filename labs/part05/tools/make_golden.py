"""Write TA-Lib reference outputs to golden/*.npz (instructors only; needs `pip install TA-Lib`).

Inputs are stored in each file too, so the tests never depend on regenerating the synthetic data.
Regenerate whenever the TA-Lib version changes:   python tools/make_golden.py
"""
import sys
from pathlib import Path

import numpy as np
import talib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common import GOLDEN, arrays, synthetic_ohlcv  # noqa: E402


def main() -> None:
    o, h, l, c, v = arrays(synthetic_ohlcv(1500, seed=7))
    inputs = {"open": o, "high": h, "low": l, "close": c, "volume": v}
    macd, sig, hist = talib.MACD(c, 12, 26, 9)
    up, mid, lo = talib.BBANDS(c, 20, 2.0, 2.0, 0)
    k, d = talib.STOCH(h, l, c, 14, 3, 0, 3, 0)
    core = {
        "sma20": talib.SMA(c, 20), "ema20": talib.EMA(c, 20), "rsi14": talib.RSI(c, 14), "atr14": talib.ATR(h, l, c, 14),
        "macd": macd, "macd_signal": sig, "macd_hist": hist, "bb_upper": up, "bb_middle": mid, "bb_lower": lo,
        "stoch_k": k, "stoch_d": d, "willr14": talib.WILLR(h, l, c, 14), "obv": talib.OBV(c, v),
        "engulfing": np.sign(talib.CDLENGULFING(o, h, l, c)).astype(float),
    }
    groups = {
        "kama10": talib.KAMA(c, 10), "adx14": talib.ADX(h, l, c, 14), "plus_di14": talib.PLUS_DI(h, l, c, 14),
        "minus_di14": talib.MINUS_DI(h, l, c, 14), "cci20": talib.CCI(h, l, c, 20), "roc10": talib.ROC(c, 10),
        "mfi14": talib.MFI(h, l, c, v, 14), "ad": talib.AD(h, l, c, v),
    }
    GOLDEN.mkdir(exist_ok=True)
    np.savez_compressed(GOLDEN / "core.npz", **inputs, **core)
    np.savez_compressed(GOLDEN / "groups.npz", **inputs, **groups)
    (GOLDEN / "VERSION").write_text(f"TA-Lib python {talib.__version__}; synthetic_ohlcv(1500, seed=7)\n")
    print("wrote", sorted(p.name for p in GOLDEN.iterdir()))


if __name__ == "__main__":
    main()
