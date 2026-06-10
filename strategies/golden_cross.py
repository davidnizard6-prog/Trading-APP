import pandas as pd
import numpy as np


def compute_signals(df: pd.DataFrame, fast: int = 50, slow: int = 200) -> pd.DataFrame:
    out = df.copy()
    out["ma_fast"] = out["close"].rolling(fast).mean()
    out["ma_slow"] = out["close"].rolling(slow).mean()

    # 1 = long, 0 = flat
    out["position"] = np.where(out["ma_fast"] > out["ma_slow"], 1, 0)

    # signal: +1 golden cross entry, -1 death cross exit
    out["signal"] = out["position"].diff()

    return out.dropna()
