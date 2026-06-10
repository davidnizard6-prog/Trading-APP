import time
import yfinance as yf
import pandas as pd


def fetch_ohlcv(ticker: str, period: str = "5y", interval: str = "1d", retries: int = 3) -> pd.DataFrame:
    for attempt in range(retries):
        try:
            df = yf.download(ticker, period=period, interval=interval, auto_adjust=True, progress=False)
            if df.empty:
                raise ValueError(f"Aucune donnée pour {ticker}")
            df.columns = [c[0].lower() if isinstance(c, tuple) else c.lower() for c in df.columns]
            df.index.name = "date"
            return df.dropna()
        except Exception as e:
            if attempt < retries - 1:
                wait = 2 ** attempt  # 1s, 2s, 4s
                time.sleep(wait)
            else:
                raise e


def fetch_multiple(tickers: list[str], period: str = "5y") -> dict[str, pd.DataFrame]:
    results = {}
    for t in tickers:
        try:
            results[t] = fetch_ohlcv(t, period)
            time.sleep(0.3)  # pause entre chaque requête pour éviter le rate limit
        except Exception:
            pass
    return results
