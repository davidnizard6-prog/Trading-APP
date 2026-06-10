import time
import requests
import pandas as pd
import yfinance as yf

# Session avec headers navigateur pour contourner le rate limit Yahoo Finance
_SESSION = requests.Session()
_SESSION.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
})


def fetch_ohlcv(ticker: str, period: str = "5y", interval: str = "1d", retries: int = 3) -> pd.DataFrame:
    for attempt in range(retries):
        try:
            t = yf.Ticker(ticker, session=_SESSION)
            df = t.history(period=period, interval=interval, auto_adjust=True)
            if df.empty:
                raise ValueError(f"Aucune donnée pour {ticker}")
            df.columns = [c.lower() for c in df.columns]
            df.index.name = "date"
            return df[["open", "high", "low", "close", "volume"]].dropna()
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                raise e


def fetch_multiple(tickers: list[str], period: str = "5y") -> dict[str, pd.DataFrame]:
    results = {}
    for t in tickers:
        try:
            results[t] = fetch_ohlcv(t, period)
            time.sleep(0.5)
        except Exception:
            pass
    return results
