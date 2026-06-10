import yfinance as yf
import pandas as pd


def fetch_ohlcv(ticker: str, period: str = "5y", interval: str = "1d") -> pd.DataFrame:
    df = yf.download(ticker, period=period, interval=interval, auto_adjust=True, progress=False)
    df.columns = [c[0].lower() if isinstance(c, tuple) else c.lower() for c in df.columns]
    df.index.name = "date"
    return df.dropna()


def fetch_multiple(tickers: list[str], period: str = "5y") -> dict[str, pd.DataFrame]:
    return {t: fetch_ohlcv(t, period) for t in tickers}
