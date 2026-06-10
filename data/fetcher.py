import time
import requests
import pandas as pd
import yfinance as yf

_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept": "application/json",
    "Accept-Language": "en-US,en;q=0.9",
}


def _fetch_chart_api(ticker: str, period: str = "5y", interval: str = "1d") -> pd.DataFrame:
    """Appel direct à l'API chart de Yahoo (moins bloquée que yfinance)."""
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
    params = {"range": period, "interval": interval, "events": "div,split"}
    r = requests.get(url, params=params, headers=_HEADERS, timeout=15)
    r.raise_for_status()
    data = r.json()

    result = data.get("chart", {}).get("result")
    if not result:
        raise ValueError(f"Aucune donnée pour {ticker}")
    result = result[0]

    timestamps = result.get("timestamp", [])
    quote = result["indicators"]["quote"][0]
    adjclose = result["indicators"].get("adjclose", [{}])[0].get("adjclose")

    df = pd.DataFrame({
        "open": quote.get("open", []),
        "high": quote.get("high", []),
        "low": quote.get("low", []),
        "close": adjclose if adjclose else quote.get("close", []),
        "volume": quote.get("volume", []),
    }, index=pd.to_datetime(timestamps, unit="s"))
    df.index.name = "date"
    return df.dropna()


def fetch_ohlcv(ticker: str, period: str = "5y", interval: str = "1d", retries: int = 3) -> pd.DataFrame:
    """API chart Yahoo directe avec retry, puis yfinance en secours."""
    last_error = None
    for attempt in range(retries):
        try:
            return _fetch_chart_api(ticker, period, interval)
        except Exception as e:
            last_error = e
            time.sleep(1.5 ** attempt)

    # Secours : yfinance classique
    try:
        df = yf.download(ticker, period=period, interval=interval, auto_adjust=True, progress=False)
        if df.empty:
            raise ValueError(f"Aucune donnée pour {ticker}")
        df.columns = [c[0].lower() if isinstance(c, tuple) else c.lower() for c in df.columns]
        df.index.name = "date"
        return df.dropna()
    except Exception:
        raise last_error


def fetch_multiple(tickers: list[str], period: str = "5y") -> dict[str, pd.DataFrame]:
    results = {}
    for t in tickers:
        try:
            results[t] = fetch_ohlcv(t, period)
            time.sleep(0.3)
        except Exception:
            pass
    return results
