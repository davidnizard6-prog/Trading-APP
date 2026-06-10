import time
import yfinance as yf
import pandas as pd
from datetime import datetime


def _safe_ticker_info(ticker: str, retries: int = 2) -> dict:
    """Récupère le .info avec retry en cas de rate limit."""
    for attempt in range(retries):
        try:
            t = yf.Ticker(ticker)
            info = t.info or {}
            if info:
                return info
        except Exception:
            if attempt < retries - 1:
                time.sleep(1)
    return {}


def get_fundamentals(ticker: str) -> dict:
    """Retourne les fondamentaux — dict vide si Yahoo bloque."""
    try:
        info = _safe_ticker_info(ticker)
    except Exception:
        info = {}

    def safe(key, default=None):
        v = info.get(key, default)
        return v if v not in (None, "N/A", "Infinity", float("inf")) else default

    return {
        "name": safe("longName", ticker),
        "sector": safe("sector", "N/A"),
        "industry": safe("industry", "N/A"),
        "market_cap": safe("marketCap"),
        "pe_ratio": safe("trailingPE"),
        "forward_pe": safe("forwardPE"),
        "pb_ratio": safe("priceToBook"),
        "ps_ratio": safe("priceToSalesTrailing12Months"),
        "eps": safe("trailingEps"),
        "eps_growth": safe("earningsGrowth"),
        "revenue_growth": safe("revenueGrowth"),
        "profit_margin": safe("profitMargins"),
        "roe": safe("returnOnEquity"),
        "debt_to_equity": safe("debtToEquity"),
        "current_ratio": safe("currentRatio"),
        "dividend_yield": safe("dividendYield"),
        "52w_high": safe("fiftyTwoWeekHigh"),
        "52w_low": safe("fiftyTwoWeekLow"),
        "beta": safe("beta"),
        "analyst_target": safe("targetMeanPrice"),
        "recommendation": safe("recommendationKey", "N/A"),
        "num_analysts": safe("numberOfAnalystOpinions", 0),
    }


def get_news(ticker: str, max_items: int = 8) -> list[dict]:
    news = []
    try:
        t = yf.Ticker(ticker)
        raw = t.news or []
        for item in raw[:max_items]:
            content = item.get("content", {})
            title = content.get("title", item.get("title", ""))
            summary = content.get("summary", "")
            pub_date = content.get("pubDate", "")
            provider = content.get("provider", {})
            source = provider.get("displayName", "") if isinstance(provider, dict) else ""
            url = ""
            cta = content.get("canonicalUrl", {})
            if isinstance(cta, dict):
                url = cta.get("url", "")
            if title:
                news.append({
                    "title": title,
                    "summary": summary,
                    "date": pub_date[:10] if pub_date else "",
                    "source": source,
                    "url": url,
                })
    except Exception:
        pass
    return news


def get_macro_indicators() -> dict:
    """Fetch key macro indicators: VIX, 10Y yield, DXY, S&P500."""
    symbols = {"vix": "^VIX", "sp500": "^GSPC", "t10y": "^TNX", "dxy": "DX-Y.NYB"}
    result = {}
    for name, sym in symbols.items():
        try:
            df = yf.download(sym, period="6mo", interval="1d", auto_adjust=True, progress=False)
            if df.empty:
                continue
            close = df["Close"] if "Close" in df.columns else df.iloc[:, 0]
            if hasattr(close, "squeeze"):
                close = close.squeeze()
            last = float(close.iloc[-1])
            prev = float(close.iloc[-2]) if len(close) >= 2 else last
            prev_month = float(close.iloc[-22]) if len(close) >= 22 else float(close.iloc[0])
            result[name] = {
                "current": round(last, 2),
                "change_pct": round((last / prev - 1) * 100, 2),
                "change_1m": round((last / prev_month - 1) * 100, 2),
                "series": close,
            }
            time.sleep(0.3)
        except Exception:
            pass
    return result
