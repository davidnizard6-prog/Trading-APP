"""
Stock screener: finds best stocks right now + dip buying opportunities.
"""
import pandas as pd
import numpy as np
from data.fetcher import fetch_ohlcv
from data.fundamentals import get_fundamentals
from strategies.golden_cross import compute_signals
from strategies.scenario_engine import _rsi


def _historical_volatility(close, days: int = 252) -> float:
    import numpy as np
    returns = close.pct_change().dropna()
    return float(returns.tail(days).std() * np.sqrt(252))

# Curated universe — major liquid stocks across sectors
UNIVERSE = {
    "Tech": ["AAPL", "MSFT", "NVDA", "GOOGL", "META", "AMZN", "TSLA", "AMD", "INTC", "CRM", "ADBE", "ORCL"],
    "Finance": ["JPM", "BAC", "GS", "MS", "V", "MA", "BRK-B", "AXP", "BLK"],
    "Santé": ["JNJ", "UNH", "PFE", "ABBV", "MRK", "LLY", "TMO", "ABT"],
    "Conso/Retail": ["WMT", "COST", "TGT", "HD", "MCD", "SBUX", "NKE", "PG", "KO", "PEP"],
    "Énergie": ["XOM", "CVX", "COP", "EOG", "SLB"],
    "ETF": ["SPY", "QQQ", "IWM", "VTI", "GLD", "TLT", "XLK", "XLF"],
}

ALL_TICKERS = [t for tickers in UNIVERSE.values() for t in tickers]


def score_ticker(ticker: str, period: str = "2y") -> dict | None:
    try:
        df = fetch_ohlcv(ticker, period)
        if len(df) < 210:
            return None
        sig = compute_signals(df, 50, 200)
        fund = get_fundamentals(ticker)
        close = df["close"]
        price = float(close.iloc[-1])
        rsi = _rsi(close)
        vol = _historical_volatility(close)
        high_52 = fund.get("52w_high") or price
        low_52 = fund.get("52w_low") or price
        pct_from_high = (price / high_52 - 1) * 100
        pct_from_low = (price / low_52 - 1) * 100
        golden = int(sig["position"].iloc[-1]) == 1

        score = 0

        # Technical
        if golden:
            score += 25
        if rsi < 40:
            score += 20
        elif rsi < 55:
            score += 10
        elif rsi > 70:
            score -= 15

        # Momentum (1 month)
        ret_1m = (price / float(close.iloc[-22]) - 1) * 100 if len(close) >= 22 else 0
        if 2 < ret_1m < 15:
            score += 10
        elif ret_1m > 15:
            score += 5
        elif ret_1m < -10:
            score -= 10

        # Fundamental
        rec = fund.get("recommendation", "")
        if rec in ("buy", "strong_buy"):
            score += 20
        elif rec == "hold":
            score += 5
        elif rec in ("sell", "strong_sell"):
            score -= 20

        target = fund.get("analyst_target")
        if target:
            upside = (target / price - 1) * 100
            if upside > 20:
                score += 15
            elif upside > 10:
                score += 8
            elif upside < 0:
                score -= 10

        rev_growth = fund.get("revenue_growth") or 0
        if rev_growth > 0.15:
            score += 15
        elif rev_growth > 0.05:
            score += 8
        elif rev_growth < 0:
            score -= 10

        eps_growth = fund.get("eps_growth") or 0
        if eps_growth > 0.15:
            score += 10
        elif eps_growth > 0:
            score += 5
        elif eps_growth < -0.10:
            score -= 10

        pe = fund.get("pe_ratio")
        fpe = fund.get("forward_pe")
        if pe and fpe and pe > 0 and fpe > 0 and fpe < pe * 0.85:
            score += 10

        # 52w position bonus (not too extended)
        if -20 < pct_from_high < -5:
            score += 5

        return {
            "ticker": ticker,
            "name": fund.get("name", ticker),
            "sector": fund.get("sector", "N/A"),
            "price": round(price, 2),
            "score": score,
            "golden_cross": "✅" if golden else "❌",
            "rsi": rsi,
            "ret_1m": round(ret_1m, 1),
            "pct_from_high": round(pct_from_high, 1),
            "pct_from_low": round(pct_from_low, 1),
            "recommendation": rec.upper() if rec else "N/A",
            "analyst_target": target,
            "upside": round((target / price - 1) * 100, 1) if target else None,
            "rev_growth": round(rev_growth * 100, 1) if rev_growth else None,
            "pe": round(pe, 1) if pe else None,
            "div_yield": round((fund.get("dividend_yield") or 0) * 100, 2),
            "volatility": round(vol * 100, 1),
        }
    except Exception:
        return None


def find_best_stocks(tickers: list[str] = None, top_n: int = 10) -> pd.DataFrame:
    """Score and rank stocks. Returns top_n by score."""
    universe = tickers or ALL_TICKERS
    results = []
    for t in universe:
        r = score_ticker(t)
        if r:
            results.append(r)
    df = pd.DataFrame(results).sort_values("score", ascending=False)
    return df.head(top_n)


def find_dip_opportunities(tickers: list[str] = None, min_drop_pct: float = 10.0, top_n: int = 10) -> pd.DataFrame:
    """
    Find quality stocks that have dropped significantly from their 52w high
    but still have solid fundamentals — potential buying opportunities.
    """
    universe = tickers or ALL_TICKERS
    results = []
    for t in universe:
        r = score_ticker(t)
        if not r:
            continue
        drop = abs(r["pct_from_high"])
        if drop < min_drop_pct:
            continue
        # Only keep fundamentally decent stocks
        if r["recommendation"] in ("SELL", "STRONG_SELL"):
            continue
        # Quality filter: positive revenue growth or good analyst consensus
        if r["rev_growth"] is not None and r["rev_growth"] < -20:
            continue
        # Opportunity score: big drop + good fundamentals + not already rebounding too much
        opportunity_score = drop * 0.4 + r["score"] * 0.6
        r["drop_from_high"] = round(-r["pct_from_high"], 1)
        r["opportunity_score"] = round(opportunity_score, 1)
        results.append(r)

    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results).sort_values("opportunity_score", ascending=False)
    return df.head(top_n)
