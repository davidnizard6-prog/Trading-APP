"""
Moteur d'alertes : surveille les positions du portefeuille et génère
des signaux d'entrée/sortie en temps réel, distinguant long et court terme.
"""
import json
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

from data.fetcher import fetch_ohlcv
from strategies.golden_cross import compute_signals
from strategies.kabbaj_analysis import (
    compute_macd, compute_bollinger, compute_stochastic,
    compute_atr, detect_candlestick_patterns, detect_market_phase,
)

ALERTS_FILE = Path(__file__).parent.parent / "alerts_history.json"


def _load_alerts() -> list:
    if ALERTS_FILE.exists():
        return json.loads(ALERTS_FILE.read_text())
    return []


def _save_alerts(alerts: list):
    ALERTS_FILE.write_text(json.dumps(alerts, indent=2, default=str))


def _rsi(series: pd.Series, period: int = 14) -> float:
    delta = series.diff().dropna()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return float((100 - 100 / (1 + rs)).iloc[-1])


# ══════════════════════════════════════════════════════════════════════════════
# SIGNAUX LONG TERME
# ══════════════════════════════════════════════════════════════════════════════

def check_long_term_signals(ticker: str, df: pd.DataFrame) -> list[dict]:
    """
    Signaux long terme (semaines/mois) :
    - Golden Cross / Death Cross MA50/200
    - Phase Wyckoff (Markup / Distribution)
    - ADX > 25 avec tendance confirmée
    - Cassure de résistance majeure
    """
    signals = []
    close = df["close"]
    price = float(close.iloc[-1])

    sig = compute_signals(df, 50, 200)
    current_pos = int(sig["position"].iloc[-1])
    prev_pos = int(sig["position"].iloc[-2]) if len(sig) > 1 else current_pos
    signal_change = int(sig["signal"].iloc[-1])

    # Golden Cross / Death Cross
    if signal_change == 1:
        signals.append({
            "ticker": ticker, "timeframe": "LONG TERME",
            "type": "ENTRÉE", "urgency": "high",
            "color": "#26a69a", "emoji": "🚀",
            "title": f"GOLDEN CROSS détecté sur {ticker}",
            "message": "MA50 croise MA200 par le haut. Signal d'achat long terme majeur selon Kabbaj.",
            "action": "ENTRER EN POSITION LONGUE",
            "price": price,
        })
    elif signal_change == -1:
        signals.append({
            "ticker": ticker, "timeframe": "LONG TERME",
            "type": "SORTIE", "urgency": "high",
            "color": "#ef5350", "emoji": "🔴",
            "title": f"DEATH CROSS détecté sur {ticker}",
            "message": "MA50 croise MA200 par le bas. Signal de sortie long terme majeur.",
            "action": "SORTIR DE POSITION",
            "price": price,
        })

    # Phase de marché
    phase = detect_market_phase(df)
    if phase["phase"] == "Distribution" and current_pos == 1:
        signals.append({
            "ticker": ticker, "timeframe": "LONG TERME",
            "type": "ALERTE", "urgency": "medium",
            "color": "#ffa726", "emoji": "⚠️",
            "title": f"Phase Distribution sur {ticker}",
            "message": "Le marché entre en phase de distribution. Les mains fortes vendent. Resserrer le stop.",
            "action": "RESSERRER LE STOP-LOSS",
            "price": price,
        })
    elif phase["phase"] == "Accumulation" and current_pos == 0:
        signals.append({
            "ticker": ticker, "timeframe": "LONG TERME",
            "type": "OPPORTUNITÉ", "urgency": "low",
            "color": "#42a5f5", "emoji": "🔄",
            "title": f"Zone d'Accumulation sur {ticker}",
            "message": "Phase d'accumulation détectée. Les mains fortes achètent discrètement.",
            "action": "SURVEILLER — entrée progressive possible",
            "price": price,
        })

    # ADX trend confirmation
    if phase["adx"] > 30 and phase["di_plus"] > phase["di_minus"] and current_pos == 1:
        signals.append({
            "ticker": ticker, "timeframe": "LONG TERME",
            "type": "CONFIRMATION", "urgency": "low",
            "color": "#26a69a", "emoji": "✅",
            "title": f"Tendance forte confirmée sur {ticker}",
            "message": f"ADX={phase['adx']} — Tendance haussière puissante. Laisser courir les gains.",
            "action": "MAINTENIR LA POSITION",
            "price": price,
        })

    # 52-week high breakout
    high_52 = float(df["high"].tail(252).max())
    if price >= high_52 * 0.995:
        signals.append({
            "ticker": ticker, "timeframe": "LONG TERME",
            "type": "CASSURE", "urgency": "high",
            "color": "#ab47bc", "emoji": "💥",
            "title": f"Cassure du plus haut 52 semaines sur {ticker}",
            "message": f"Prix ({price:.2f}$) au plus haut annuel. Breakout majeur — continuation probable.",
            "action": "RENFORCER OU ENTRER EN POSITION",
            "price": price,
        })

    return signals


# ══════════════════════════════════════════════════════════════════════════════
# SIGNAUX COURT TERME
# ══════════════════════════════════════════════════════════════════════════════

def check_short_term_signals(ticker: str, df: pd.DataFrame) -> list[dict]:
    """
    Signaux court terme (1-10 jours) :
    - Croisement MACD
    - RSI zones extrêmes
    - Stochastique croisement
    - Patterns chandeliers récents
    - Bollinger squeeze/breakout
    """
    signals = []
    close = df["close"]
    high = df["high"]
    low = df["low"]
    price = float(close.iloc[-1])

    macd_df = compute_macd(close)
    stoch_df = compute_stochastic(high, low, close)
    bb_df = compute_bollinger(close)
    rsi = _rsi(close)
    patterns = detect_candlestick_patterns(df, lookback=3)

    # MACD crossover
    macd_curr = macd_df["macd"].iloc[-1]
    macd_prev = macd_df["macd"].iloc[-2]
    sig_curr = macd_df["signal"].iloc[-1]
    sig_prev = macd_df["signal"].iloc[-2]

    if macd_prev < sig_prev and macd_curr > sig_curr:
        signals.append({
            "ticker": ticker, "timeframe": "COURT TERME",
            "type": "ENTRÉE", "urgency": "medium",
            "color": "#26a69a", "emoji": "📈",
            "title": f"Croisement MACD haussier sur {ticker}",
            "message": f"MACD vient de croiser le signal par le haut. Momentum court terme positif.",
            "action": "SIGNAL D'ACHAT COURT TERME",
            "price": price,
        })
    elif macd_prev > sig_prev and macd_curr < sig_curr:
        signals.append({
            "ticker": ticker, "timeframe": "COURT TERME",
            "type": "SORTIE", "urgency": "medium",
            "color": "#ef5350", "emoji": "📉",
            "title": f"Croisement MACD baissier sur {ticker}",
            "message": "MACD vient de croiser le signal par le bas. Momentum court terme négatif.",
            "action": "SIGNAL DE VENTE COURT TERME",
            "price": price,
        })

    # RSI extremes
    if rsi < 30:
        signals.append({
            "ticker": ticker, "timeframe": "COURT TERME",
            "type": "OPPORTUNITÉ", "urgency": "medium",
            "color": "#26a69a", "emoji": "🛒",
            "title": f"RSI en zone de survente sur {ticker} (RSI={rsi:.0f})",
            "message": f"RSI={rsi:.1f} — Zone de survente extrême. Rebond court terme probable.",
            "action": "OPPORTUNITÉ D'ACHAT COURT TERME",
            "price": price,
        })
    elif rsi > 75:
        signals.append({
            "ticker": ticker, "timeframe": "COURT TERME",
            "type": "ALERTE", "urgency": "medium",
            "color": "#ffa726", "emoji": "🔥",
            "title": f"RSI en zone de surachat sur {ticker} (RSI={rsi:.0f})",
            "message": f"RSI={rsi:.1f} — Zone de surachat. Possible consolidation ou correction.",
            "action": "PRENDRE DES BÉNÉFICES PARTIELS",
            "price": price,
        })

    # Stochastic crossover
    k_curr = stoch_df["stoch_k"].iloc[-1]
    d_curr = stoch_df["stoch_d"].iloc[-1]
    k_prev = stoch_df["stoch_k"].iloc[-2]
    d_prev = stoch_df["stoch_d"].iloc[-2]

    if k_prev < d_prev and k_curr > d_curr and k_curr < 30:
        signals.append({
            "ticker": ticker, "timeframe": "COURT TERME",
            "type": "ENTRÉE", "urgency": "medium",
            "color": "#26a69a", "emoji": "🎯",
            "title": f"Croisement Stochastique haussier en survente sur {ticker}",
            "message": f"Stochastique %K croise %D en zone de survente ({k_curr:.0f}). Fort signal d'achat.",
            "action": "SIGNAL D'ACHAT COURT TERME FORT",
            "price": price,
        })
    elif k_prev > d_prev and k_curr < d_curr and k_curr > 75:
        signals.append({
            "ticker": ticker, "timeframe": "COURT TERME",
            "type": "SORTIE", "urgency": "medium",
            "color": "#ef5350", "emoji": "🎯",
            "title": f"Croisement Stochastique baissier en surachat sur {ticker}",
            "message": f"Stochastique %K croise %D en zone de surachat ({k_curr:.0f}). Signal de vente.",
            "action": "RÉDUIRE OU SORTIR LA POSITION",
            "price": price,
        })

    # Bollinger breakout
    bb_upper = bb_df["bb_upper"].iloc[-1]
    bb_lower = bb_df["bb_lower"].iloc[-1]
    bb_width = bb_df["bb_width"].iloc[-1]
    bb_width_avg = bb_df["bb_width"].tail(20).mean()

    if price > bb_upper and bb_width > bb_width_avg:
        signals.append({
            "ticker": ticker, "timeframe": "COURT TERME",
            "type": "CASSURE", "urgency": "high",
            "color": "#ab47bc", "emoji": "💥",
            "title": f"Cassure bande haute Bollinger sur {ticker}",
            "message": f"Prix ({price:.2f}$) sort au-dessus de la bande haute. Momentum fort.",
            "action": "TENDANCE FORTE — SUIVRE LE MOUVEMENT",
            "price": price,
        })
    elif price < bb_lower:
        signals.append({
            "ticker": ticker, "timeframe": "COURT TERME",
            "type": "ALERTE", "urgency": "high",
            "color": "#ef5350", "emoji": "⚠️",
            "title": f"Cassure bande basse Bollinger sur {ticker}",
            "message": f"Prix ({price:.2f}$) sous la bande basse. Forte pression vendeuse.",
            "action": "STOP-LOSS À SURVEILLER",
            "price": price,
        })

    # Candlestick patterns (last 2 days only)
    recent = [p for p in patterns if p["candle_idx"] >= len(df) - 3]
    for p in recent[-2:]:
        urgency = "high" if p["pattern"] in ("Avalement Haussier", "Avalement Baissier",
                                               "Étoile du Matin", "Étoile du Soir") else "low"
        entry_type = "ENTRÉE" if p["type"] == "bullish" else "SORTIE" if p["type"] == "bearish" else "ALERTE"
        color = "#26a69a" if p["type"] == "bullish" else "#ef5350" if p["type"] == "bearish" else "#ffa726"
        signals.append({
            "ticker": ticker, "timeframe": "COURT TERME",
            "type": entry_type, "urgency": urgency,
            "color": color, "emoji": "🕯️",
            "title": f"Pattern chandelier : {p['pattern']} sur {ticker}",
            "message": p["signal"],
            "action": "SURVEILLER LA CONFIRMATION",
            "price": price,
        })

    return signals


# ══════════════════════════════════════════════════════════════════════════════
# SCAN DU PORTEFEUILLE
# ══════════════════════════════════════════════════════════════════════════════

def scan_portfolio_alerts(positions: dict, period: str = "2y") -> dict:
    """
    positions: {ticker: {"quantity": x, "avg_price": y}} or paper broker format.
    Retourne les alertes long terme et court terme par ticker.
    """
    results = {}
    timestamp = datetime.now().isoformat()

    for ticker in positions:
        try:
            df = fetch_ohlcv(ticker, period)
            lt_signals = check_long_term_signals(ticker, df)
            st_signals = check_short_term_signals(ticker, df)
            price = float(df["close"].iloc[-1])

            pos_data = positions[ticker]
            qty = pos_data if isinstance(pos_data, (int, float)) else pos_data.get("quantity", 0)
            avg = pos_data.get("avg_price") if isinstance(pos_data, dict) else None
            pnl_pct = round((price / avg - 1) * 100, 2) if avg else None

            results[ticker] = {
                "price": price,
                "quantity": qty,
                "avg_price": avg,
                "pnl_pct": pnl_pct,
                "long_term": lt_signals,
                "short_term": st_signals,
                "total_alerts": len(lt_signals) + len(st_signals),
                "has_entry": any(s["type"] in ("ENTRÉE", "CASSURE") for s in lt_signals + st_signals),
                "has_exit": any(s["type"] == "SORTIE" for s in lt_signals + st_signals),
                "scanned_at": timestamp,
            }
        except Exception as e:
            results[ticker] = {"error": str(e), "ticker": ticker}

    # Persist alerts
    history = _load_alerts()
    history.append({"timestamp": timestamp, "scan": results})
    history = history[-50:]  # keep last 50 scans
    _save_alerts(history)

    return results


def get_alert_history() -> list:
    return _load_alerts()
