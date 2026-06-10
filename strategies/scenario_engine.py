"""
Scenario engine: combines technical + fundamental + macro signals
to produce Bull / Base / Bear scenarios with probabilities and return estimates.
Also produces 3 investment scenarios: Risky / Moderate / Safe.
"""
import pandas as pd
import numpy as np


def _rsi(series: pd.Series, period: int = 14) -> float:
    delta = series.diff().dropna()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return round(float(rsi.iloc[-1]), 1)


def _historical_volatility(close: pd.Series, days: int = 252) -> float:
    returns = close.pct_change().dropna()
    return float(returns.tail(days).std() * np.sqrt(252))


def _trend_strength(close: pd.Series) -> str:
    ma20 = close.rolling(20).mean().iloc[-1]
    ma50 = close.rolling(50).mean().iloc[-1]
    ma200 = close.rolling(200).mean().iloc[-1]
    price = close.iloc[-1]
    if price > ma20 > ma50 > ma200:
        return "forte hausse"
    elif price > ma50 > ma200:
        return "hausse modérée"
    elif price < ma20 < ma50 < ma200:
        return "forte baisse"
    elif price < ma50 < ma200:
        return "baisse modérée"
    else:
        return "neutre"


def compute_scenarios(df_signals: pd.DataFrame, fundamentals: dict, macro: dict) -> dict:
    close = df_signals["close"]
    scores = {"bull": 0, "base": 0, "bear": 0}
    signals_log = []

    # ── Technical signals ──────────────────────────────────────────────────────
    golden_cross_active = int(df_signals["position"].iloc[-1]) == 1
    if golden_cross_active:
        scores["bull"] += 2
        signals_log.append(("✅ Golden Cross actif", "haussier", 2))
    else:
        scores["bear"] += 2
        signals_log.append(("❌ Death Cross actif", "baissier", 2))

    rsi = _rsi(close)
    if rsi < 35:
        scores["bull"] += 1.5
        signals_log.append((f"📉 RSI survendu ({rsi})", "haussier (rebond probable)", 1.5))
    elif rsi > 70:
        scores["bear"] += 1.5
        signals_log.append((f"📈 RSI suracheté ({rsi})", "baissier (correction probable)", 1.5))
    else:
        scores["base"] += 1
        signals_log.append((f"➡️ RSI neutre ({rsi})", "neutre", 1))

    trend = _trend_strength(close)
    if "forte hausse" in trend:
        scores["bull"] += 2
        signals_log.append((f"📊 Tendance : {trend}", "haussier", 2))
    elif "hausse" in trend:
        scores["bull"] += 1
        signals_log.append((f"📊 Tendance : {trend}", "légèrement haussier", 1))
    elif "forte baisse" in trend:
        scores["bear"] += 2
        signals_log.append((f"📊 Tendance : {trend}", "baissier", 2))
    elif "baisse" in trend:
        scores["bear"] += 1
        signals_log.append((f"📊 Tendance : {trend}", "légèrement baissier", 1))
    else:
        scores["base"] += 1
        signals_log.append((f"📊 Tendance : {trend}", "neutre", 1))

    high_52 = fundamentals.get("52w_high")
    low_52 = fundamentals.get("52w_low")
    price_now = float(close.iloc[-1])
    if high_52 and low_52:
        pct_from_high = (price_now / high_52 - 1) * 100
        pct_from_low = (price_now / low_52 - 1) * 100
        if pct_from_high > -5:
            scores["bull"] += 1
            signals_log.append((f"🔝 Proche du plus haut 52s ({pct_from_high:.1f}%)", "haussier", 1))
        elif pct_from_low < 20:
            scores["bear"] += 1
            signals_log.append((f"🔻 Proche du plus bas 52s (+{pct_from_low:.1f}%)", "baissier", 1))

    # ── Fundamental signals ────────────────────────────────────────────────────
    pe = fundamentals.get("pe_ratio")
    forward_pe = fundamentals.get("forward_pe")
    if pe and forward_pe and pe > 0 and forward_pe > 0:
        if forward_pe < pe * 0.9:
            scores["bull"] += 1.5
            signals_log.append((f"💰 PER forward ({forward_pe:.1f}) < PER actuel ({pe:.1f})", "haussier (bénéfices en hausse)", 1.5))
        elif forward_pe > pe * 1.1:
            scores["bear"] += 1
            signals_log.append((f"⚠️ PER forward ({forward_pe:.1f}) > PER actuel ({pe:.1f})", "baissier (bénéfices en baisse)", 1))

    rev_growth = fundamentals.get("revenue_growth")
    if rev_growth is not None:
        if rev_growth > 0.15:
            scores["bull"] += 1.5
            signals_log.append((f"📈 Croissance revenus : +{rev_growth*100:.1f}%", "haussier", 1.5))
        elif rev_growth > 0:
            scores["bull"] += 0.5
            signals_log.append((f"📈 Croissance revenus : +{rev_growth*100:.1f}%", "légèrement haussier", 0.5))
        else:
            scores["bear"] += 1
            signals_log.append((f"📉 Croissance revenus : {rev_growth*100:.1f}%", "baissier", 1))

    eps_growth = fundamentals.get("eps_growth")
    if eps_growth is not None:
        if eps_growth > 0.10:
            scores["bull"] += 1
            signals_log.append((f"💵 Croissance EPS : +{eps_growth*100:.1f}%", "haussier", 1))
        elif eps_growth < 0:
            scores["bear"] += 1
            signals_log.append((f"💵 Croissance EPS : {eps_growth*100:.1f}%", "baissier", 1))

    rec = fundamentals.get("recommendation", "")
    analyst_target = fundamentals.get("analyst_target")
    if rec in ("buy", "strong_buy"):
        scores["bull"] += 1.5
        signals_log.append((f"🏦 Consensus analystes : {rec.upper()}", "haussier", 1.5))
    elif rec in ("sell", "strong_sell"):
        scores["bear"] += 1.5
        signals_log.append((f"🏦 Consensus analystes : {rec.upper()}", "baissier", 1.5))
    elif rec == "hold":
        scores["base"] += 1
        signals_log.append(("🏦 Consensus analystes : HOLD", "neutre", 1))

    if analyst_target and price_now:
        upside = (analyst_target / price_now - 1) * 100
        if upside > 15:
            scores["bull"] += 1
            signals_log.append((f"🎯 Objectif analystes : ${analyst_target:.2f} (+{upside:.1f}%)", "haussier", 1))
        elif upside < -10:
            scores["bear"] += 1
            signals_log.append((f"🎯 Objectif analystes : ${analyst_target:.2f} ({upside:.1f}%)", "baissier", 1))

    # ── Macro signals ──────────────────────────────────────────────────────────
    vix = macro.get("VIX", {})
    if vix:
        v = vix["value"]
        if v < 15:
            scores["bull"] += 1
            signals_log.append((f"😌 VIX bas ({v}) — faible volatilité", "haussier", 1))
        elif v > 30:
            scores["bear"] += 2
            signals_log.append((f"😱 VIX élevé ({v}) — forte peur du marché", "baissier", 2))
        elif v > 20:
            scores["bear"] += 0.5
            signals_log.append((f"😟 VIX modéré ({v})", "légèrement baissier", 0.5))

    sp500 = macro.get("SP500", {})
    if sp500:
        chg = sp500["change_1m"]
        if chg > 3:
            scores["bull"] += 1
            signals_log.append((f"🌍 S&P500 +{chg:.1f}% sur 1 mois", "haussier (marché porteur)", 1))
        elif chg < -5:
            scores["bear"] += 1.5
            signals_log.append((f"🌍 S&P500 {chg:.1f}% sur 1 mois", "baissier (marché en recul)", 1.5))

    t10y = macro.get("T10Y", {})
    if t10y:
        rate = t10y["value"]
        chg = t10y["change_1m"]
        if rate > 5 and chg > 0:
            scores["bear"] += 1
            signals_log.append((f"📊 Taux 10 ans : {rate:.2f}% (en hausse)", "baissier (coût du capital élevé)", 1))
        elif rate < 4 or chg < -0.3:
            scores["bull"] += 0.5
            signals_log.append((f"📊 Taux 10 ans : {rate:.2f}% (en baisse)", "légèrement haussier", 0.5))

    # ── Compute probabilities ──────────────────────────────────────────────────
    total = scores["bull"] + scores["base"] + scores["bear"]
    if total == 0:
        probs = {"bull": 33, "base": 34, "bear": 33}
    else:
        raw = {k: max(v / total, 0.05) for k, v in scores.items()}
        s = sum(raw.values())
        probs = {k: round(v / s * 100) for k, v in raw.items()}
        diff = 100 - sum(probs.values())
        probs["base"] += diff

    dominant = max(probs, key=probs.get)

    # ── Return estimates ───────────────────────────────────────────────────────
    vol = _historical_volatility(close)
    target = fundamentals.get("analyst_target")
    upside_analyst = ((target / price_now - 1) * 100) if target and price_now else None
    div_yield = (fundamentals.get("dividend_yield") or 0) * 100

    bull_return = _estimate_return("bull", probs, upside_analyst, vol, div_yield, scores)
    base_return = _estimate_return("base", probs, upside_analyst, vol, div_yield, scores)
    bear_return = _estimate_return("bear", probs, upside_analyst, vol, div_yield, scores)

    scenarios = {
        "bull": {
            "label": "Scénario Haussier (Bull)",
            "probability": probs["bull"],
            "color": "#26a69a",
            "estimated_return": bull_return,
            "description": _bull_description(fundamentals, close, macro, bull_return),
        },
        "base": {
            "label": "Scénario Neutre (Base)",
            "probability": probs["base"],
            "color": "#ffa726",
            "estimated_return": base_return,
            "description": _base_description(fundamentals, close, base_return),
        },
        "bear": {
            "label": "Scénario Baissier (Bear)",
            "probability": probs["bear"],
            "color": "#ef5350",
            "estimated_return": bear_return,
            "description": _bear_description(fundamentals, close, macro, bear_return),
        },
    }

    # ── 3 Investment scenarios ─────────────────────────────────────────────────
    investment_scenarios = _build_investment_scenarios(
        probs, vol, upside_analyst, div_yield, fundamentals, close
    )

    return {
        "scenarios": scenarios,
        "investment_scenarios": investment_scenarios,
        "signals": signals_log,
        "dominant": dominant,
        "scores": scores,
        "rsi": rsi,
        "volatility": round(vol * 100, 1),
    }


def _estimate_return(scenario: str, probs: dict, upside_analyst, vol: float, div_yield: float, scores: dict) -> float:
    base_analyst = upside_analyst if upside_analyst else 0
    if scenario == "bull":
        # Optimistic: analyst target + momentum bonus + dividends
        momentum_bonus = min(scores["bull"] * 2, 20)
        ret = max(base_analyst, 10) + momentum_bonus * 0.5 + div_yield
        ret = min(ret, 80)  # cap
    elif scenario == "base":
        ret = base_analyst * 0.5 if base_analyst else 5
        ret = max(ret, -5) + div_yield
        ret = min(ret, 25)
    else:  # bear
        drawdown_est = -vol * 100 * 0.6
        ret = min(drawdown_est, -5) + div_yield
        ret = max(ret, -60)
    return round(ret, 1)


def _build_investment_scenarios(probs, vol, upside_analyst, div_yield, fundamentals, close) -> list[dict]:
    price = float(close.iloc[-1])
    beta = fundamentals.get("beta") or 1.0
    pe = fundamentals.get("pe_ratio")
    rec = fundamentals.get("recommendation", "")
    target = fundamentals.get("analyst_target")

    # ── Scénario Risqué — maximum upside, higher volatility ───────────────────
    risky_base = (upside_analyst or 15) * 1.4
    risky_return = round(min(risky_base + vol * 50, 120), 1)
    risky_min = round(-vol * 80, 1)

    # ── Scénario Modéré — analyst consensus, partial exposure ─────────────────
    moderate_return = round((upside_analyst or 10) * 0.7 + div_yield, 1)
    moderate_min = round(-vol * 40, 1)

    # ── Scénario Défensif — dividends + capital preservation ──────────────────
    safe_return = round(div_yield + 3 + (2 if rec in ("buy", "strong_buy") else 0), 1)
    safe_min = round(-vol * 20, 1)

    return [
        {
            "label": "🚀 Scénario Risqué",
            "color": "#ef5350",
            "emoji": "🚀",
            "estimated_return": f"+{risky_return}%",
            "worst_case": f"{risky_min}%",
            "probability": min(probs["bull"] + 10, 45),
            "horizon": "3-6 mois",
            "strategy": "Position pleine, levier ou options d'achat",
            "condition": "Golden Cross confirmé + momentum fort + VIX bas",
            "description": (
                f"Pari agressif sur la continuation de la tendance haussière. "
                f"Rendement potentiel estimé à **+{risky_return}%** si les catalyseurs se matérialisent. "
                f"Risque de perte max estimé à **{risky_min}%**. "
                f"Convient uniquement si vous acceptez une forte volatilité."
            ),
            "stop_loss": round(price * 0.90, 2),
            "take_profit": round(price * (1 + risky_return / 100), 2),
        },
        {
            "label": "📈 Scénario Modéré",
            "color": "#ffa726",
            "emoji": "📈",
            "estimated_return": f"+{moderate_return}%",
            "worst_case": f"{moderate_min}%",
            "probability": probs["base"] + probs["bull"] // 2,
            "horizon": "6-12 mois",
            "strategy": "Position partielle (50-70%), renforcement progressif",
            "condition": "Tendance haussière + fondamentaux solides",
            "description": (
                f"Approche équilibrée alignée sur le consensus des analystes. "
                f"Rendement potentiel estimé à **+{moderate_return}%** sur 12 mois. "
                f"Risque de perte max estimé à **{moderate_min}%**. "
                f"Idéal pour un portefeuille diversifié avec gestion du risque."
            ),
            "stop_loss": round(price * 0.93, 2),
            "take_profit": round(price * (1 + moderate_return / 100), 2),
        },
        {
            "label": "🛡️ Scénario Défensif",
            "color": "#42a5f5",
            "emoji": "🛡️",
            "estimated_return": f"+{safe_return}%",
            "worst_case": f"{safe_min}%",
            "probability": 85,
            "horizon": "12-24 mois",
            "strategy": f"Position faible (20-30%), focus dividendes ({div_yield:.1f}%/an)",
            "condition": "Capital préservation prioritaire",
            "description": (
                f"Approche conservatrice basée sur les dividendes et la résilience. "
                f"Rendement estimé à **+{safe_return}%/an** (dividendes inclus). "
                f"Risque de perte max estimé à **{safe_min}%**. "
                f"Convient pour un capital que vous ne pouvez pas vous permettre de perdre."
            ),
            "stop_loss": round(price * 0.95, 2),
            "take_profit": round(price * (1 + safe_return / 100), 2),
        },
    ]


def _bull_description(f, close, macro, ret):
    price = float(close.iloc[-1])
    target = f.get("analyst_target")
    upside = f"Objectif potentiel : ${target:.2f} (+{(target/price-1)*100:.1f}%)" if target else ""
    return (f"Les indicateurs techniques et fondamentaux sont alignés positivement. "
            f"Rendement estimé : **+{ret}%** sur 12 mois. {upside}")


def _base_description(f, close, ret):
    pe = f.get("pe_ratio")
    pe_txt = f"PER actuel : {pe:.1f}x." if pe else ""
    sign = "+" if ret >= 0 else ""
    return (f"Signaux mitigés. Évolution latérale probable à court terme. "
            f"Rendement estimé : **{sign}{ret}%** sur 12 mois. {pe_txt}")


def _bear_description(f, close, macro, ret):
    vix = macro.get("VIX", {}).get("value")
    vix_txt = f"VIX à {vix} — volatilité élevée." if vix and vix > 20 else ""
    return (f"Plusieurs alertes détectées. Correction possible. "
            f"Rendement estimé : **{ret}%** sur 12 mois. {vix_txt}")
