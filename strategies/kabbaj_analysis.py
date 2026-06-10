"""
Analyse technique inspirée de "L'Art du Trading" de Thami Kabbaj.
Couvre : chandeliers japonais, phases de marché, indicateurs avancés,
supports/résistances, money management et score de setup.
"""
import pandas as pd
import numpy as np


# ══════════════════════════════════════════════════════════════════════════════
# INDICATEURS TECHNIQUES
# ══════════════════════════════════════════════════════════════════════════════

def compute_macd(close: pd.Series, fast=12, slow=26, signal=9) -> pd.DataFrame:
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return pd.DataFrame({
        "macd": macd_line,
        "signal": signal_line,
        "histogram": histogram,
    }, index=close.index)


def compute_bollinger(close: pd.Series, period=20, std_dev=2) -> pd.DataFrame:
    ma = close.rolling(period).mean()
    std = close.rolling(period).std()
    return pd.DataFrame({
        "bb_upper": ma + std_dev * std,
        "bb_mid": ma,
        "bb_lower": ma - std_dev * std,
        "bb_width": (2 * std_dev * std) / ma * 100,  # bandwidth %
    }, index=close.index)


def compute_stochastic(high: pd.Series, low: pd.Series, close: pd.Series, k=14, d=3) -> pd.DataFrame:
    lowest_low = low.rolling(k).min()
    highest_high = high.rolling(k).max()
    stoch_k = 100 * (close - lowest_low) / (highest_high - lowest_low).replace(0, np.nan)
    stoch_d = stoch_k.rolling(d).mean()
    return pd.DataFrame({"stoch_k": stoch_k, "stoch_d": stoch_d}, index=close.index)


def compute_atr(high: pd.Series, low: pd.Series, close: pd.Series, period=14) -> pd.Series:
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def compute_adx(high: pd.Series, low: pd.Series, close: pd.Series, period=14) -> pd.DataFrame:
    """Average Directional Index — force de tendance."""
    tr = compute_atr(high, low, close, 1)
    dm_plus = ((high - high.shift()) > (low.shift() - low)).astype(float) * (high - high.shift()).clip(lower=0)
    dm_minus = ((low.shift() - low) > (high - high.shift())).astype(float) * (low.shift() - low).clip(lower=0)
    atr = tr.ewm(span=period, adjust=False).mean()
    di_plus = 100 * dm_plus.ewm(span=period, adjust=False).mean() / atr.replace(0, np.nan)
    di_minus = 100 * dm_minus.ewm(span=period, adjust=False).mean() / atr.replace(0, np.nan)
    dx = 100 * (di_plus - di_minus).abs() / (di_plus + di_minus).replace(0, np.nan)
    adx = dx.ewm(span=period, adjust=False).mean()
    return pd.DataFrame({"adx": adx, "di_plus": di_plus, "di_minus": di_minus}, index=close.index)


# ══════════════════════════════════════════════════════════════════════════════
# CHANDELIERS JAPONAIS (Kabbaj ch. 4-6)
# ══════════════════════════════════════════════════════════════════════════════

def detect_candlestick_patterns(df: pd.DataFrame, lookback: int = 5) -> list[dict]:
    """
    Détecte les patterns de chandeliers japonais sur les N dernières bougies.
    Retourne une liste de patterns détectés avec leur interprétation.
    """
    patterns = []
    recent = df.tail(lookback + 3).copy()

    o = recent["open"]
    h = recent["high"]
    l = recent["low"]
    c = recent["close"]
    body = (c - o).abs()
    candle_range = h - l
    avg_body = body.rolling(5).mean()

    for i in range(2, len(recent)):
        idx = recent.index[i]
        o0, h0, l0, c0 = o.iloc[i], h.iloc[i], l.iloc[i], c.iloc[i]
        o1, h1, l1, c1 = o.iloc[i-1], h.iloc[i-1], l.iloc[i-1], c.iloc[i-1]
        o2, h2, l2, c2 = o.iloc[i-2], h.iloc[i-2], l.iloc[i-2], c.iloc[i-2]
        body0 = abs(c0 - o0)
        body1 = abs(c1 - o1)
        range0 = h0 - l0
        avg = avg_body.iloc[i] if not pd.isna(avg_body.iloc[i]) else body0

        # ── Doji ──────────────────────────────────────────────────────────────
        if range0 > 0 and body0 / range0 < 0.1:
            patterns.append({
                "date": idx, "pattern": "Doji",
                "type": "neutral",
                "signal": "⚠️ Indécision — possible retournement",
                "color": "#ffa726",
                "candle_idx": i,
            })

        # ── Marteau (Hammer) ──────────────────────────────────────────────────
        lower_shadow = min(o0, c0) - l0
        upper_shadow = h0 - max(o0, c0)
        if (range0 > 0 and lower_shadow >= 2 * body0 and
                upper_shadow <= 0.1 * range0 and c1 < o1):
            patterns.append({
                "date": idx, "pattern": "Marteau (Hammer)",
                "type": "bullish",
                "signal": "🟢 Signal haussier — potentiel retournement bas",
                "color": "#26a69a",
                "candle_idx": i,
            })

        # ── Étoile filante (Shooting Star) ────────────────────────────────────
        if (range0 > 0 and upper_shadow >= 2 * body0 and
                lower_shadow <= 0.1 * range0 and c1 > o1):
            patterns.append({
                "date": idx, "pattern": "Étoile Filante",
                "type": "bearish",
                "signal": "🔴 Signal baissier — potentiel retournement haut",
                "color": "#ef5350",
                "candle_idx": i,
            })

        # ── Avalement haussier (Bullish Engulfing) ────────────────────────────
        if (c1 < o1 and c0 > o0 and
                o0 < c1 and c0 > o1 and
                body0 > body1 * 1.1):
            patterns.append({
                "date": idx, "pattern": "Avalement Haussier",
                "type": "bullish",
                "signal": "🟢 Fort signal haussier — les acheteurs reprennent le contrôle",
                "color": "#26a69a",
                "candle_idx": i,
            })

        # ── Avalement baissier (Bearish Engulfing) ────────────────────────────
        if (c1 > o1 and c0 < o0 and
                o0 > c1 and c0 < o1 and
                body0 > body1 * 1.1):
            patterns.append({
                "date": idx, "pattern": "Avalement Baissier",
                "type": "bearish",
                "signal": "🔴 Fort signal baissier — les vendeurs reprennent le contrôle",
                "color": "#ef5350",
                "candle_idx": i,
            })

        # ── Étoile du matin (Morning Star) ────────────────────────────────────
        if i >= 2:
            if (c2 < o2 and body1 < avg * 0.3 and c0 > o0 and
                    c0 > (o2 + c2) / 2):
                patterns.append({
                    "date": idx, "pattern": "Étoile du Matin",
                    "type": "bullish",
                    "signal": "🟢 Très fort signal haussier — retournement de tendance",
                    "color": "#26a69a",
                    "candle_idx": i,
                })

        # ── Étoile du soir (Evening Star) ─────────────────────────────────────
        if i >= 2:
            if (c2 > o2 and body1 < avg * 0.3 and c0 < o0 and
                    c0 < (o2 + c2) / 2):
                patterns.append({
                    "date": idx, "pattern": "Étoile du Soir",
                    "type": "bearish",
                    "signal": "🔴 Très fort signal baissier — retournement de tendance",
                    "color": "#ef5350",
                    "candle_idx": i,
                })

        # ── Marubozu haussier ─────────────────────────────────────────────────
        if (c0 > o0 and range0 > 0 and
                body0 / range0 > 0.9 and body0 > avg * 1.5):
            patterns.append({
                "date": idx, "pattern": "Marubozu Haussier",
                "type": "bullish",
                "signal": "🟢 Bougie de force — acheteurs dominants, pas d'hésitation",
                "color": "#26a69a",
                "candle_idx": i,
            })

        # ── Marubozu baissier ─────────────────────────────────────────────────
        if (c0 < o0 and range0 > 0 and
                body0 / range0 > 0.9 and body0 > avg * 1.5):
            patterns.append({
                "date": idx, "pattern": "Marubozu Baissier",
                "type": "bearish",
                "signal": "🔴 Bougie de faiblesse — vendeurs dominants",
                "color": "#ef5350",
                "candle_idx": i,
            })

    return patterns


# ══════════════════════════════════════════════════════════════════════════════
# PHASES DE MARCHÉ WYCKOFF (Kabbaj ch. 10)
# ══════════════════════════════════════════════════════════════════════════════

def detect_market_phase(df: pd.DataFrame) -> dict:
    """
    Identifie la phase de marché selon Wyckoff/Kabbaj :
    Accumulation → Markup → Distribution → Markdown
    """
    close = df["close"]
    volume = df["volume"] if "volume" in df.columns else pd.Series(1, index=df.index)

    ma20 = close.rolling(20).mean()
    ma50 = close.rolling(50).mean()
    ma200 = close.rolling(200).mean()

    price = close.iloc[-1]
    p_ma20 = ma20.iloc[-1]
    p_ma50 = ma50.iloc[-1]
    p_ma200 = ma200.iloc[-1]

    # Trend 3 months
    ret_3m = (price / close.iloc[-63] - 1) * 100 if len(close) >= 63 else 0
    ret_1m = (price / close.iloc[-22] - 1) * 100 if len(close) >= 22 else 0

    # Volume trend
    vol_recent = volume.tail(20).mean()
    vol_old = volume.tail(60).head(40).mean()
    vol_ratio = vol_recent / vol_old if vol_old > 0 else 1

    # ATR for volatility
    atr = compute_atr(df["high"], df["low"], close).iloc[-1]
    atr_pct = atr / price * 100

    # Bollinger squeeze (low volatility = potential breakout)
    bb = compute_bollinger(close)
    bb_width = bb["bb_width"].iloc[-1]
    bb_width_avg = bb["bb_width"].tail(50).mean()
    bb_squeeze = bb_width < bb_width_avg * 0.7

    # ADX
    adx_df = compute_adx(df["high"], df["low"], close)
    adx = adx_df["adx"].iloc[-1]
    di_plus = adx_df["di_plus"].iloc[-1]
    di_minus = adx_df["di_minus"].iloc[-1]

    # ── Phase detection ────────────────────────────────────────────────────────
    if price > p_ma20 > p_ma50 > p_ma200 and ret_3m > 10 and adx > 25:
        phase = "Markup (Tendance Haussière)"
        color = "#26a69a"
        emoji = "📈"
        description = (
            "Prix au-dessus de toutes les MM. Tendance haussière forte (ADX > 25). "
            "Phase idéale pour rester en position longue et laisser courir les gains."
        )
        advice = "Rester long. Renforcer sur les retracements vers MA20/MA50."

    elif price < p_ma20 < p_ma50 < p_ma200 and ret_3m < -10 and adx > 25:
        phase = "Markdown (Tendance Baissière)"
        color = "#ef5350"
        emoji = "📉"
        description = (
            "Prix sous toutes les MM. Tendance baissière confirmée. "
            "Phase à éviter pour les achats. Risque de continuation à la baisse."
        )
        advice = "Éviter les achats. Attendre un retournement confirmé."

    elif ret_3m < -15 and ret_1m > -3 and adx < 20:
        phase = "Accumulation"
        color = "#42a5f5"
        emoji = "🔄"
        description = (
            "Après une baisse significative, le prix se stabilise en zone basse. "
            "Les 'mains fortes' accumulent discrètement. Volume souvent croissant."
        )
        advice = "Zone d'intérêt pour un achat progressif. Attendre confirmation technique."

    elif ret_3m > 20 and ret_1m < 2 and adx < 20:
        phase = "Distribution"
        color = "#ffa726"
        emoji = "⚖️"
        description = (
            "Après une hausse importante, le prix plafonne. "
            "Les 'mains fortes' distribuent aux retail investors. "
            "Prudence — retournement possible."
        )
        advice = "Prendre des bénéfices partiels. Resserrer le stop-loss."

    elif bb_squeeze and adx < 20:
        phase = "Consolidation (Squeeze)"
        color = "#ab47bc"
        emoji = "⏸️"
        description = (
            "Faible volatilité, bandes de Bollinger resserrées. "
            "Un mouvement directionnel fort est imminent. "
            "Surveiller la direction du breakout."
        )
        advice = "Attendre le breakout. Préparer les deux scénarios (hausse/baisse)."

    else:
        phase = "Transition / Neutre"
        color = "#78909c"
        emoji = "➡️"
        description = "Signaux mixtes. Pas de tendance claire identifiée."
        advice = "Prudence. Réduire l'exposition ou rester en attente."

    return {
        "phase": phase,
        "color": color,
        "emoji": emoji,
        "description": description,
        "advice": advice,
        "adx": round(float(adx), 1) if not pd.isna(adx) else 0,
        "adx_interpretation": (
            "Tendance forte" if adx > 25 else
            "Tendance modérée" if adx > 20 else
            "Pas de tendance (range)"
        ),
        "di_plus": round(float(di_plus), 1) if not pd.isna(di_plus) else 0,
        "di_minus": round(float(di_minus), 1) if not pd.isna(di_minus) else 0,
        "bb_squeeze": bb_squeeze,
        "vol_ratio": round(vol_ratio, 2),
        "ret_3m": round(ret_3m, 1),
        "ret_1m": round(ret_1m, 1),
    }


# ══════════════════════════════════════════════════════════════════════════════
# SUPPORTS ET RÉSISTANCES (Kabbaj ch. 7)
# ══════════════════════════════════════════════════════════════════════════════

def find_support_resistance(df: pd.DataFrame, n_levels: int = 5) -> dict:
    """
    Identifie les niveaux de support et résistance clés
    par détection des pivots hauts/bas.
    """
    close = df["close"]
    high = df["high"]
    low = df["low"]
    price = float(close.iloc[-1])

    # Find pivot highs and lows (local extremes)
    pivot_highs = []
    pivot_lows = []
    window = 10

    for i in range(window, len(df) - window):
        if high.iloc[i] == high.iloc[i-window:i+window+1].max():
            pivot_highs.append(float(high.iloc[i]))
        if low.iloc[i] == low.iloc[i-window:i+window+1].min():
            pivot_lows.append(float(low.iloc[i]))

    # Cluster nearby levels
    def cluster_levels(levels, tolerance=0.02):
        if not levels:
            return []
        levels = sorted(set(levels))
        clusters = []
        group = [levels[0]]
        for lv in levels[1:]:
            if (lv - group[-1]) / group[-1] < tolerance:
                group.append(lv)
            else:
                clusters.append(round(np.mean(group), 2))
                group = [lv]
        clusters.append(round(np.mean(group), 2))
        return clusters

    resistances = [r for r in cluster_levels(pivot_highs) if r > price]
    supports = [s for s in cluster_levels(pivot_lows) if s < price]

    # Round numbers (psychological levels) — Kabbaj insiste dessus
    psychological = []
    base = round(price / 10) * 10
    for mult in [-2, -1, 0, 1, 2]:
        lvl = round(base + mult * 10, 2)
        if abs(lvl - price) / price < 0.15:
            psychological.append(lvl)

    nearest_support = max(supports[-3:], default=None) if supports else None
    nearest_resistance = min(resistances[:3], default=None) if resistances else None

    risk_reward = None
    if nearest_support and nearest_resistance:
        risk = price - nearest_support
        reward = nearest_resistance - price
        risk_reward = round(reward / risk, 2) if risk > 0 else None

    return {
        "supports": supports[-n_levels:],
        "resistances": resistances[:n_levels],
        "nearest_support": nearest_support,
        "nearest_resistance": nearest_resistance,
        "psychological_levels": psychological,
        "risk_reward_ratio": risk_reward,
        "price": price,
    }


# ══════════════════════════════════════════════════════════════════════════════
# MONEY MANAGEMENT (Kabbaj ch. 14-15)
# ══════════════════════════════════════════════════════════════════════════════

def compute_money_management(
    price: float,
    capital: float,
    stop_loss_price: float,
    risk_pct: float = 2.0,
    atr_value: float = None,
) -> dict:
    """
    Calcule la taille de position selon la règle des 2% de Kabbaj.
    risk_pct = % max du capital à risquer par trade (défaut 2%).
    """
    max_risk_amount = capital * risk_pct / 100
    risk_per_share = abs(price - stop_loss_price)

    if risk_per_share <= 0:
        return {"error": "Stop-loss invalide"}

    position_size = max_risk_amount / risk_per_share
    position_value = position_size * price
    position_pct = position_value / capital * 100

    # ATR-based stop suggestion
    atr_stop = round(price - 2 * atr_value, 2) if atr_value else None
    atr_stop_pct = round((price - atr_stop) / price * 100, 1) if atr_stop else None

    return {
        "max_risk_amount": round(max_risk_amount, 2),
        "risk_per_share": round(risk_per_share, 2),
        "position_size_shares": round(position_size, 2),
        "position_value": round(position_value, 2),
        "position_pct_capital": round(position_pct, 1),
        "stop_loss": round(stop_loss_price, 2),
        "stop_loss_pct": round((price - stop_loss_price) / price * 100, 1),
        "atr_stop_suggestion": atr_stop,
        "atr_stop_pct": atr_stop_pct,
        "rule": f"Règle des {risk_pct}% : ne jamais risquer plus de {risk_pct}% du capital par trade",
    }


# ══════════════════════════════════════════════════════════════════════════════
# SCORE DE SETUP KABBAJ (qualité du point d'entrée)
# ══════════════════════════════════════════════════════════════════════════════

def compute_setup_score(
    df: pd.DataFrame,
    market_phase: dict,
    sr_levels: dict,
    candlestick_patterns: list,
    macd_df: pd.DataFrame,
    bb_df: pd.DataFrame,
    stoch_df: pd.DataFrame,
) -> dict:
    """
    Score de qualité du setup selon les critères de Kabbaj :
    confluence de signaux = meilleur setup.
    """
    score = 0
    criteria = []
    close = df["close"]
    price = float(close.iloc[-1])

    # 1. Phase de marché
    if "Markup" in market_phase["phase"]:
        score += 20
        criteria.append(("✅ Phase Markup (tendance haussière)", 20, "bullish"))
    elif "Accumulation" in market_phase["phase"]:
        score += 15
        criteria.append(("✅ Phase Accumulation (potentiel retournement)", 15, "bullish"))
    elif "Distribution" in market_phase["phase"] or "Markdown" in market_phase["phase"]:
        score -= 15
        criteria.append(("❌ Phase Distribution/Markdown", -15, "bearish"))

    # 2. ADX — force de tendance
    adx = market_phase["adx"]
    if adx > 30:
        score += 15
        criteria.append((f"✅ Tendance forte (ADX={adx})", 15, "bullish"))
    elif adx > 20:
        score += 8
        criteria.append((f"⚠️ Tendance modérée (ADX={adx})", 8, "neutral"))
    else:
        criteria.append((f"❌ Pas de tendance (ADX={adx})", 0, "neutral"))

    # 3. MACD
    macd_val = macd_df["macd"].iloc[-1]
    macd_sig = macd_df["signal"].iloc[-1]
    macd_hist = macd_df["histogram"].iloc[-1]
    prev_hist = macd_df["histogram"].iloc[-2]
    if macd_val > macd_sig and macd_hist > 0:
        score += 15
        criteria.append(("✅ MACD au-dessus du signal (momentum haussier)", 15, "bullish"))
    elif macd_hist > prev_hist and macd_hist < 0:
        score += 8
        criteria.append(("⚠️ MACD histogramme en hausse (momentum qui s'améliore)", 8, "neutral"))
    elif macd_val < macd_sig and macd_hist < 0:
        score -= 10
        criteria.append(("❌ MACD sous le signal (momentum baissier)", -10, "bearish"))

    # 4. Bollinger Bands
    bb_upper = bb_df["bb_upper"].iloc[-1]
    bb_lower = bb_df["bb_lower"].iloc[-1]
    bb_mid = bb_df["bb_mid"].iloc[-1]
    if price > bb_mid and price < bb_upper * 0.98:
        score += 10
        criteria.append(("✅ Prix au-dessus de la MM20 (bande médiane)", 10, "bullish"))
    elif price <= bb_lower * 1.02:
        score += 12
        criteria.append(("✅ Prix sur bande basse Bollinger (zone d'achat potentielle)", 12, "bullish"))
    elif price >= bb_upper * 0.98:
        score -= 5
        criteria.append(("⚠️ Prix sur bande haute Bollinger (zone de résistance)", -5, "neutral"))

    # 5. Stochastique
    stoch_k = stoch_df["stoch_k"].iloc[-1]
    stoch_d = stoch_df["stoch_d"].iloc[-1]
    prev_k = stoch_df["stoch_k"].iloc[-2]
    if stoch_k < 25 and stoch_k > prev_k:
        score += 12
        criteria.append((f"✅ Stochastique en zone survente et remonte ({stoch_k:.0f})", 12, "bullish"))
    elif stoch_k > 80 and stoch_k < prev_k:
        score -= 10
        criteria.append((f"❌ Stochastique en zone surachat et baisse ({stoch_k:.0f})", -10, "bearish"))
    elif 40 < stoch_k < 60:
        score += 5
        criteria.append((f"⚠️ Stochastique neutre ({stoch_k:.0f})", 5, "neutral"))

    # 6. Patterns chandeliers récents (3 derniers jours)
    if candlestick_patterns:
        recent_patterns = [p for p in candlestick_patterns if p["candle_idx"] >= len(df) - 4]
        for p in recent_patterns[-2:]:
            if p["type"] == "bullish":
                score += 15
                criteria.append((f"✅ Pattern chandelier : {p['pattern']}", 15, "bullish"))
            elif p["type"] == "bearish":
                score -= 10
                criteria.append((f"❌ Pattern chandelier : {p['pattern']}", -10, "bearish"))
            elif p["type"] == "neutral":
                criteria.append((f"⚠️ Pattern chandelier : {p['pattern']}", 0, "neutral"))

    # 7. Support/résistance
    nearest_sup = sr_levels.get("nearest_support")
    rr = sr_levels.get("risk_reward_ratio")
    if nearest_sup and abs(price - nearest_sup) / price < 0.03:
        score += 10
        criteria.append((f"✅ Prix proche d'un support clé (${nearest_sup:.2f})", 10, "bullish"))
    if rr and rr >= 2:
        score += 10
        criteria.append((f"✅ Ratio Risk/Reward favorable ({rr}:1)", 10, "bullish"))
    elif rr and rr < 1:
        score -= 10
        criteria.append((f"❌ Ratio Risk/Reward défavorable ({rr}:1)", -10, "bearish"))

    # ── Final rating ───────────────────────────────────────────────────────────
    score = max(0, min(100, score))
    if score >= 70:
        rating = "⭐⭐⭐⭐⭐ Excellent setup"
        rating_color = "#26a69a"
        action = "CONDITIONS RÉUNIES POUR ENTRER EN POSITION"
    elif score >= 55:
        rating = "⭐⭐⭐⭐ Bon setup"
        rating_color = "#66bb6a"
        action = "Setup valide — respecter le money management"
    elif score >= 40:
        rating = "⭐⭐⭐ Setup moyen"
        rating_color = "#ffa726"
        action = "Attendre une meilleure confluence de signaux"
    elif score >= 25:
        rating = "⭐⭐ Setup faible"
        rating_color = "#ff7043"
        action = "Nombreux signaux contre-tendance — prudence"
    else:
        rating = "⭐ Setup à éviter"
        rating_color = "#ef5350"
        action = "NE PAS ENTRER EN POSITION — conditions défavorables"

    return {
        "score": score,
        "rating": rating,
        "rating_color": rating_color,
        "action": action,
        "criteria": criteria,
    }


# ══════════════════════════════════════════════════════════════════════════════
# ANALYSE COMPLÈTE
# ══════════════════════════════════════════════════════════════════════════════

def full_kabbaj_analysis(df: pd.DataFrame, capital: float = 10_000, risk_pct: float = 2.0) -> dict:
    close = df["close"]
    high = df["high"]
    low = df["low"]
    price = float(close.iloc[-1])

    macd_df = compute_macd(close)
    bb_df = compute_bollinger(close)
    stoch_df = compute_stochastic(high, low, close)
    atr_series = compute_atr(high, low, close)
    atr_val = float(atr_series.iloc[-1])

    market_phase = detect_market_phase(df)
    sr_levels = find_support_resistance(df)
    candlesticks = detect_candlestick_patterns(df, lookback=10)
    # Stop : sous le support le plus proche, mais jamais à plus de 2×ATR
    # du prix (un stop trop large détruit le ratio risque/gain)
    support_stop = sr_levels["nearest_support"]
    atr_stop = round(price - 2 * atr_val, 2)
    if support_stop and support_stop >= atr_stop:
        stop_loss = support_stop
    else:
        stop_loss = atr_stop
    stop_loss = min(stop_loss, round(price * 0.995, 2))  # toujours sous le prix
    mm = compute_money_management(price, capital, stop_loss, risk_pct, atr_val)
    setup_score = compute_setup_score(df, market_phase, sr_levels, candlesticks, macd_df, bb_df, stoch_df)

    return {
        "price": price,
        "macd": macd_df,
        "bollinger": bb_df,
        "stochastic": stoch_df,
        "atr": atr_series,
        "atr_value": round(atr_val, 2),
        "market_phase": market_phase,
        "support_resistance": sr_levels,
        "candlestick_patterns": candlesticks,
        "money_management": mm,
        "setup_score": setup_score,
    }


# ══════════════════════════════════════════════════════════════════════════════
# SCAN MULTI-TICKERS — Meilleures opportunités Kabbaj
# ══════════════════════════════════════════════════════════════════════════════

def scan_entry_signals(tickers: list[str], period: str = "2y", capital: float = 10_000, top_n: int = 10) -> list[dict]:
    """
    Scanne une liste de tickers avec l'analyse Kabbaj complète.
    Retourne les meilleures opportunités triées par score de setup.
    """
    from data.fetcher import fetch_ohlcv
    from data.fundamentals import get_fundamentals

    results = []
    for ticker in tickers:
        try:
            df = fetch_ohlcv(ticker, period)
            if len(df) < 210:
                continue
            kab = full_kabbaj_analysis(df, capital=capital, risk_pct=2.0)
            fund = get_fundamentals(ticker)
            score = kab["setup_score"]["score"]
            phase = kab["market_phase"]
            sr = kab["support_resistance"]
            mm = kab["money_management"]
            patterns = kab["candlestick_patterns"]
            recent_patterns = [p for p in patterns if p["candle_idx"] >= len(df) - 4]
            last_pattern = recent_patterns[-1]["pattern"] if recent_patterns else None

            # Stop/objectif cohérents : risque réel = prix - stop,
            # objectif = max(résistance, 2× le risque), R/R recalculé dessus
            price_k = kab["price"]
            stop_k = mm.get("stop_loss") or round(price_k * 0.95, 2)
            risk_k = max(price_k - stop_k, 0.01)
            resistance_k = sr["nearest_resistance"] or 0
            take_profit = round(max(resistance_k, price_k + 2 * risk_k), 2)
            rr_clean = round((take_profit - price_k) / risk_k, 2)
            target_pct = round((take_profit / price_k - 1) * 100, 1)

            # Only collect meaningful results
            results.append({
                "ticker": ticker,
                "name": fund.get("name", ticker)[:30],
                "sector": fund.get("sector", "N/A"),
                "price": kab["price"],
                "setup_score": score,
                "rating": kab["setup_score"]["rating"],
                "rating_color": kab["setup_score"]["rating_color"],
                "action": kab["setup_score"]["action"],
                "phase": phase["phase"],
                "phase_color": phase["color"],
                "phase_emoji": phase["emoji"],
                "adx": phase["adx"],
                "ret_1m": phase["ret_1m"],
                "ret_3m": phase["ret_3m"],
                "nearest_support": sr["nearest_support"],
                "nearest_resistance": sr["nearest_resistance"],
                "risk_reward": rr_clean,
                "take_profit": take_profit,
                "target_pct": target_pct,
                "stop_loss": mm.get("stop_loss"),
                "position_shares": mm.get("position_size_shares"),
                "position_value": mm.get("position_value"),
                "atr": kab["atr_value"],
                "last_pattern": last_pattern,
                "recommendation": (fund.get("recommendation") or "N/A").upper(),
                "analyst_target": fund.get("analyst_target"),
                "upside": round((fund["analyst_target"] / kab["price"] - 1) * 100, 1)
                          if fund.get("analyst_target") else None,
            })
        except Exception:
            continue

    results.sort(key=lambda x: x["setup_score"], reverse=True)
    return results[:top_n]
