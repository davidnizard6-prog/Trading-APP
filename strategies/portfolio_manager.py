"""
Gestion de portefeuille intelligente : synthétise toutes les analyses
(Golden Cross, Kabbaj, Wyckoff, alertes, fondamentaux) pour chaque position
et propose un remaniement complet du portefeuille.
"""
import pandas as pd

from data.fetcher import fetch_ohlcv
from data.fundamentals import get_fundamentals
from strategies.golden_cross import compute_signals
from strategies.kabbaj_analysis import full_kabbaj_analysis
from alerts.alert_engine import check_long_term_signals, check_short_term_signals


def _analyze_position(ticker: str, qty: float, avg_price: float | None, capital: float) -> dict:
    """Analyse complète d'une position : retourne santé 0-100 + recommandation."""
    df = fetch_ohlcv(ticker, "2y")
    price = float(df["close"].iloc[-1])
    value = qty * price
    pnl_pct = round((price / avg_price - 1) * 100, 1) if avg_price else None

    # ── Collecte de tous les signaux ──────────────────────────────────────────
    kab = full_kabbaj_analysis(df, capital=capital)
    setup_score = kab["setup_score"]["score"]
    phase = kab["market_phase"]

    gc = compute_signals(df, 50, 200)
    golden_active = int(gc["position"].iloc[-1]) == 1

    lt_signals = check_long_term_signals(ticker, df)
    st_signals = check_short_term_signals(ticker, df)
    has_exit = any(s["type"] == "SORTIE" for s in lt_signals + st_signals)
    has_entry = any(s["type"] in ("ENTRÉE", "CASSURE") for s in lt_signals + st_signals)

    fund = get_fundamentals(ticker)
    sector = fund.get("sector", "N/A")

    # ── Score de santé 0-100 ──────────────────────────────────────────────────
    health = 50
    reasons = []

    if golden_active:
        health += 15
        reasons.append("✅ Golden Cross actif (tendance LT haussière)")
    else:
        health -= 15
        reasons.append("❌ Sous la MA200 (tendance LT baissière)")

    if phase["phase"] in ("Markup (Tendance Haussière)", "Accumulation"):
        health += 15
        reasons.append(f"✅ Phase {phase['phase']}")
    elif phase["phase"] in ("Distribution", "Markdown (Tendance Baissière)"):
        health -= 20
        reasons.append(f"⚠️ Phase {phase['phase']} — les vendeurs dominent")

    health += round((setup_score - 50) * 0.4)
    if setup_score >= 60:
        reasons.append(f"✅ Setup Kabbaj solide ({setup_score}/100)")
    elif setup_score <= 35:
        reasons.append(f"⚠️ Setup Kabbaj faible ({setup_score}/100)")

    if has_exit:
        health -= 15
        reasons.append("🔴 Signal de sortie actif (alertes)")
    if has_entry:
        health += 10
        reasons.append("🟢 Signal d'entrée/cassure actif")

    if pnl_pct is not None and pnl_pct < -15:
        health -= 10
        reasons.append(f"⚠️ Perte importante ({pnl_pct}%) — réévaluer la thèse")

    health = max(0, min(100, health))

    # ── Recommandation ────────────────────────────────────────────────────────
    if health >= 70:
        action, action_color, action_emoji = "RENFORCER", "#26a69a", "💪"
    elif health >= 50:
        action, action_color, action_emoji = "CONSERVER", "#42a5f5", "🤝"
    elif health >= 30:
        action, action_color, action_emoji = "RÉDUIRE", "#ffa726", "✂️"
    else:
        action, action_color, action_emoji = "VENDRE", "#ef5350", "🚪"

    return {
        "ticker": ticker,
        "name": fund.get("name", ticker),
        "sector": sector,
        "price": price,
        "quantity": qty,
        "value": round(value, 2),
        "avg_price": avg_price,
        "pnl_pct": pnl_pct,
        "health": health,
        "action": action,
        "action_color": action_color,
        "action_emoji": action_emoji,
        "reasons": reasons,
        "setup_score": setup_score,
        "phase": phase["phase"],
        "golden_cross": golden_active,
        "stop_loss": kab["money_management"].get("stop_loss"),
        "n_alerts_exit": sum(1 for s in lt_signals + st_signals if s["type"] == "SORTIE"),
        "n_alerts_entry": sum(1 for s in lt_signals + st_signals if s["type"] in ("ENTRÉE", "CASSURE")),
    }


def analyze_portfolio(positions: dict, capital: float = 10_000) -> dict:
    """
    positions: {ticker: {"quantity": x, "avg_price": y}}
    Retourne l'analyse complète + propositions de remaniement.
    """
    analyses = []
    errors = []
    for ticker, pos in positions.items():
        try:
            qty = pos.get("quantity", 1) if isinstance(pos, dict) else float(pos)
            avg = pos.get("avg_price") if isinstance(pos, dict) else None
            analyses.append(_analyze_position(ticker, qty, avg, capital))
        except Exception as e:
            errors.append({"ticker": ticker, "error": str(e)})

    if not analyses:
        return {"positions": [], "errors": errors, "summary": None}

    total_value = sum(a["value"] for a in analyses)

    # ── Allocation par secteur + concentration ────────────────────────────────
    sectors = {}
    for a in analyses:
        a["weight_pct"] = round(a["value"] / total_value * 100, 1) if total_value else 0
        sectors[a["sector"]] = sectors.get(a["sector"], 0) + a["weight_pct"]

    warnings = []
    for a in analyses:
        if a["weight_pct"] > 35:
            warnings.append(f"⚠️ **{a['ticker']}** représente {a['weight_pct']}% du portefeuille — concentration excessive (max conseillé : 25%)")
    for sec, w in sectors.items():
        if w > 50 and len(analyses) > 2:
            warnings.append(f"⚠️ Le secteur **{sec}** représente {w:.0f}% du portefeuille — diversifier")

    # ── Plan de remaniement ───────────────────────────────────────────────────
    to_sell = [a for a in analyses if a["action"] == "VENDRE"]
    to_reduce = [a for a in analyses if a["action"] == "RÉDUIRE"]
    to_strengthen = [a for a in analyses if a["action"] == "RENFORCER"]

    freed_cash = sum(a["value"] for a in to_sell) + sum(a["value"] * 0.5 for a in to_reduce)

    rebalance_plan = []
    for a in to_sell:
        rebalance_plan.append(f"🚪 **Vendre {a['ticker']}** ({a['quantity']:.1f} actions ≈ {a['value']:,.0f}$) — santé {a['health']}/100")
    for a in to_reduce:
        rebalance_plan.append(f"✂️ **Réduire {a['ticker']} de moitié** (récupérer ≈ {a['value']*0.5:,.0f}$) — santé {a['health']}/100")
    for a in to_strengthen:
        rebalance_plan.append(f"💪 **Renforcer {a['ticker']}** (santé {a['health']}/100, setup {a['setup_score']}/100) — utiliser le cash libéré")
    if freed_cash > 0 and not to_strengthen:
        rebalance_plan.append(f"💰 Garder les {freed_cash:,.0f}$ libérés en cash et utiliser l'onglet 💡 Opportunités pour trouver de nouveaux titres")

    avg_health = round(sum(a["health"] for a in analyses) / len(analyses))

    return {
        "positions": sorted(analyses, key=lambda x: x["health"]),
        "errors": errors,
        "summary": {
            "total_value": round(total_value, 2),
            "n_positions": len(analyses),
            "avg_health": avg_health,
            "sectors": sectors,
            "warnings": warnings,
            "rebalance_plan": rebalance_plan,
            "freed_cash": round(freed_cash, 2),
            "n_sell": len(to_sell),
            "n_reduce": len(to_reduce),
            "n_strengthen": len(to_strengthen),
        },
    }
