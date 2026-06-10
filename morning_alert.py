"""
Alerte matinale automatique via Telegram.
Lancé chaque matin par GitHub Actions à 7h30 (Paris).
Envoie : score marché, alertes portefeuille, top Kabbaj, scénarios du jour.
"""
import os
import sys
import requests
from datetime import datetime

# ── Ajouter le répertoire racine au path pour les imports ──────────────────────
sys.path.insert(0, os.path.dirname(__file__))

from data.fetcher import fetch_ohlcv, fetch_multiple
from data.fundamentals import get_fundamentals, get_macro_indicators
from strategies.kabbaj_analysis import full_kabbaj_analysis, scan_entry_signals
from strategies.stock_screener import find_best_stocks, find_dip_opportunities, UNIVERSE
from strategies.scenario_engine import compute_scenarios
from strategies.golden_cross import compute_signals

# ── Configuration Telegram (secrets GitHub Actions) ────────────────────────────
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")

# Portefeuille à surveiller (modifiable selon vos positions réelles)
PORTFOLIO = os.environ.get("PORTFOLIO_TICKERS", "AAPL,MSFT,NVDA,BTC-USD,ETH-USD")
CAPITAL = float(os.environ.get("CAPITAL", "10000"))

# Top tickers à analyser chaque matin
TOP_TICKERS = ["AAPL", "MSFT", "NVDA", "TSLA", "AMZN", "META", "GOOGL", "BTC-USD", "ETH-USD"]


def send_telegram(message: str):
    """Envoie un message Telegram (supporte le Markdown)."""
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("⚠️  TELEGRAM_TOKEN ou TELEGRAM_CHAT_ID manquant")
        print(message)
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "Markdown",
        "disable_web_page_preview": True,
    }
    r = requests.post(url, json=payload, timeout=15)
    if r.status_code != 200:
        print(f"Erreur Telegram: {r.text}")


def score_market(macro: dict) -> tuple[int, str]:
    """Calcule un score de marché global 0-100."""
    score = 50
    commentary = []

    vix = macro.get("vix", {}).get("current", 20)
    if vix < 15:
        score += 15
        commentary.append("VIX bas → calme")
    elif vix < 20:
        score += 8
        commentary.append("VIX modéré")
    elif vix > 30:
        score -= 20
        commentary.append("VIX élevé → peur")
    elif vix > 25:
        score -= 10
        commentary.append("VIX tendu")

    sp500_chg = macro.get("sp500", {}).get("change_pct", 0)
    if sp500_chg > 1:
        score += 10
        commentary.append("SP500 en hausse")
    elif sp500_chg > 0:
        score += 5
    elif sp500_chg < -1:
        score -= 15
        commentary.append("SP500 en baisse")
    elif sp500_chg < 0:
        score -= 5

    score = max(0, min(100, score))

    if score >= 70:
        mood = "🟢 RISK-ON"
    elif score >= 50:
        mood = "🟡 NEUTRE"
    elif score >= 30:
        mood = "🟠 PRUDENT"
    else:
        mood = "🔴 RISK-OFF"

    return score, mood, commentary


def format_morning_report() -> str:
    """Construit le message Telegram complet."""
    now = datetime.now().strftime("%d/%m/%Y %H:%M")
    lines = [f"📊 *RAPPORT TRADING — {now}*", ""]

    # ── 1. SCORE DE MARCHÉ ─────────────────────────────────────────────────────
    try:
        macro = get_macro_indicators()
        score, mood, commentary = score_market(macro)
        vix = macro.get("vix", {}).get("current", "N/A")
        sp_chg = macro.get("sp500", {}).get("change_pct", 0)

        lines.append("━━━━━━━━━━━━━━━━━━━━")
        lines.append(f"🌍 *SCORE MARCHÉ : {score}/100 — {mood}*")
        lines.append(f"VIX: {vix:.1f}  |  SP500: {sp_chg:+.2f}%")
        if commentary:
            lines.append("_" + " · ".join(commentary) + "_")
        lines.append("")
    except Exception as e:
        lines.append(f"_(Macro indisponible: {e})_\n")

    # ── 2. ALERTES PORTEFEUILLE ────────────────────────────────────────────────
    portfolio_tickers = [t.strip() for t in PORTFOLIO.split(",") if t.strip()]
    lines.append("━━━━━━━━━━━━━━━━━━━━")
    lines.append("📋 *VOTRE PORTEFEUILLE*")

    for ticker in portfolio_tickers:
        try:
            df = fetch_ohlcv(ticker, "6mo")
            price = float(df["close"].iloc[-1])
            chg = float((df["close"].iloc[-1] / df["close"].iloc[-2] - 1) * 100)
            chg_icon = "📈" if chg > 0 else "📉"
            lines.append(f"{chg_icon} *{ticker}*: ${price:.2f} ({chg:+.2f}%)")

            # Signal Golden Cross rapide
            sig = compute_signals(df, 50, 200)
            if int(sig["signal"].iloc[-1]) == 1:
                lines.append("  ✅ _Golden Cross détecté — signal ACHAT long terme_")
            elif int(sig["signal"].iloc[-1]) == -1:
                lines.append("  ❌ _Death Cross détecté — signal VENTE long terme_")
        except Exception as e:
            lines.append(f"  _{ticker}: erreur ({e})_")

    lines.append("")

    # ── 3. ANALYSE KABBAJ — TOP OPPORTUNITÉS DU JOUR ──────────────────────────
    lines.append("━━━━━━━━━━━━━━━━━━━━")
    lines.append("🎯 *TOP SETUPS KABBAJ DU JOUR*")

    try:
        entry_signals = scan_entry_signals(TOP_TICKERS, period="6mo", capital=CAPITAL, top_n=3)
        if entry_signals:
            for sig in entry_signals[:3]:
                ticker = sig["ticker"]
                score_k = sig["setup_score"]
                phase = sig["market_phase"]
                price = sig["price"]
                sl = sig.get("stop_loss", 0)
                tp = sig.get("take_profit", 0)
                risk_reward = sig.get("risk_reward", 0)

                stars = "⭐" * min(5, int(score_k / 20))
                lines.append(f"\n*{ticker}* {stars} Score: {score_k:.0f}/100")
                lines.append(f"  Prix: ${price:.2f} | Phase: {phase}")
                if sl and tp:
                    sl_pct = (sl / price - 1) * 100
                    tp_pct = (tp / price - 1) * 100
                    lines.append(f"  🛑 Stop: ${sl:.2f} ({sl_pct:+.1f}%) | 🎯 Cible: ${tp:.2f} ({tp_pct:+.1f}%)")
                if risk_reward:
                    lines.append(f"  R/R: {risk_reward:.1f}x")
        else:
            lines.append("_Aucun setup fort détecté aujourd'hui_")
    except Exception as e:
        lines.append(f"_(Kabbaj indisponible: {e})_")

    lines.append("")

    # ── 4. SCÉNARIOS DU JOUR ───────────────────────────────────────────────────
    lines.append("━━━━━━━━━━━━━━━━━━━━")
    lines.append("📈 *SCÉNARIOS DU JOUR (NVDA)*")

    try:
        df_nvda = fetch_ohlcv("NVDA", "1y")
        fund_nvda = get_fundamentals("NVDA")
        sig_nvda = compute_signals(df_nvda)
        scenarios = compute_scenarios(sig_nvda, fund_nvda, macro)

        bull = scenarios.get("bull", {})
        base = scenarios.get("base", {})
        bear = scenarios.get("bear", {})

        lines.append(f"🟢 Taureau ({bull.get('probability', 0):.0f}%): +{bull.get('estimated_return', 0):.1f}%")
        lines.append(f"🟡 Base    ({base.get('probability', 0):.0f}%): {base.get('estimated_return', 0):+.1f}%")
        lines.append(f"🔴 Ours    ({bear.get('probability', 0):.0f}%): {bear.get('estimated_return', 0):+.1f}%")

        # Gain espéré sur 10K€
        expected = (
            bull.get("probability", 0) / 100 * bull.get("estimated_return", 0) +
            base.get("probability", 0) / 100 * base.get("estimated_return", 0) +
            bear.get("probability", 0) / 100 * bear.get("estimated_return", 0)
        )
        gain_euro = CAPITAL * expected / 100
        lines.append(f"\n💰 *Gain espéré sur {CAPITAL:.0f}€ : {gain_euro:+.0f}€ ({expected:+.1f}%)*")
    except Exception as e:
        lines.append(f"_(Scénarios indisponibles: {e})_")

    lines.append("")

    # ── 5. DIPS À ACHETER ─────────────────────────────────────────────────────
    lines.append("━━━━━━━━━━━━━━━━━━━━")
    lines.append("🛒 *DIPS À SURVEILLER*")

    try:
        dips = find_dip_opportunities(list(UNIVERSE.keys())[:20], min_drop_pct=8, top_n=3)
        if dips:
            for d in dips[:3]:
                lines.append(f"  📉 *{d['ticker']}*: -{d['drop_pct']:.1f}% depuis le sommet | Score: {d['score']:.0f}")
        else:
            lines.append("_Aucun dip de qualité détecté_")
    except Exception as e:
        lines.append(f"_(Dips indisponibles: {e})_")

    lines.append("")
    lines.append("━━━━━━━━━━━━━━━━━━━━")
    lines.append("_Bonne journée de trading ! 🚀_")
    lines.append("_[Ouvrir l'application](https://trading-app.streamlit.app)_")

    return "\n".join(lines)


if __name__ == "__main__":
    print("Génération du rapport matinal...")
    try:
        report = format_morning_report()
        send_telegram(report)
        print("✅ Rapport envoyé avec succès")
    except Exception as e:
        error_msg = f"❌ Erreur rapport trading: {e}"
        send_telegram(error_msg)
        print(error_msg)
        sys.exit(1)
