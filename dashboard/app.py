import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

from data.fetcher import fetch_ohlcv
from data.fundamentals import get_fundamentals, get_news, get_macro_indicators
from data.portfolio_reader import extract_portfolio_from_image, generate_portfolio_recommendations
from strategies.golden_cross import compute_signals
from strategies.scenario_engine import compute_scenarios
from strategies.stock_screener import find_best_stocks, find_dip_opportunities, UNIVERSE
from strategies.kabbaj_analysis import full_kabbaj_analysis, scan_entry_signals
from alerts.alert_engine import scan_portfolio_alerts, get_alert_history
from backtest.engine import run_backtest
from broker.paper_broker import get_portfolio, place_order, reset_portfolio

st.set_page_config(page_title="Trading App", layout="wide", page_icon="📈")

# ── Sidebar ────────────────────────────────────────────────────────────────────
st.sidebar.title("⚙️ Paramètres")
ticker = st.sidebar.text_input("Ticker principal", value="SPY").upper()
period = st.sidebar.selectbox("Période historique", ["1y", "2y", "5y", "10y"], index=2)
ma_fast = st.sidebar.slider("MA rapide (jours)", 10, 100, 50)
ma_slow = st.sidebar.slider("MA lente (jours)", 50, 300, 200)
initial_capital = st.sidebar.number_input("Capital initial ($)", 1000, 1_000_000, 10_000, step=1000)

st.sidebar.markdown("---")
st.sidebar.subheader("📦 Paper Trading")
latest_price_placeholder = st.sidebar.empty()
if st.sidebar.button("🔄 Réinitialiser le portefeuille"):
    reset_portfolio(initial_capital)
    st.sidebar.success("Portefeuille réinitialisé")

# ── Load main data ─────────────────────────────────────────────────────────────
@st.cache_data(ttl=1800)
def load_main(ticker, period, fast, slow):
    try:
        df = fetch_ohlcv(ticker, period)
        signals = compute_signals(df, fast, slow)
        result = run_backtest(signals, initial_capital=10_000)
        return df, signals, result
    except Exception as e:
        return None, None, None

@st.cache_data(ttl=3600)
def load_fundamentals(ticker):
    try:
        return get_fundamentals(ticker)
    except Exception:
        return {}

@st.cache_data(ttl=3600)
def load_news(ticker):
    try:
        return get_news(ticker)
    except Exception:
        return []

@st.cache_data(ttl=3600)
def load_macro():
    try:
        return get_macro_indicators()
    except Exception:
        return {}

with st.spinner(f"Chargement de {ticker}…"):
    try:
        df_raw, df_sig, bt = load_main(ticker, period, ma_fast, ma_slow)
    except Exception as e:
        st.error(f"Impossible de charger {ticker} : {e}")
        st.stop()

latest_price = df_raw["close"].iloc[-1]
latest_price_placeholder.metric("Dernier prix", f"${latest_price:,.2f}")

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
    "📊 Graphique & Signaux",
    "🧪 Backtest",
    "🔍 Analyse Fondamentale",
    "📚 Analyse Kabbaj",
    "🔔 Alertes Portefeuille",
    "⚖️ Comparaison",
    "💡 Opportunités",
    "📸 Mon Portefeuille",
    "💼 Portefeuille Papier",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Chart
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.header(f"{ticker} — Golden Cross ({ma_fast}/{ma_slow})")
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.75, 0.25], vertical_spacing=0.03)
    fig.add_trace(go.Candlestick(
        x=df_raw.index, open=df_raw["open"], high=df_raw["high"],
        low=df_raw["low"], close=df_raw["close"], name="Prix",
        increasing_line_color="#26a69a", decreasing_line_color="#ef5350"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_sig.index, y=df_sig["ma_fast"],
                             name=f"MA{ma_fast}", line=dict(color="#ffa726", width=1.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_sig.index, y=df_sig["ma_slow"],
                             name=f"MA{ma_slow}", line=dict(color="#42a5f5", width=1.5)), row=1, col=1)
    buys = df_sig[df_sig["signal"] == 1.0]
    fig.add_trace(go.Scatter(x=buys.index, y=buys["close"], mode="markers",
                             name="Golden Cross ✅",
                             marker=dict(symbol="triangle-up", size=12, color="#26a69a")), row=1, col=1)
    sells = df_sig[df_sig["signal"] == -1.0]
    fig.add_trace(go.Scatter(x=sells.index, y=sells["close"], mode="markers",
                             name="Death Cross ❌",
                             marker=dict(symbol="triangle-down", size=12, color="#ef5350")), row=1, col=1)
    colors = ["#26a69a" if c >= o else "#ef5350"
              for c, o in zip(df_raw["close"], df_raw["open"])]
    fig.add_trace(go.Bar(x=df_raw.index, y=df_raw["volume"], name="Volume",
                         marker_color=colors, opacity=0.6), row=2, col=1)
    fig.update_layout(height=600, xaxis_rangeslider_visible=False,
                      legend=dict(orientation="h", y=1.02),
                      margin=dict(l=0, r=0, t=30, b=0))
    st.plotly_chart(fig, use_container_width=True)

    recent_signals = df_sig[df_sig["signal"].isin([1.0, -1.0])][["close", "signal"]].tail(10).copy()
    recent_signals["signal"] = recent_signals["signal"].map({1.0: "🟢 Golden Cross", -1.0: "🔴 Death Cross"})
    recent_signals.columns = ["Prix", "Signal"]
    st.subheader("Derniers signaux")
    st.dataframe(recent_signals[::-1], use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Backtest
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.header("Résultats du Backtest")
    m = bt["metrics"]
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Rendement stratégie", f"{m['total_return']}%",
              delta=f"{m['total_return'] - m['buy_hold_return']:.1f}% vs B&H")
    c2.metric("Buy & Hold", f"{m['buy_hold_return']}%")
    c3.metric("Sharpe Ratio", m["sharpe_ratio"])
    c4.metric("Max Drawdown", f"{m['max_drawdown']}%")
    c5.metric("Taux de réussite", f"{m['win_rate']}%",
              help=f"{m['num_trades']} trades total")

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=bt["equity_curve"].index, y=bt["equity_curve"]["equity"],
                              name="Stratégie Golden Cross", fill="tozeroy",
                              line=dict(color="#42a5f5")))
    bh_equity = (df_sig["close"] / df_sig["close"].iloc[0]) * initial_capital
    fig2.add_trace(go.Scatter(x=bh_equity.index, y=bh_equity,
                              name="Buy & Hold", line=dict(color="#ffa726", dash="dash")))
    fig2.update_layout(title="Courbe de capital", height=350,
                       yaxis_title="Capital ($)", margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(fig2, use_container_width=True)

    fig3 = go.Figure(go.Scatter(x=bt["drawdown"].index, y=bt["drawdown"],
                                fill="tozeroy", line=dict(color="#ef5350"), name="Drawdown"))
    fig3.update_layout(title="Drawdown (%)", height=200,
                       yaxis_title="%", margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(fig3, use_container_width=True)

    st.subheader("Historique des trades")
    if not bt["trades"].empty:
        st.dataframe(bt["trades"].set_index("date"), use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Analyse Fondamentale + Scénarios
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.header(f"Analyse Fondamentale & Scénarios — {ticker}")

    with st.spinner("Chargement des données fondamentales…"):
        try:
            fund = load_fundamentals(ticker)
        except Exception:
            fund = {}
        try:
            news = load_news(ticker)
        except Exception:
            news = []
        try:
            macro = load_macro()
        except Exception:
            macro = {}

    if not fund:
        st.warning("⚠️ Yahoo Finance limite les requêtes depuis Streamlit Cloud. Les données fondamentales sont temporairement indisponibles. Réessayez dans quelques minutes ou lancez l'application en local.")
        st.stop()

    with st.spinner("Calcul des scénarios…"):
        scenario_result = compute_scenarios(df_sig, fund, macro)

    # ── Company overview ───────────────────────────────────────────────────────
    st.subheader(f"🏢 {fund.get('name', ticker)}")
    col_s, col_i = st.columns(2)
    col_s.markdown(f"**Secteur :** {fund.get('sector', 'N/A')}")
    col_i.markdown(f"**Industrie :** {fund.get('industry', 'N/A')}")

    # ── Key ratios ─────────────────────────────────────────────────────────────
    st.subheader("📐 Ratios clés")
    r1, r2, r3, r4, r5, r6 = st.columns(6)

    def fmt_pct(v): return f"{v*100:.1f}%" if v is not None else "N/A"
    def fmt_x(v): return f"{v:.1f}x" if v is not None else "N/A"
    def fmt_num(v, suffix=""): return f"{v:.2f}{suffix}" if v is not None else "N/A"

    r1.metric("PER (TTM)", fmt_x(fund.get("pe_ratio")))
    r2.metric("PER Forward", fmt_x(fund.get("forward_pe")))
    r3.metric("P/B", fmt_x(fund.get("pb_ratio")))
    r4.metric("Croissance CA", fmt_pct(fund.get("revenue_growth")))
    r5.metric("Croissance EPS", fmt_pct(fund.get("eps_growth")))
    r6.metric("Marge nette", fmt_pct(fund.get("profit_margin")))

    r7, r8, r9, r10, r11, r12 = st.columns(6)
    r7.metric("ROE", fmt_pct(fund.get("roe")))
    r8.metric("Dettes/CP", fmt_num(fund.get("debt_to_equity")))
    r9.metric("Beta", fmt_num(fund.get("beta")))
    r10.metric("Dividende", fmt_pct(fund.get("dividend_yield")))
    r11.metric("Plus haut 52s", fmt_num(fund.get("52w_high"), "$") if fund.get("52w_high") else "N/A")
    r12.metric("Plus bas 52s", fmt_num(fund.get("52w_low"), "$") if fund.get("52w_low") else "N/A")

    # ── Macro indicators ───────────────────────────────────────────────────────
    st.subheader("🌍 Indicateurs Macro")
    macro_cols = st.columns(4)
    macro_labels = {"VIX": "VIX (Peur marché)", "SP500": "S&P 500", "T10Y": "Taux 10 ans US (%)", "DXY": "Dollar Index"}
    for i, (key, label) in enumerate(macro_labels.items()):
        data = macro.get(key)
        if data:
            macro_cols[i].metric(label, data["value"], delta=f"{data['change_1m']:+.2f}% (1 mois)")

    # ── Macro charts ───────────────────────────────────────────────────────────
    with st.expander("📈 Graphiques macro (6 mois)"):
        macro_chart_cols = st.columns(2)
        chart_pairs = [("VIX", "SP500"), ("T10Y", "DXY")]
        for col_idx, (k1, k2) in enumerate(chart_pairs):
            with macro_chart_cols[col_idx]:
                for key in (k1, k2):
                    data = macro.get(key)
                    if data and "series" in data:
                        fig_m = go.Figure(go.Scatter(
                            x=data["series"].index, y=data["series"],
                            fill="tozeroy", name=key,
                            line=dict(color="#42a5f5" if key in ("SP500",) else "#ffa726")
                        ))
                        fig_m.update_layout(title=macro_labels.get(key, key),
                                            height=200, margin=dict(l=0, r=0, t=30, b=0))
                        st.plotly_chart(fig_m, use_container_width=True)

    # ── Signal table ───────────────────────────────────────────────────────────
    st.subheader("🔎 Signaux détectés")
    sig_df = pd.DataFrame(scenario_result["signals"],
                          columns=["Signal", "Interprétation", "Poids"])
    st.dataframe(sig_df.style.map(
        lambda v: "color: #26a69a" if "haussier" in str(v).lower()
        else ("color: #ef5350" if "baissier" in str(v).lower() else ""),
        subset=["Interprétation"]
    ), use_container_width=True, hide_index=True)

    # ── Market scenarios (Bull/Base/Bear) ──────────────────────────────────────
    st.subheader("🎯 Scénarios de Marché & Probabilités")
    dominant = scenario_result["dominant"]
    scenarios = scenario_result["scenarios"]

    scen_cols = st.columns(3)
    order = ["bull", "base", "bear"]
    for i, key in enumerate(order):
        s = scenarios[key]
        border = "3px solid" if key == dominant else "1px solid #444"
        ret = s.get("estimated_return", 0)
        ret_str = f"+{ret}%" if ret >= 0 else f"{ret}%"
        ret_color = "#26a69a" if ret >= 0 else "#ef5350"
        with scen_cols[i]:
            st.markdown(f"""
<div style="border: {border} {s['color']}; border-radius: 10px; padding: 16px; background: #1a1a2e;">
  <h3 style="color: {s['color']}; margin: 0;">{s['label']}</h3>
  <h1 style="color: {s['color']}; margin: 8px 0;">{s['probability']}%</h1>
  <p style="margin: 4px 0;">Rendement estimé 12 mois : <b style="color:{ret_color}; font-size:1.1em;">{ret_str}</b></p>
  <p style="color: #ccc; font-size: 0.85em;">{s['description']}</p>
  {'<p style="color: gold; margin:4px 0;">⭐ Scénario dominant</p>' if key == dominant else ''}
</div>
""", unsafe_allow_html=True)

    fig_prob = go.Figure(go.Bar(
        x=[scenarios[k]["probability"] for k in order],
        y=[scenarios[k]["label"] for k in order],
        orientation="h",
        marker_color=[scenarios[k]["color"] for k in order],
        text=[f"{scenarios[k]['probability']}% | {'+' if scenarios[k].get('estimated_return',0)>=0 else ''}{scenarios[k].get('estimated_return',0)}%" for k in order],
        textposition="inside",
    ))
    fig_prob.update_layout(title="Distribution des probabilités",
                           height=200, margin=dict(l=0, r=0, t=40, b=0),
                           xaxis=dict(range=[0, 100], ticksuffix="%"))
    st.plotly_chart(fig_prob, use_container_width=True)

    # ── Investment scenarios (Risky / Moderate / Safe) ─────────────────────────
    st.markdown("---")
    st.subheader("💰 Scénarios d'Investissement")
    st.caption(f"Volatilité annualisée : {scenario_result.get('volatility', '?')}% · RSI actuel : {scenario_result.get('rsi', '?')}")

    inv_scenarios = scenario_result.get("investment_scenarios", [])
    inv_cols = st.columns(3)
    for i, s in enumerate(inv_scenarios):
        with inv_cols[i]:
            st.markdown(f"""
<div style="border: 2px solid {s['color']}; border-radius: 12px; padding: 18px; background: #0d1117; height: 100%;">
  <h3 style="color: {s['color']}; margin: 0 0 8px 0;">{s['label']}</h3>
  <div style="display:flex; justify-content:space-between; margin-bottom:8px;">
    <div>
      <div style="color:#aaa; font-size:0.75em;">RENDEMENT ESTIMÉ</div>
      <div style="color:#26a69a; font-size:1.6em; font-weight:bold;">{s['estimated_return']}</div>
    </div>
    <div>
      <div style="color:#aaa; font-size:0.75em;">PIRE CAS</div>
      <div style="color:#ef5350; font-size:1.6em; font-weight:bold;">{s['worst_case']}</div>
    </div>
    <div>
      <div style="color:#aaa; font-size:0.75em;">PROBABILITÉ</div>
      <div style="color:{s['color']}; font-size:1.6em; font-weight:bold;">{s['probability']}%</div>
    </div>
  </div>
  <hr style="border-color:#333; margin:8px 0;"/>
  <p style="color:#bbb; font-size:0.82em; margin:4px 0;">⏱ Horizon : <b>{s['horizon']}</b></p>
  <p style="color:#bbb; font-size:0.82em; margin:4px 0;">📌 Stratégie : {s['strategy']}</p>
  <p style="color:#bbb; font-size:0.82em; margin:4px 0;">✅ Condition : {s['condition']}</p>
  <hr style="border-color:#333; margin:8px 0;"/>
  <p style="color:#888; font-size:0.8em; margin:4px 0;">🔴 Stop-loss : <b>${s['stop_loss']}</b></p>
  <p style="color:#888; font-size:0.8em; margin:4px 0;">🟢 Objectif : <b>${s['take_profit']}</b></p>
</div>
""", unsafe_allow_html=True)

    # Analyst target
    target = fund.get("analyst_target")
    num_analysts = fund.get("num_analysts", 0)
    rec = fund.get("recommendation", "")
    if target:
        upside = (target / latest_price - 1) * 100
        color = "#26a69a" if upside > 0 else "#ef5350"
        st.markdown(f"""
**🏦 Consensus de {num_analysts} analystes :**
Recommandation **{rec.upper()}** · Objectif de prix :
<span style="color:{color}; font-size:1.2em; font-weight:bold;">${target:.2f} ({upside:+.1f}%)</span>
""", unsafe_allow_html=True)

    # ── News ───────────────────────────────────────────────────────────────────
    st.subheader(f"📰 Actualités récentes — {ticker}")
    if news:
        for item in news:
            with st.expander(f"**{item['date']}** — {item['title']} *({item['source']})*"):
                if item["summary"]:
                    st.write(item["summary"])
                if item["url"]:
                    st.markdown(f"[Lire l'article]({item['url']})")
    else:
        st.info("Aucune actualité disponible pour ce ticker.")

# ══════════════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Analyse Kabbaj
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.header(f"📚 Analyse Technique — Méthode Kabbaj · {ticker}")
    st.caption("Basée sur *L'Art du Trading* de Thami Kabbaj — chandeliers, Wyckoff, MACD, Bollinger, Stochastique, Money Management")

    kab_capital = st.number_input("Capital total ($)", 1000, 10_000_000, int(initial_capital), step=1000, key="kab_capital")
    kab_risk = st.slider("Risque max par trade (%)", 0.5, 5.0, 2.0, step=0.5, key="kab_risk")

    @st.cache_data(ttl=300)
    def load_kabbaj(ticker, period, capital, risk):
        df = fetch_ohlcv(ticker, period)
        return full_kabbaj_analysis(df, capital=capital, risk_pct=risk), df

    with st.spinner("Analyse Kabbaj en cours…"):
        kab, df_kab = load_kabbaj(ticker, period, kab_capital, kab_risk)

    # ── Setup Score ─────────────────────────────────────────────────────────
    setup = kab["setup_score"]
    phase = kab["market_phase"]
    mm = kab["money_management"]
    sr = kab["support_resistance"]

    score_col, phase_col = st.columns([1, 2])
    with score_col:
        st.markdown(f"""
<div style="border: 3px solid {setup['rating_color']}; border-radius:12px; padding:20px; text-align:center; background:#0d1117;">
  <div style="color:#aaa; font-size:0.85em; margin-bottom:4px;">SCORE DE SETUP</div>
  <div style="font-size:3.5em; font-weight:bold; color:{setup['rating_color']};">{setup['score']}<span style="font-size:0.4em;">/100</span></div>
  <div style="color:{setup['rating_color']}; font-size:1em; margin:8px 0;">{setup['rating']}</div>
  <div style="color:#ccc; font-size:0.8em; background:#1a1a2e; padding:8px; border-radius:6px; margin-top:8px;">{setup['action']}</div>
</div>
""", unsafe_allow_html=True)

    with phase_col:
        st.markdown(f"""
<div style="border: 2px solid {phase['color']}; border-radius:12px; padding:16px; background:#0d1117;">
  <div style="font-size:1.3em; font-weight:bold; color:{phase['color']};">{phase['emoji']} Phase de Marché : {phase['phase']}</div>
  <p style="color:#ccc; margin:8px 0;">{phase['description']}</p>
  <div style="display:flex; gap:20px; flex-wrap:wrap; margin-top:8px;">
    <span style="color:#aaa; font-size:0.85em;">📊 ADX : <b style="color:{phase['color']};">{phase['adx']}</b> — {phase['adx_interpretation']}</span>
    <span style="color:#aaa; font-size:0.85em;">DI+ : <b style="color:#26a69a;">{phase['di_plus']}</b></span>
    <span style="color:#aaa; font-size:0.85em;">DI- : <b style="color:#ef5350;">{phase['di_minus']}</b></span>
    <span style="color:#aaa; font-size:0.85em;">3 mois : <b>{phase['ret_3m']:+.1f}%</b></span>
    <span style="color:#aaa; font-size:0.85em;">Volume : <b>×{phase['vol_ratio']}</b> vs moyenne</span>
  </div>
  <div style="margin-top:10px; padding:8px; background:#1a1a2e; border-radius:6px; color:#ffa726;">
    💡 <b>Conseil Kabbaj :</b> {phase['advice']}
  </div>
</div>
""", unsafe_allow_html=True)

    st.markdown("---")

    # ── Criteria breakdown ───────────────────────────────────────────────────
    st.subheader("📋 Détail des critères")
    crit_cols = st.columns(2)
    for i, (label, pts, typ) in enumerate(setup["criteria"]):
        color = "#26a69a" if typ == "bullish" else "#ef5350" if typ == "bearish" else "#ffa726"
        pts_str = f"+{pts}" if pts > 0 else str(pts)
        with crit_cols[i % 2]:
            st.markdown(f"""
<div style="display:flex; justify-content:space-between; padding:6px 10px; margin-bottom:4px;
     background:#0d1117; border-radius:6px; border-left:3px solid {color};">
  <span style="color:#ccc; font-size:0.85em;">{label}</span>
  <b style="color:{color};">{pts_str} pts</b>
</div>
""", unsafe_allow_html=True)

    st.markdown("---")

    # ── Charts: MACD + Bollinger + Stochastique ──────────────────────────────
    st.subheader("📈 Indicateurs Techniques")

    fig_ind = make_subplots(
        rows=4, cols=1, shared_xaxes=True,
        row_heights=[0.45, 0.2, 0.2, 0.15],
        vertical_spacing=0.03,
        subplot_titles=["Prix + Bandes de Bollinger", "MACD", "Stochastique", "ATR"],
    )

    # Prix + Bollinger
    bb = kab["bollinger"]
    fig_ind.add_trace(go.Candlestick(
        x=df_kab.index, open=df_kab["open"], high=df_kab["high"],
        low=df_kab["low"], close=df_kab["close"], name="Prix",
        increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
        showlegend=False), row=1, col=1)
    fig_ind.add_trace(go.Scatter(x=bb.index, y=bb["bb_upper"], name="BB Sup",
        line=dict(color="#42a5f5", width=1, dash="dot"), showlegend=False), row=1, col=1)
    fig_ind.add_trace(go.Scatter(x=bb.index, y=bb["bb_mid"], name="BB Mid",
        line=dict(color="#ffa726", width=1)), row=1, col=1)
    fig_ind.add_trace(go.Scatter(x=bb.index, y=bb["bb_lower"], name="BB Inf",
        line=dict(color="#42a5f5", width=1, dash="dot"),
        fill="tonexty", fillcolor="rgba(66,165,245,0.05)", showlegend=False), row=1, col=1)

    # Supports/résistances
    for sup in sr["supports"][-3:]:
        fig_ind.add_hline(y=sup, line_color="#26a69a", line_dash="dash",
                          line_width=1, row=1, col=1,
                          annotation_text=f"S {sup}", annotation_font_color="#26a69a")
    for res in sr["resistances"][:3]:
        fig_ind.add_hline(y=res, line_color="#ef5350", line_dash="dash",
                          line_width=1, row=1, col=1,
                          annotation_text=f"R {res}", annotation_font_color="#ef5350")

    # MACD
    macd = kab["macd"]
    colors_hist = ["#26a69a" if v >= 0 else "#ef5350" for v in macd["histogram"]]
    fig_ind.add_trace(go.Bar(x=macd.index, y=macd["histogram"], name="Histogramme",
        marker_color=colors_hist, showlegend=False), row=2, col=1)
    fig_ind.add_trace(go.Scatter(x=macd.index, y=macd["macd"], name="MACD",
        line=dict(color="#42a5f5", width=1.5)), row=2, col=1)
    fig_ind.add_trace(go.Scatter(x=macd.index, y=macd["signal"], name="Signal",
        line=dict(color="#ffa726", width=1.5)), row=2, col=1)

    # Stochastique
    stoch = kab["stochastic"]
    fig_ind.add_trace(go.Scatter(x=stoch.index, y=stoch["stoch_k"], name="%K",
        line=dict(color="#ab47bc", width=1.5)), row=3, col=1)
    fig_ind.add_trace(go.Scatter(x=stoch.index, y=stoch["stoch_d"], name="%D",
        line=dict(color="#ffa726", width=1.5)), row=3, col=1)
    fig_ind.add_hline(y=80, line_color="#ef5350", line_dash="dot", line_width=1, row=3, col=1)
    fig_ind.add_hline(y=20, line_color="#26a69a", line_dash="dot", line_width=1, row=3, col=1)

    # ATR
    fig_ind.add_trace(go.Scatter(x=kab["atr"].index, y=kab["atr"], name="ATR",
        line=dict(color="#78909c"), fill="tozeroy", showlegend=False), row=4, col=1)

    fig_ind.update_layout(height=750, xaxis_rangeslider_visible=False,
                          legend=dict(orientation="h", y=1.02),
                          margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(fig_ind, use_container_width=True)

    # ── Candlestick patterns ─────────────────────────────────────────────────
    st.subheader("🕯️ Patterns Chandeliers Détectés (10 dernières bougies)")
    patterns = kab["candlestick_patterns"]
    if patterns:
        for p in reversed(patterns[-8:]):
            st.markdown(f"""
<div style="border-left:4px solid {p['color']}; padding:8px 12px; margin-bottom:6px;
     background:#0d1117; border-radius:0 6px 6px 0;">
  <b style="color:{p['color']};">{p['pattern']}</b>
  <span style="color:#888; font-size:0.85em; margin-left:8px;">{str(p['date'])[:10]}</span>
  <br/><span style="color:#ccc; font-size:0.85em;">{p['signal']}</span>
</div>
""", unsafe_allow_html=True)
    else:
        st.info("Aucun pattern significatif détecté sur les 10 dernières bougies.")

    st.markdown("---")

    # ── Support / Résistance ─────────────────────────────────────────────────
    st.subheader("🎯 Niveaux Clés (Supports & Résistances)")
    sr_c1, sr_c2, sr_c3 = st.columns(3)
    with sr_c1:
        st.markdown("**🟢 Supports**")
        for s in reversed(sr["supports"][-5:]):
            dist = round((kab["price"] - s) / kab["price"] * 100, 1)
            st.markdown(f"<span style='color:#26a69a;'>${s}</span> <span style='color:#555;'>(-{dist}%)</span>", unsafe_allow_html=True)
    with sr_c2:
        st.markdown("**🔴 Résistances**")
        for r in sr["resistances"][:5]:
            dist = round((r - kab["price"]) / kab["price"] * 100, 1)
            st.markdown(f"<span style='color:#ef5350;'>${r}</span> <span style='color:#555;'>(+{dist}%)</span>", unsafe_allow_html=True)
    with sr_c3:
        st.markdown("**⚪ Niveaux Psychologiques**")
        for p in sr["psychological_levels"]:
            st.markdown(f"<span style='color:#ffa726;'>${p}</span>", unsafe_allow_html=True)
        if sr["risk_reward_ratio"]:
            color_rr = "#26a69a" if sr["risk_reward_ratio"] >= 2 else "#ef5350"
            st.markdown(f"<br/>**Ratio R/R :** <span style='color:{color_rr}; font-size:1.2em;'>{sr['risk_reward_ratio']}:1</span>", unsafe_allow_html=True)

    st.markdown("---")

    # ── Money Management ─────────────────────────────────────────────────────
    st.subheader(f"💰 Money Management — Règle des {kab_risk}% (Kabbaj)")
    if "error" not in mm:
        mm_c1, mm_c2, mm_c3, mm_c4 = st.columns(4)
        mm_c1.metric("Risque max par trade", f"${mm['max_risk_amount']:,.0f}")
        mm_c2.metric("Nb d'actions à acheter", f"{mm['position_size_shares']:.1f}")
        mm_c3.metric("Valeur de la position", f"${mm['position_value']:,.0f}",
                     delta=f"{mm['position_pct_capital']}% du capital")
        mm_c4.metric("Stop-loss suggéré", f"${mm['stop_loss']}",
                     delta=f"-{mm['stop_loss_pct']}%")

        if mm.get("atr_stop_suggestion"):
            st.info(f"📏 Stop ATR (2×ATR) suggéré : **${mm['atr_stop_suggestion']}** ({mm['atr_stop_pct']}% sous le prix) · ATR actuel : **${kab['atr_value']}**")

        st.markdown(f"""
<div style="background:#1a1a2e; border-left:4px solid #ffa726; padding:12px 16px; border-radius:0 8px 8px 0; margin-top:8px;">
  <b style="color:#ffa726;">📖 Règle Kabbaj :</b>
  <span style="color:#ccc;"> {mm['rule']}. Cette discipline est la base de la survie en trading à long terme.</span>
</div>
""", unsafe_allow_html=True)

    # ── Scanner — Meilleures entrées en position ──────────────────────────────
    st.markdown("---")
    st.subheader("🚨 Scanner — Actions avec signal d'entrée en position")
    st.caption("Scan des meilleures actions selon la méthode Kabbaj : score de setup, phase de marché, confluence d'indicateurs")

    sc1, sc2, sc3 = st.columns([2, 1, 1])
    with sc1:
        scan_sectors = st.multiselect(
            "Secteurs à scanner",
            options=list(UNIVERSE.keys()),
            default=["Tech", "Finance", "ETF"],
            key="kab_scan_sectors"
        )
    with sc2:
        scan_timeframe = st.selectbox("Horizon", ["Les deux", "Long terme uniquement", "Court terme uniquement"], key="kab_tf")
    with sc3:
        scan_min_score = st.slider("Score minimum", 30, 80, 55, key="kab_min_score")

    scan_tickers = [t for s, tickers in UNIVERSE.items() if s in scan_sectors for t in tickers]

    # Timeframe legend
    st.markdown("""
<div style="display:flex; gap:24px; margin:8px 0 12px 0;">
  <div style="background:#0d2818; border:1px solid #26a69a; border-radius:6px; padding:6px 14px;">
    <b style="color:#26a69a;">🕰️ LONG TERME</b> <span style="color:#888; font-size:0.85em;">MA50/200 · Wyckoff · ADX · Fondamentaux · Semaines/mois</span>
  </div>
  <div style="background:#1a0d0d; border:1px solid #ef5350; border-radius:6px; padding:6px 14px;">
    <b style="color:#ef5350;">⚡ COURT TERME</b> <span style="color:#888; font-size:0.85em;">MACD · RSI · Stochastique · Chandeliers · 1-10 jours</span>
  </div>
</div>
""", unsafe_allow_html=True)

    if st.button("🔍 Lancer le scan Kabbaj", use_container_width=True):
        @st.cache_data(ttl=1800)
        def run_kabbaj_scan(tickers_tuple, capital, period):
            return scan_entry_signals(list(tickers_tuple), period=period, capital=capital)

        with st.spinner(f"Scan Kabbaj sur {len(scan_tickers)} actions… (~2-3 min)"):
            scan_results = run_kabbaj_scan(tuple(scan_tickers), kab_capital, period)

        filtered = [r for r in scan_results if r["setup_score"] >= scan_min_score]

        if not filtered:
            st.warning(f"Aucune action avec un score ≥ {scan_min_score} dans les secteurs sélectionnés.")
        else:
            st.success(f"✅ {len(filtered)} action(s) avec signal d'entrée détecté(s)")

            # Split long vs short term display
            lt_tab, st_tab_inner = st.tabs(["🕰️ Long Terme (semaines/mois)", "⚡ Court Terme (1-10 jours)"])

            def render_scan_card(r, timeframe_hint):
                score = r["setup_score"]
                rc = r["rating_color"]
                stars = "⭐" * (5 if score >= 70 else 4 if score >= 55 else 3)
                upside_str = f"+{r['upside']}%" if r.get("upside") else "N/A"
                rr_str = f"{r['risk_reward']}:1" if r.get("risk_reward") else "N/A"
                rr_color = "#26a69a" if (r.get("risk_reward") or 0) >= 2 else "#ffa726"
                pattern_str = f"🕯️ {r['last_pattern']}" if r.get("last_pattern") else ""
                pos_str = (f"{r['position_shares']:.1f} actions (${r['position_value']:,.0f})"
                           if r.get("position_shares") else "N/A")
                ret1_color = "#26a69a" if r["ret_1m"] >= 0 else "#ef5350"
                ret3_color = "#26a69a" if r["ret_3m"] >= 0 else "#ef5350"
                pattern_html = f"&nbsp;&nbsp;·&nbsp;&nbsp;<span style='color:#ab47bc;font-size:0.85em;'>{pattern_str}</span>" if pattern_str else ""
                tf_badge_color = "#26a69a" if "Long" in timeframe_hint else "#ef5350"
                st.markdown(f"""
<div style="border:2px solid {rc};border-radius:12px;padding:16px;margin-bottom:12px;background:#0d1117;">
  <div style="display:flex;justify-content:space-between;align-items:flex-start;flex-wrap:wrap;gap:8px;">
    <div>
      <span style="background:{tf_badge_color};color:#fff;font-size:0.7em;padding:2px 8px;border-radius:4px;font-weight:bold;">{timeframe_hint}</span>
      <span style="font-size:1.4em;font-weight:bold;color:{rc};margin-left:8px;">{r["ticker"]}</span>
      <span style="color:#888;margin-left:8px;">{r["name"]}</span>
      <span style="color:#555;font-size:0.8em;margin-left:8px;">{r["sector"]}</span><br/>
      <span style="color:{r["phase_color"]};font-size:0.9em;">{r["phase_emoji"]} {r["phase"]}</span>{pattern_html}
    </div>
    <div style="text-align:right;">
      <div style="font-size:2em;font-weight:bold;color:{rc};">{score}<span style="font-size:0.4em;">/100</span></div>
      <div style="color:{rc};font-size:0.85em;">{stars}</div>
    </div>
  </div>
  <div style="background:#1a1a2e;border-left:3px solid {rc};padding:8px 12px;margin:10px 0;border-radius:0 6px 6px 0;">
    <b style="color:{rc};">👉 {r["action"]}</b>
  </div>
  <div style="display:flex;gap:20px;flex-wrap:wrap;">
    <div><span style="color:#888;font-size:0.8em;">PRIX</span><br/><b style="color:#eee;">${r["price"]:,.2f}</b></div>
    <div><span style="color:#888;font-size:0.8em;">ADX</span><br/><b style="color:#eee;">{r["adx"]}</b></div>
    <div><span style="color:#888;font-size:0.8em;">1 MOIS</span><br/><b style="color:{ret1_color};">{r["ret_1m"]:+.1f}%</b></div>
    <div><span style="color:#888;font-size:0.8em;">3 MOIS</span><br/><b style="color:{ret3_color};">{r["ret_3m"]:+.1f}%</b></div>
    <div><span style="color:#888;font-size:0.8em;">STOP-LOSS</span><br/><b style="color:#ef5350;">${r["stop_loss"]}</b></div>
    <div><span style="color:#888;font-size:0.8em;">R/R</span><br/><b style="color:{rr_color};">{rr_str}</b></div>
    <div><span style="color:#888;font-size:0.8em;">OBJECTIF</span><br/><b style="color:#26a69a;">{upside_str}</b></div>
    <div><span style="color:#888;font-size:0.8em;">CONSENSUS</span><br/><b style="color:#42a5f5;">{r["recommendation"]}</b></div>
    <div><span style="color:#888;font-size:0.8em;">POSITION (2%)</span><br/><b style="color:#eee;">{pos_str}</b></div>
  </div>
</div>
""", unsafe_allow_html=True)

            with lt_tab:
                st.caption("Basé sur Golden Cross, phase Wyckoff, ADX et fondamentaux — horizon semaines/mois")
                long_results = [r for r in filtered if r["setup_score"] >= scan_min_score
                                and ("Markup" in r["phase"] or "Accumulation" in r["phase"])]
                if not long_results:
                    long_results = filtered  # fallback
                for r in long_results:
                    render_scan_card(r, "LONG TERME")

            with st_tab_inner:
                st.caption("Basé sur MACD, RSI, Stochastique, chandeliers — horizon 1 à 10 jours")
                # Short term: prioritize RSI/MACD signals — sort by 1-month momentum
                short_results = sorted(filtered, key=lambda x: abs(x["ret_1m"]), reverse=True)
                for r in short_results:
                    render_scan_card(r, "COURT TERME")



# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — Alertes Portefeuille
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.header("🔔 Alertes — Mon Portefeuille en Temps Réel")
    st.markdown("Surveillance automatique de vos positions avec signaux d'entrée/sortie **long terme** et **court terme**.")

    alert_source = st.radio(
        "Source du portefeuille à surveiller",
        ["📦 Paper Trading (simulé)", "✏️ Saisir manuellement"],
        horizontal=True
    )

    if alert_source == "📦 Paper Trading (simulé)":
        pf = get_portfolio()
        watch_positions = {t: {"quantity": q, "avg_price": None}
                           for t, q in pf["positions"].items()}
        if not watch_positions:
            st.info("Votre portefeuille paper est vide. Ajoutez des positions dans l'onglet 💼 ou saisissez manuellement.")
    else:
        manual_input = st.text_area(
            "Entrez vos positions (une par ligne : TICKER,quantité,prix_moyen)",
            placeholder="AAPL,10,175.50\nMSFT,5,380.00\nNVDA,3,900.00",
            height=120
        )
        watch_positions = {}
        if manual_input.strip():
            for line in manual_input.strip().split("\n"):
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 1 and parts[0]:
                    t = parts[0].upper()
                    qty = float(parts[1]) if len(parts) > 1 else 1
                    avg = float(parts[2]) if len(parts) > 2 else None
                    watch_positions[t] = {"quantity": qty, "avg_price": avg}

    if watch_positions:
        st.markdown(f"**Positions surveillées :** {', '.join(f'`{t}`' for t in watch_positions)}")

        if st.button("🚨 Scanner les alertes maintenant", use_container_width=True):
            with st.spinner("Scan en cours…"):
                alert_results = scan_portfolio_alerts(watch_positions, period="2y")

            total_entry = sum(1 for r in alert_results.values()
                              if isinstance(r, dict) and not r.get("error") and r.get("has_entry"))
            total_exit = sum(1 for r in alert_results.values()
                             if isinstance(r, dict) and not r.get("error") and r.get("has_exit"))
            total_alerts = sum(r.get("total_alerts", 0) for r in alert_results.values()
                               if isinstance(r, dict) and not r.get("error"))

            s1, s2, s3 = st.columns(3)
            s1.metric("🟢 Signaux d'entrée", total_entry)
            s2.metric("🔴 Signaux de sortie", total_exit)
            s3.metric("📊 Alertes totales", total_alerts)

            for ticker, data in alert_results.items():
                if data.get("error"):
                    st.error(f"**{ticker}** — Erreur : {data['error']}")
                    continue

                lt_signals = data.get("long_term", [])
                st_signals = data.get("short_term", [])
                all_signals = lt_signals + st_signals
                price = data.get("price", 0)
                pnl = data.get("pnl_pct")
                qty = data.get("quantity", 0)
                avg = data.get("avg_price")
                pnl_str = f"{pnl:+.1f}%" if pnl is not None else "N/A"
                avg_str = f"${avg:.2f}" if avg else "N/A"
                has_entry = data.get("has_entry")
                has_exit = data.get("has_exit")
                badge = "🟢 ENTRÉE DÉTECTÉE" if has_entry else "🔴 SORTIE DÉTECTÉE" if has_exit else "ℹ️ INFO"

                with st.expander(f"**{ticker}** — {len(all_signals)} alerte(s) · {badge}", expanded=has_entry or has_exit):
                    pc1, pc2, pc3, pc4 = st.columns(4)
                    pc1.metric("Prix actuel", f"${price:,.2f}")
                    pc2.metric("Quantité", f"{qty:.2f}")
                    pc3.metric("Prix moyen", avg_str)
                    pc4.metric("P&L", pnl_str)

                    if not all_signals:
                        st.info("Aucun signal détecté — position neutre.")
                        continue

                    def render_signal_card(sig):
                        border = "3px" if sig["urgency"] == "high" else "2px" if sig["urgency"] == "medium" else "1px"
                        st.markdown(f"""
<div style="border-left:{border} solid {sig['color']};padding:10px 14px;margin-bottom:8px;
     background:#0d1117;border-radius:0 8px 8px 0;">
  <div style="display:flex;justify-content:space-between;align-items:center;">
    <b style="color:{sig['color']};">{sig['emoji']} {sig['title']}</b>
    <span style="background:{sig['color']};color:#000;padding:2px 8px;border-radius:4px;font-size:0.75em;font-weight:bold;">{sig['type']}</span>
  </div>
  <p style="color:#ccc;margin:6px 0 4px 0;font-size:0.9em;">{sig['message']}</p>
  <div style="background:#1a1a2e;padding:6px 10px;border-radius:4px;margin-top:4px;">
    <b style="color:{sig['color']};">👉 {sig['action']}</b>
  </div>
</div>
""", unsafe_allow_html=True)

                    if lt_signals:
                        st.markdown("#### 🕰️ Signaux Long Terme")
                        for sig in lt_signals:
                            render_signal_card(sig)
                    if st_signals:
                        st.markdown("#### ⚡ Signaux Court Terme")
                        for sig in st_signals:
                            render_signal_card(sig)

            with st.expander("📜 Historique des alertes"):
                history = get_alert_history()
                if history:
                    hist_rows = []
                    for scan in history[-10:]:
                        for t, d in scan.get("scan", {}).items():
                            if isinstance(d, dict) and not d.get("error"):
                                for sig in d.get("long_term", []) + d.get("short_term", []):
                                    hist_rows.append({
                                        "Date": scan["timestamp"][:10],
                                        "Ticker": t,
                                        "Horizon": sig["timeframe"],
                                        "Type": sig["type"],
                                        "Signal": sig["title"],
                                        "Prix": f"${sig['price']:,.2f}",
                                    })
                    if hist_rows:
                        st.dataframe(pd.DataFrame(hist_rows), use_container_width=True, hide_index=True)
                else:
                    st.info("Aucun historique disponible.")
    else:
        st.info("Ajoutez des positions pour activer la surveillance.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — Comparaison multi-tickers
# ══════════════════════════════════════════════════════════════════════════════
with tab6:
    st.header("⚖️ Comparaison de Tickers")
    raw_input = st.text_input(
        "Tickers à comparer (séparés par des virgules)",
        value="SPY, QQQ, AAPL, MSFT, NVDA"
    )
    compare_tickers = [t.strip().upper() for t in raw_input.split(",") if t.strip()]

    if st.button("🔄 Lancer la comparaison") or True:
        @st.cache_data(ttl=300)
        def load_comparison(tickers_tuple, period, fast, slow):
            results = {}
            for t in tickers_tuple:
                try:
                    df = fetch_ohlcv(t, period)
                    sig = compute_signals(df, fast, slow)
                    bt_res = run_backtest(sig, initial_capital=10_000)
                    fund_res = get_fundamentals(t)
                    results[t] = {"metrics": bt_res["metrics"], "fund": fund_res,
                                  "signals": sig, "equity": bt_res["equity_curve"]}
                except Exception as e:
                    results[t] = {"error": str(e)}
            return results

        with st.spinner("Comparaison en cours…"):
            comp = load_comparison(tuple(compare_tickers), period, ma_fast, ma_slow)

        # ── Metrics table ──────────────────────────────────────────────────────
        rows = []
        for t, data in comp.items():
            if "error" in data:
                rows.append({"Ticker": t, "Erreur": data["error"]})
                continue
            m = data["metrics"]
            f = data["fund"]
            pos = int(data["signals"]["position"].iloc[-1])
            rows.append({
                "Ticker": t,
                "Signal": "🟢 Long" if pos == 1 else "🔴 Hors position",
                "Rendement strat.": f"{m['total_return']}%",
                "Buy & Hold": f"{m['buy_hold_return']}%",
                "Sharpe": m["sharpe_ratio"],
                "Drawdown max": f"{m['max_drawdown']}%",
                "Taux réussite": f"{m['win_rate']}%",
                "Nb trades": m["num_trades"],
                "PER": f"{f.get('pe_ratio'):.1f}x" if f.get("pe_ratio") else "N/A",
                "Croiss. CA": f"{f.get('revenue_growth')*100:.1f}%" if f.get("revenue_growth") else "N/A",
                "Consensus": (f.get("recommendation") or "N/A").upper(),
            })

        df_comp = pd.DataFrame(rows).set_index("Ticker")
        st.dataframe(df_comp, use_container_width=True)

        # ── Equity curves comparison ───────────────────────────────────────────
        st.subheader("📈 Courbes de capital normalisées (base 100)")
        fig_cmp = go.Figure()
        colors_palette = ["#42a5f5", "#26a69a", "#ffa726", "#ab47bc", "#ef5350",
                          "#26c6da", "#d4e157", "#ff7043"]
        for i, (t, data) in enumerate(comp.items()):
            if "error" in data:
                continue
            eq = data["equity"]["equity"]
            normalized = eq / eq.iloc[0] * 100
            fig_cmp.add_trace(go.Scatter(
                x=normalized.index, y=normalized,
                name=t, line=dict(color=colors_palette[i % len(colors_palette)])
            ))
        fig_cmp.update_layout(height=400, yaxis_title="Performance (base 100)",
                               margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig_cmp, use_container_width=True)

        # ── Bar chart returns ──────────────────────────────────────────────────
        st.subheader("📊 Rendements comparés")
        valid = {t: d for t, d in comp.items() if "error" not in d}
        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(
            name="Stratégie Golden Cross",
            x=list(valid.keys()),
            y=[d["metrics"]["total_return"] for d in valid.values()],
            marker_color="#42a5f5"
        ))
        fig_bar.add_trace(go.Bar(
            name="Buy & Hold",
            x=list(valid.keys()),
            y=[d["metrics"]["buy_hold_return"] for d in valid.values()],
            marker_color="#ffa726"
        ))
        fig_bar.update_layout(barmode="group", height=350,
                               yaxis_title="Rendement (%)",
                               margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig_bar, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 7 — Opportunités (screener + dips)
# ══════════════════════════════════════════════════════════════════════════════
with tab7:
    st.header("💡 Opportunités du Moment")

    opp_col1, opp_col2 = st.columns([1, 1])
    with opp_col1:
        sectors_selected = st.multiselect(
            "Secteurs à analyser",
            options=list(UNIVERSE.keys()),
            default=list(UNIVERSE.keys()),
        )
    with opp_col2:
        min_drop = st.slider("Baisse min. depuis plus haut 52s (%)", 5, 50, 15,
                             help="Pour le détecteur de bonnes affaires")

    tickers_scope = [t for s, tickers in UNIVERSE.items()
                     if s in sectors_selected for t in tickers]

    col_screen, col_dip = st.columns(2)

    with col_screen:
        if st.button("🔍 Meilleures actions du moment", use_container_width=True):
            with st.spinner(f"Analyse de {len(tickers_scope)} actions…"):
                @st.cache_data(ttl=1800)
                def run_screener(tickers_tuple):
                    return find_best_stocks(list(tickers_tuple), top_n=12)
                df_best = run_screener(tuple(tickers_scope))

            st.subheader("🏆 Top actions — Score global")
            st.caption("Combinaison technique (Golden Cross, RSI, momentum) + fondamentaux (analystes, croissance, PER)")

            for _, row in df_best.iterrows():
                score = row["score"]
                color = "#26a69a" if score >= 50 else "#ffa726" if score >= 25 else "#ef5350"
                upside_str = f"+{row['upside']}%" if row.get("upside") else "N/A"
                rev_str = f"+{row['rev_growth']}%" if row.get("rev_growth") else "N/A"
                st.markdown(f"""
<div style="border-left: 4px solid {color}; padding: 10px 14px; margin-bottom: 8px; background: #0d1117; border-radius: 0 8px 8px 0;">
  <div style="display:flex; justify-content:space-between; align-items:center;">
    <div>
      <b style="color:{color}; font-size:1.1em;">{row['ticker']}</b>
      <span style="color:#888; font-size:0.85em; margin-left:8px;">{row.get('name','')[:35]}</span>
      <span style="color:#555; font-size:0.8em; margin-left:8px;">{row.get('sector','')}</span>
    </div>
    <div style="text-align:right;">
      <b style="color:{color};">Score {score}</b>
      <span style="color:#888; font-size:0.85em; margin-left:12px;">${row['price']}</span>
    </div>
  </div>
  <div style="margin-top:6px; display:flex; gap:16px; flex-wrap:wrap;">
    <span style="color:#aaa; font-size:0.8em;">Golden Cross : {row['golden_cross']}</span>
    <span style="color:#aaa; font-size:0.8em;">RSI : {row['rsi']}</span>
    <span style="color:#aaa; font-size:0.8em;">1 mois : {row['ret_1m']:+.1f}%</span>
    <span style="color:#aaa; font-size:0.8em;">Consensus : {row['recommendation']}</span>
    <span style="color:#26a69a; font-size:0.8em;">Objectif : {upside_str}</span>
    <span style="color:#42a5f5; font-size:0.8em;">Croiss. CA : {rev_str}</span>
  </div>
</div>
""", unsafe_allow_html=True)

    with col_dip:
        if st.button("📉 Bonnes affaires (baisses récentes)", use_container_width=True):
            with st.spinner(f"Recherche de baisses significatives…"):
                @st.cache_data(ttl=1800)
                def run_dip(tickers_tuple, min_drop):
                    return find_dip_opportunities(list(tickers_tuple), min_drop_pct=min_drop, top_n=12)
                df_dips = run_dip(tuple(tickers_scope), min_drop)

            st.subheader(f"🛒 Actions en baisse ≥{min_drop}% avec bon potentiel")
            st.caption("Qualité fondamentale + chute temporaire = opportunité d'achat potentielle")

            if df_dips.empty:
                st.info(f"Aucune action en baisse de plus de {min_drop}% dans la sélection actuelle.")
            else:
                for _, row in df_dips.iterrows():
                    drop = row["drop_from_high"]
                    score = row["score"]
                    upside_str = f"+{row['upside']}%" if row.get("upside") else "N/A"
                    low_dist = row.get("pct_from_low", 0)
                    # Gauge: where is price between 52w low and high?
                    gauge = max(0, min(100, 100 - drop))
                    st.markdown(f"""
<div style="border-left: 4px solid #ffa726; padding: 10px 14px; margin-bottom: 8px; background: #0d1117; border-radius: 0 8px 8px 0;">
  <div style="display:flex; justify-content:space-between; align-items:center;">
    <div>
      <b style="color:#ffa726; font-size:1.1em;">{row['ticker']}</b>
      <span style="color:#888; font-size:0.85em; margin-left:8px;">{row.get('name','')[:35]}</span>
    </div>
    <div style="text-align:right;">
      <b style="color:#ef5350;">-{drop}% depuis haut 52s</b>
      <span style="color:#888; font-size:0.85em; margin-left:8px;">${row['price']}</span>
    </div>
  </div>
  <div style="margin: 6px 0 4px 0; background:#222; border-radius:4px; height:6px;">
    <div style="width:{gauge}%; background: linear-gradient(90deg, #ef5350, #ffa726, #26a69a); height:6px; border-radius:4px;"></div>
  </div>
  <div style="display:flex; gap:16px; flex-wrap:wrap; margin-top:4px;">
    <span style="color:#aaa; font-size:0.8em;">RSI : {row['rsi']}</span>
    <span style="color:#aaa; font-size:0.8em;">Consensus : {row['recommendation']}</span>
    <span style="color:#26a69a; font-size:0.8em;">Objectif : {upside_str}</span>
    <span style="color:#aaa; font-size:0.8em;">Golden Cross : {row['golden_cross']}</span>
    <span style="color:#42a5f5; font-size:0.8em;">Score qualité : {score}</span>
  </div>
</div>
""", unsafe_allow_html=True)

    st.markdown("---")
    st.caption(
        "⚠️ Ces analyses sont générées automatiquement à partir de données publiques. "
        "Elles ne constituent pas un conseil en investissement. "
        "Toujours vérifier avant d'investir."
    )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 8 — Mon Portefeuille (screenshot analysis)
# ══════════════════════════════════════════════════════════════════════════════
with tab8:
    st.header("📸 Analyse de Mon Portefeuille Réel")
    st.markdown(
        "Uploadez un screenshot de votre portefeuille (Boursorama, Degiro, Trading212, Interactive Brokers…). "
        "L'IA extrait vos positions, les analyse et propose des ajustements."
    )

    api_key = st.text_input(
        "🔑 Clé API Anthropic",
        type="password",
        help="Obtenez votre clé sur console.anthropic.com — elle n'est jamais stockée."
    )

    uploaded_file = st.file_uploader(
        "📂 Déposez votre screenshot ici",
        type=["png", "jpg", "jpeg", "webp"],
        help="Capture d'écran de votre interface broker ou relevé de portefeuille"
    )

    if uploaded_file and api_key:
        import os
        os.environ["ANTHROPIC_API_KEY"] = api_key

        image_bytes = uploaded_file.read()
        ext = uploaded_file.name.split(".")[-1].lower()
        media_map = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png", "webp": "image/webp"}
        media_type = media_map.get(ext, "image/png")

        # Show the uploaded image
        st.image(image_bytes, caption="Screenshot uploadé", use_container_width=True)

        if st.button("🚀 Analyser mon portefeuille"):
            # Step 1 — extract positions
            with st.spinner("🔍 Lecture du portefeuille par l'IA…"):
                try:
                    positions = extract_portfolio_from_image(image_bytes, media_type)
                except Exception as e:
                    st.error(f"Erreur de lecture : {e}")
                    st.stop()

            if not positions:
                st.warning("Aucune position détectée. Essayez avec un screenshot plus lisible.")
                st.stop()

            st.success(f"✅ {len(positions)} position(s) détectée(s)")

            # Show extracted table
            df_pos = pd.DataFrame(positions)
            display_cols = [c for c in ["ticker", "name", "quantity", "avg_price",
                                         "current_price", "current_value", "gain_loss_pct"]
                            if c in df_pos.columns]
            st.dataframe(df_pos[display_cols], use_container_width=True, hide_index=True)

            # Step 2 — analyse each ticker
            st.subheader("📊 Analyse par position")
            analyses = {}
            tickers_found = [p["ticker"] for p in positions if p.get("ticker")]

            progress = st.progress(0)
            for i, t in enumerate(tickers_found):
                with st.spinner(f"Analyse de {t}…"):
                    try:
                        df_t = fetch_ohlcv(t, "5y")
                        sig_t = compute_signals(df_t, ma_fast, ma_slow)
                        bt_t = run_backtest(sig_t, initial_capital=10_000)
                        fund_t = get_fundamentals(t)
                        sc_t = compute_scenarios(sig_t, fund_t, macro if "macro" in dir() else {})
                        analyses[t] = {
                            "signals": sig_t,
                            "metrics": bt_t["metrics"],
                            "fund": fund_t,
                            "scenarios": sc_t["scenarios"],
                            "dominant": sc_t["dominant"],
                            "signal_details": sc_t["signals"],
                        }
                    except Exception as e:
                        analyses[t] = {"error": str(e)}
                progress.progress((i + 1) / len(tickers_found))

            # Show per-ticker scenario cards
            cols = st.columns(min(len(tickers_found), 3))
            for i, t in enumerate(tickers_found):
                a = analyses.get(t, {})
                with cols[i % 3]:
                    if "error" in a:
                        st.error(f"**{t}** — {a['error']}")
                        continue
                    dominant = a["dominant"]
                    sc = a["scenarios"][dominant]
                    color = sc["color"]
                    prob = sc["probability"]
                    pos_data = next((p for p in positions if p["ticker"] == t), {})
                    gain = pos_data.get("gain_loss_pct")
                    gain_str = f"{gain:+.1f}%" if gain is not None else "N/A"
                    st.markdown(f"""
<div style="border: 2px solid {color}; border-radius:10px; padding:12px; margin-bottom:10px;">
  <h4 style="color:{color}; margin:0;">{t}</h4>
  <p style="margin:4px 0; color:#ccc;">{a['fund'].get('name', '')}</p>
  <b style="color:{color};">{sc['label']} — {prob}%</b><br/>
  <span style="color:#aaa; font-size:0.85em;">P&L actuel : {gain_str}</span><br/>
  <span style="color:#aaa; font-size:0.85em;">Consensus : {(a['fund'].get('recommendation') or 'N/A').upper()}</span>
</div>
""", unsafe_allow_html=True)

            # Step 3 — global recommendation
            st.subheader("🎯 Recommandations d'ajustement")
            with st.spinner("Génération des recommandations par l'IA…"):
                try:
                    recommendations = generate_portfolio_recommendations(positions, analyses)
                    st.markdown(recommendations)
                except Exception as e:
                    st.error(f"Erreur lors de la génération des recommandations : {e}")

    elif uploaded_file and not api_key:
        st.warning("Veuillez entrer votre clé API Anthropic pour lancer l'analyse.")
    elif not uploaded_file:
        st.info("👆 Uploadez un screenshot pour commencer.")

        st.markdown("""
**Formats acceptés :** PNG, JPG, WEBP

**Brokers compatibles :** Boursorama, Degiro, Trading212, Interactive Brokers,
Saxo Bank, Fortuneo, BinckBank, eToro, Robinhood, et tout autre broker
dont l'interface affiche les positions avec tickers ou noms d'entreprises.

**Ce que l'analyse produit :**
- Extraction automatique de vos positions
- Scénario Bull/Base/Bear pour chaque ligne
- Score de risque global du portefeuille
- Propositions concrètes d'ajustement (alléger, renforcer, couper)
""")


# ══════════════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════════════
# TAB 9 — Paper Trading
# ══════════════════════════════════════════════════════════════════════════════
with tab9:
    st.header("💼 Portefeuille Paper Trading")
    portfolio = get_portfolio()
    total_value = portfolio["cash"]
    positions_data = []
    for sym, qty in portfolio["positions"].items():
        try:
            price_df = fetch_ohlcv(sym, period="5d")
            p = price_df["close"].iloc[-1]
            val = qty * p
            total_value += val
            positions_data.append({"Ticker": sym, "Quantité": round(qty, 4),
                                    "Prix actuel": f"${p:,.2f}", "Valeur": f"${val:,.2f}"})
        except Exception:
            positions_data.append({"Ticker": sym, "Quantité": round(qty, 4),
                                    "Prix actuel": "N/A", "Valeur": "N/A"})

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Valeur totale", f"${total_value:,.2f}")
    col_b.metric("Cash disponible", f"${portfolio['cash']:,.2f}")
    col_c.metric("Positions ouvertes", len(portfolio["positions"]))

    if positions_data:
        st.subheader("Positions")
        st.dataframe(pd.DataFrame(positions_data), use_container_width=True)

    st.subheader(f"Passer un ordre — {ticker}")
    current_position = df_sig["position"].iloc[-1]
    signal_text = "🟢 En position (Golden Cross actif)" if current_position == 1 else "🔴 Hors position (Death Cross actif)"
    st.info(f"Signal actuel : {signal_text}")

    col_buy, col_sell = st.columns(2)
    with col_buy:
        if st.button(f"✅ Acheter {ticker} à ${latest_price:,.2f}"):
            result = place_order(ticker, "buy", latest_price)
            if result["ok"]:
                st.success(f"Ordre exécuté : {result['order']['qty']:.4f} actions achetées")
                st.rerun()
            else:
                st.error(result["error"])
    with col_sell:
        if st.button(f"❌ Vendre {ticker} à ${latest_price:,.2f}"):
            result = place_order(ticker, "sell", latest_price)
            if result["ok"]:
                st.success(f"Ordre exécuté : {result['order']['qty']:.4f} actions vendues")
                st.rerun()
            else:
                st.error(result["error"])

    st.subheader("Historique des ordres")
    if portfolio["history"]:
        hist_df = pd.DataFrame(portfolio["history"])
        st.dataframe(hist_df.set_index("date"), use_container_width=True)
    else:
        st.info("Aucun ordre passé pour l'instant.")
