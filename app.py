"""
app.py
------
Metals Co-Pilot — production-grade Streamlit dashboard for India-focused
intelligent dip-based metals investing.

Run with:
    streamlit run app.py

Modes:
    - demo (default): synthetic data, no internet needed
    - live          : pulls from Yahoo Finance
    - csv           : reads ./data/<SYMBOL>.csv

DISCLAIMER: This is an educational decision-support tool. It does not provide
financial, legal, or tax advice. Verify every recommendation independently
before deploying capital. Commodities are volatile.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yaml
from plotly.subplots import make_subplots

from allocation_engine import UserInputs, recommend_today, rebalance_signals
from backtest import run_all_strategies
from data_providers import make_provider
from indicators import compute_indicator_pack
from portfolio import Holding, Portfolio, build_alerts
from scoring_engine import score_metal, MetalScore
from thesis import THESIS, INSTRUMENTS


# ---------------------------------------------------------------------------
# Page config + theming
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Metals Co-Pilot",
    page_icon="🪙",
    layout="wide",
    initial_sidebar_state="expanded",
)

DARK_CSS = """
<style>
:root {
  --bg: #0d1117;
  --bg-soft: #161b22;
  --border: #30363d;
  --text: #e6edf3;
  --muted: #8b949e;
  --gold: #E6B800;
  --silver: #C0C0C0;
  --copper: #B87333;
  --green: #3fb950;
  --red: #f85149;
  --amber: #d29922;
}
.stApp { background: var(--bg) !important; }
section[data-testid="stSidebar"] { background: var(--bg-soft) !important; border-right: 1px solid var(--border); }
h1, h2, h3, h4 { color: var(--text) !important; letter-spacing: -0.01em; }
p, span, label, div { color: var(--text); }
.metal-card {
  background: linear-gradient(180deg, var(--bg-soft) 0%, #0f141b 100%);
  border: 1px solid var(--border);
  border-radius: 14px;
  padding: 16px;
  height: 100%;
}
.metal-card h4 { margin: 0 0 6px 0; font-size: 14px; color: var(--muted); font-weight: 500; }
.metal-card .price { font-size: 24px; font-weight: 700; color: var(--text); }
.metal-card .delta-pos { color: var(--green); font-weight: 600; }
.metal-card .delta-neg { color: var(--red); font-weight: 600; }
.metal-card .score-pill {
  display: inline-block; padding: 2px 10px; border-radius: 999px;
  font-size: 12px; font-weight: 600; margin-top: 8px;
}
.score-extreme { background: rgba(229,62,62,0.18); color: #ff8a8a; }
.score-strong { background: rgba(63,185,113,0.20); color: #5dd285; }
.score-buy { background: rgba(63,185,113,0.12); color: #5dd285; }
.score-small { background: rgba(210,153,34,0.18); color: #f1c34c; }
.score-watch { background: rgba(139,148,158,0.18); color: #c5cdd5; }
.score-avoid { background: rgba(248,81,73,0.18); color: #ff8a8a; }
.headline-card {
  background: linear-gradient(135deg, rgba(230,184,0,0.10) 0%, rgba(184,115,51,0.06) 100%);
  border: 1px solid var(--border);
  border-left: 4px solid var(--gold);
  border-radius: 14px;
  padding: 22px 26px;
  margin: 8px 0 16px 0;
}
.headline-card h2 { margin: 0 0 6px 0; font-size: 22px; color: var(--text); }
.headline-card .sub { color: var(--muted); font-size: 13px; }
.callout { padding: 10px 14px; border-radius: 10px; margin: 6px 0; font-size: 14px; }
.callout-info { background: rgba(56,139,253,0.10); border: 1px solid rgba(56,139,253,0.35); }
.callout-warn { background: rgba(210,153,34,0.10); border: 1px solid rgba(210,153,34,0.35); }
.callout-danger { background: rgba(248,81,73,0.10); border: 1px solid rgba(248,81,73,0.35); }
.disclaimer {
  background: rgba(139,148,158,0.07); border: 1px solid var(--border);
  border-radius: 8px; padding: 10px 14px; font-size: 12px; color: var(--muted);
}
[data-testid="stMetricValue"] { color: var(--text) !important; }
[data-testid="stMetricLabel"] { color: var(--muted) !important; }
.stTabs [data-baseweb="tab-list"] { gap: 4px; }
.stTabs [data-baseweb="tab"] {
  background: var(--bg-soft); border: 1px solid var(--border);
  border-radius: 8px 8px 0 0; padding: 10px 18px; color: var(--muted);
}
.stTabs [aria-selected="true"] { background: var(--bg) !important; color: var(--text) !important; border-bottom-color: var(--bg); }
hr { border-color: var(--border); }
</style>
"""
st.markdown(DARK_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Config loading (cached)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


CFG = load_config()
METALS_CFG = {k: v for k, v in CFG["metals"].items() if v.get("enabled", True)}
SYMBOLS = list(METALS_CFG.keys())


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🪙 Metals Co-Pilot")
    st.caption("India-focused intelligent metals allocation")

    st.markdown("---")
    st.markdown("**Data source**")
    mode = st.selectbox(
        "Mode", ["demo", "live", "csv"],
        index=["demo", "live", "csv"].index(CFG["app"]["default_mode"]),
        help="Demo = synthetic but realistic series. Live = Yahoo Finance. CSV = ./data folder.",
    )

    st.markdown("---")
    st.markdown("**Strategy**")
    strategy = st.selectbox(
        "Strategy mode",
        ["smart_sip", "dip_accumulator", "momentum_dip", "gs_ratio", "industrial_cycle"],
        index=0,
        format_func=lambda x: {
            "smart_sip": "Smart SIP",
            "dip_accumulator": "Dip Accumulator",
            "momentum_dip": "Momentum + Dip Hybrid",
            "gs_ratio": "Gold-Silver Ratio",
            "industrial_cycle": "Industrial Metals Cycle",
        }[x],
    )

    st.markdown("---")
    st.markdown("**Risk profile**")
    risk_profile = st.selectbox(
        "Risk profile", ["conservative", "balanced", "aggressive"],
        index=["conservative", "balanced", "aggressive"].index(CFG["app"]["default_risk"]),
    )
    horizon_years = st.selectbox("Horizon", [1, 3, 5, 7, 10], index=2)

    st.markdown("---")
    st.markdown("**Capital**")
    available_cash = st.number_input("Available cash today (₹)", min_value=0, value=50000, step=1000)
    monthly_budget = st.number_input("Monthly investment budget (₹)", min_value=0, value=25000, step=1000)
    min_buy = st.number_input("Minimum buy per metal (₹)", min_value=100, value=500, step=100)
    max_per_metal = st.slider("Max allocation to a single metal", 0.20, 1.00, 0.60, 0.05)

    st.markdown("---")
    st.markdown("**Risk controls**")
    using_loan = st.checkbox("Investing borrowed/loan money (more conservative rules)", value=False)
    enable_futures = st.checkbox("Enable MCX futures view (advanced — leveraged)", value=False)

    st.markdown("---")
    st.markdown("**Current holdings**")
    st.caption("Enter your existing exposure. Leave at 0 if none.")

    holdings_inputs = {}
    avg_price_inputs = {}
    with st.expander("Edit holdings", expanded=False):
        for sym in SYMBOLS:
            cfg = METALS_CFG[sym]
            c1, c2 = st.columns(2)
            with c1:
                holdings_inputs[sym] = st.number_input(
                    f"{cfg['display']} invested (₹)", min_value=0, value=0, step=1000, key=f"h_{sym}"
                )
            with c2:
                avg_price_inputs[sym] = st.number_input(
                    f"Avg price ({cfg['inr_unit']})", min_value=0.0, value=0.0, step=10.0, key=f"a_{sym}"
                )

    st.markdown("---")
    st.markdown(
        '<div class="disclaimer">⚠ Educational tool. Not financial, legal, or tax advice. Commodities are volatile. Always verify independently.</div>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Data loading + scoring (cached, mode-aware)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=True, ttl=60 * 30)
def load_histories(mode: str) -> dict[str, pd.DataFrame]:
    provider = make_provider(mode, METALS_CFG)
    out = {}
    for sym in SYMBOLS:
        df = provider.get_history(sym, lookback_days=900)
        if df is not None and not df.empty:
            out[sym] = df
    return out


@st.cache_data(show_spinner=False, ttl=60 * 30)
def compute_packs(mode: str) -> dict:
    histories = load_histories(mode)
    return {sym: compute_indicator_pack(df) for sym, df in histories.items()}


def compute_scores(packs: dict, holdings: dict, risk_profile: str) -> list[MetalScore]:
    targets = CFG["allocation_bands"][risk_profile]
    total_holdings = sum(holdings.values()) or 0.0
    out = []
    for sym, pack in packs.items():
        if not pack:
            continue
        category = METALS_CFG[sym].get("category", "industrial")
        target_pct = float(targets.get(sym, 0))
        cur_pct = (holdings.get(sym, 0) / total_holdings * 100) if total_holdings > 0 else 0.0
        ms = score_metal(sym, pack, CFG, category=category,
                         target_pct=target_pct, current_pct=cur_pct)
        if ms:
            out.append(ms)
    return out


# Load
histories = load_histories(mode)
packs = compute_packs(mode)
scores = compute_scores(packs, holdings_inputs, risk_profile)
scores_by_sym = {s.symbol: s for s in scores}

# Build user inputs for engines
user = UserInputs(
    available_cash_today=available_cash,
    monthly_budget=monthly_budget,
    holdings_inr=holdings_inputs,
    avg_buy_price=avg_price_inputs,
    risk_profile=risk_profile,
    horizon_years=horizon_years,
    strategy=strategy,
    max_per_metal_pct=max_per_metal,
    using_loan_money=using_loan,
    enable_futures=enable_futures,
    min_buy_inr=min_buy,
    refresh_days=7,
)

# Gold-Silver ratio: prefer USD/oz ratio if Yahoo is live; else INR-derived approximation
def compute_gs_ratio(packs: dict) -> float | None:
    g = packs.get("GOLD")
    s = packs.get("SILVER")
    if not g or not s:
        return None
    # INR/10g vs INR/kg comparison: convert silver to INR/10g equivalent
    silver_per_10g = s["price"] / 100.0  # 1 kg = 100 * 10g
    if silver_per_10g <= 0:
        return None
    return g["price"] / silver_per_10g


gs_ratio = compute_gs_ratio(packs)

recommendation = recommend_today(scores, user, CFG, gs_ratio=gs_ratio)

# Portfolio object for alerts
portfolio_obj = Portfolio({
    sym: Holding(symbol=sym, invested_inr=holdings_inputs.get(sym, 0),
                 avg_buy_price=avg_price_inputs.get(sym, 0))
    for sym in SYMBOLS if holdings_inputs.get(sym, 0) > 0
})
current_prices = {sym: pk["price"] for sym, pk in packs.items() if pk}
target_bands = {k: v for k, v in CFG["allocation_bands"][risk_profile].items() if k != "cash"}
alerts = build_alerts(scores, portfolio_obj, current_prices, target_bands, gs_ratio, CFG)


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown(
    f"""
    <div style="display:flex; justify-content:space-between; align-items:flex-end; margin-bottom: 10px;">
      <div>
        <h1 style="margin:0;">🪙 Metals Co-Pilot</h1>
        <span style="color: var(--muted);">India-focused intelligent dip-based investing across precious & industrial metals</span>
      </div>
      <div style="text-align:right; color: var(--muted); font-size: 13px;">
        Mode: <b style="color:var(--text);">{mode.upper()}</b> &nbsp;•&nbsp;
        Strategy: <b style="color:var(--text);">{strategy.replace('_',' ').title()}</b> &nbsp;•&nbsp;
        Risk: <b style="color:var(--text);">{risk_profile.title()}</b><br/>
        {datetime.now().strftime('%A, %d %b %Y')}
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_action, tab_market, tab_metal, tab_alloc, tab_backtest, tab_thesis, tab_alerts = st.tabs([
    "🎯 Today's Action",
    "📊 Market Overview",
    "🔍 Metal Detail",
    "⚖ Allocation & Rebalance",
    "📈 Backtest",
    "📚 Thesis & Instruments",
    "🔔 Alerts",
])


# ===========================================================================
# Helpers used across tabs
# ===========================================================================
ACCENT_BY_SYM = {sym: METALS_CFG[sym].get("accent", "#888") for sym in SYMBOLS}


def class_css(buy_class: str) -> str:
    return {
        "extreme": "score-extreme",
        "strong_buy": "score-strong",
        "buy": "score-buy",
        "small_buy": "score-small",
        "watch": "score-watch",
        "avoid": "score-avoid",
    }.get(buy_class, "score-watch")


def fmt_inr(x: float) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    if x >= 1e7:
        return f"₹{x/1e7:.2f} Cr"
    if x >= 1e5:
        return f"₹{x/1e5:.2f} L"
    if x >= 1000:
        return f"₹{x:,.0f}"
    return f"₹{x:.2f}"


def delta_html(value: float) -> str:
    cls = "delta-pos" if value >= 0 else "delta-neg"
    sign = "+" if value >= 0 else ""
    return f'<span class="{cls}">{sign}{value:.2f}%</span>'


def score_color(val: float, vmin: float = 0.0, vmax: float = 100.0) -> str:
    """
    Pure-Python red->yellow->green gradient for a 0-100 score.
    Avoids the matplotlib dependency that pandas Styler.background_gradient pulls in.
    Returns a CSS background-color string (rgb).
    """
    try:
        v = float(val)
    except (TypeError, ValueError):
        return ""
    if np.isnan(v):
        return ""
    # normalize 0..1
    t = (v - vmin) / max(vmax - vmin, 1e-9)
    t = max(0.0, min(1.0, t))
    # red (220, 53, 69) -> yellow (210, 153, 34) -> green (63, 185, 80)
    if t < 0.5:
        # red -> yellow
        u = t / 0.5
        r = int(220 + (210 - 220) * u)
        g = int(53 + (153 - 53) * u)
        b = int(69 + (34 - 69) * u)
    else:
        # yellow -> green
        u = (t - 0.5) / 0.5
        r = int(210 + (63 - 210) * u)
        g = int(153 + (185 - 153) * u)
        b = int(34 + (80 - 34) * u)
    return f"background-color: rgba({r},{g},{b},0.55); color: #f0f6fc;"


def style_score_column(df: pd.DataFrame, col: str, vmin: float = 0.0, vmax: float = 100.0):
    """Apply score_color to a single column without needing matplotlib."""
    return df.style.apply(
        lambda s: [score_color(v, vmin, vmax) if s.name == col else "" for v in s],
        axis=0,
    )


# ===========================================================================
# TAB 1 — Today's Action
# ===========================================================================
with tab_action:
    # Headline card
    head_color = {
        "extreme": "#f85149", "strong_buy": "#3fb950", "buy": "#3fb950",
        "small_buy": "#d29922", "watch": "#8b949e", "avoid": "#f85149",
    }
    top_score = max(scores, key=lambda s: s.final) if scores else None
    accent = head_color.get(top_score.buy_class, "#E6B800") if top_score else "#E6B800"

    st.markdown(
        f"""
        <div class="headline-card" style="border-left-color: {accent};">
          <h2>{recommendation.headline}</h2>
          <div class="sub">
            Top opportunity: <b style="color:{accent};">{top_score.symbol.title() if top_score else '—'}</b>
            • Score {top_score.final:.0f}/100 ({top_score.buy_class.replace('_',' ').title()})
            • Confidence {recommendation.confidence:.0f}/100
          </div>
        </div>
        """ if top_score else "<div class='headline-card'><h2>No data available</h2></div>",
        unsafe_allow_html=True,
    )

    # 4-up summary
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Deploy today", fmt_inr(recommendation.deploy_amount),
              help="Suggested capital deployment for today, after applying safety rails.")
    c2.metric("Cash remaining", fmt_inr(recommendation.cash_remaining),
              help="What stays as buffer (always preserved).")
    c3.metric("Action", recommendation.action.replace("_", " ").title(),
              help="Buy / Wait / Hold cash / Rebalance.")
    c4.metric("Next review", f"{recommendation.next_review_days} days",
              help="When the dashboard suggests you re-check the recommendation.")

    st.markdown("---")

    # Allocation breakdown
    left, right = st.columns([1.2, 1])

    with left:
        st.subheader("Today's deployment")
        if recommendation.allocations:
            df = pd.DataFrame([{
                "Metal": METALS_CFG[a.symbol]["display"],
                "Symbol": a.symbol,
                "Amount (₹)": a.amount,
                "Share": a.weight,
                "Reason": a.reason,
            } for a in recommendation.allocations])

            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=df["Amount (₹)"], y=df["Metal"], orientation="h",
                marker=dict(color=[ACCENT_BY_SYM[s] for s in df["Symbol"]]),
                text=[fmt_inr(a) for a in df["Amount (₹)"]], textposition="outside",
                hovertemplate="%{y}: ₹%{x:,.0f}<extra></extra>",
            ))
            fig.update_layout(
                height=260, margin=dict(l=10, r=40, t=10, b=10),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#e6edf3"), xaxis=dict(showgrid=False), yaxis=dict(showgrid=False),
            )
            st.plotly_chart(fig, use_container_width=True)

            st.dataframe(
                df[["Metal", "Amount (₹)", "Share", "Reason"]].style.format({
                    "Amount (₹)": lambda v: f"₹{v:,.0f}",
                    "Share": "{:.1%}",
                }),
                hide_index=True, use_container_width=True,
            )
        else:
            st.info("No deployment recommended today. Reasoning below.")

    with right:
        st.subheader("Why this recommendation")
        for r in recommendation.reasoning:
            st.markdown(f"<div class='callout callout-info'>• {r}</div>", unsafe_allow_html=True)

        if recommendation.risk_warnings:
            st.subheader("Risk notes")
            for w in recommendation.risk_warnings:
                st.markdown(f"<div class='callout callout-warn'>⚠ {w}</div>", unsafe_allow_html=True)

        if top_score:
            st.subheader("What would change this")
            tweaks = []
            if top_score.dip_category in {"strong", "extreme"}:
                tweaks.append("If RSI bounces above 50 or price closes above the 50DMA, the dip thesis weakens.")
            if top_score.trend < 50:
                tweaks.append("If price recovers above the 200DMA, trend score lifts and confidence improves.")
            if top_score.risk > 65:
                tweaks.append("If 30-day volatility falls, the risk score drops and position sizes can grow.")
            if not tweaks:
                tweaks.append("A drop in the score below 46 would shift the action to 'wait'.")
            for t in tweaks:
                st.markdown(f"<div class='callout callout-info'>↳ {t}</div>", unsafe_allow_html=True)


# ===========================================================================
# TAB 2 — Market Overview (cards + heatmap + ranking)
# ===========================================================================
with tab_market:
    st.subheader("Live snapshot")

    # Cards: 3 per row
    cards_per_row = 3
    rows = [scores[i:i + cards_per_row] for i in range(0, len(scores), cards_per_row)]

    for row in rows:
        cols = st.columns(cards_per_row)
        for col, s in zip(cols, row):
            cfg = METALS_CFG[s.symbol]
            ret_1d = s.pack.get("ret_1d", 0)
            ret_1m = s.pack.get("ret_1m", 0)
            ret_3m = s.pack.get("ret_3m", 0)
            html = f"""
            <div class="metal-card" style="border-top: 3px solid {cfg['accent']};">
              <h4>{cfg['display']} <span style="float:right; font-size:11px;">{cfg['inr_unit']}</span></h4>
              <div class="price">{fmt_inr(s.price)}</div>
              <div style="margin-top: 4px; font-size:13px;">
                1D {delta_html(ret_1d)} &nbsp;•&nbsp; 1M {delta_html(ret_1m)} &nbsp;•&nbsp; 3M {delta_html(ret_3m)}
              </div>
              <div style="margin-top: 10px; color: var(--muted); font-size: 12px;">
                52W: {fmt_inr(s.pack.get('low_52w',0))} – {fmt_inr(s.pack.get('high_52w',0))}<br/>
                From high: {s.pack.get('dist_from_high_pct',0):.1f}% • From 200DMA: {s.pack.get('dist_from_200dma_pct',0):+.1f}%<br/>
                RSI {s.pack.get('rsi14',50):.0f} • Vol {s.pack.get('vol_30',0)*100:.1f}%
              </div>
              <div style="margin-top:10px;">
                <span class="score-pill {class_css(s.buy_class)}">{s.final:.0f}/100 · {s.buy_class.replace('_',' ').upper()}</span>
                <span class="score-pill score-watch" style="margin-left:6px;">Risk {s.risk:.0f}</span>
              </div>
            </div>
            """
            col.markdown(html, unsafe_allow_html=True)

    st.markdown("---")

    # Opportunity heatmap (sub-scores)
    st.subheader("Score heatmap")
    sub_cols = ["Dip", "Trend", "Rel Value", "Vol-Adj Opp", "Portfolio Need", "Macro", "Final"]
    heat_df = pd.DataFrame({
        "Metal": [METALS_CFG[s.symbol]["display"] for s in scores],
        "Dip": [s.dip for s in scores],
        "Trend": [s.trend for s in scores],
        "Rel Value": [s.rel_value for s in scores],
        "Vol-Adj Opp": [s.vol_adj_opp for s in scores],
        "Portfolio Need": [s.portfolio_need for s in scores],
        "Macro": [s.macro for s in scores],
        "Final": [s.final for s in scores],
    }).set_index("Metal")

    heat = go.Figure(data=go.Heatmap(
        z=heat_df.values, x=heat_df.columns, y=heat_df.index,
        colorscale="RdYlGn", zmin=0, zmax=100,
        text=[[f"{v:.0f}" for v in row] for row in heat_df.values],
        texttemplate="%{text}", textfont={"size": 11, "color": "white"},
        hovertemplate="%{y} · %{x}: %{z:.0f}<extra></extra>",
    ))
    heat.update_layout(
        height=380, margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e6edf3"),
    )
    st.plotly_chart(heat, use_container_width=True)

    # Ranking
    st.subheader("Ranking")
    rank_df = pd.DataFrame([{
        "Metal": METALS_CFG[s.symbol]["display"],
        "Price": s.price,
        "1D %": s.pack.get("ret_1d", 0),
        "1M %": s.pack.get("ret_1m", 0),
        "3M %": s.pack.get("ret_3m", 0),
        "From High %": s.pack.get("dist_from_high_pct", 0),
        "RSI": s.pack.get("rsi14", 50),
        "Dip Cat.": s.dip_category.replace("_", " "),
        "Final": s.final,
        "Class": s.buy_class.replace("_", " "),
        "Risk": s.risk,
        "Conf.": s.confidence,
    } for s in sorted(scores, key=lambda x: x.final, reverse=True)])

    st.dataframe(
        rank_df.style.format({
            "Price": "{:,.2f}", "1D %": "{:+.2f}", "1M %": "{:+.2f}", "3M %": "{:+.2f}",
            "From High %": "{:+.1f}", "RSI": "{:.0f}", "Final": "{:.0f}",
            "Risk": "{:.0f}", "Conf.": "{:.0f}",
        }).apply(
            lambda s: [score_color(v, 0, 100) if s.name == "Final" else "" for v in s],
            axis=0,
        ),
        hide_index=True, use_container_width=True,
    )

    # 52w drawdown bar chart
    st.subheader("52-week drawdown from peak")
    dd_df = pd.DataFrame({
        "Metal": [METALS_CFG[s.symbol]["display"] for s in scores],
        "Drawdown %": [s.pack.get("drawdown_pct", 0) for s in scores],
    }).sort_values("Drawdown %")
    fig_dd = go.Figure(go.Bar(
        x=dd_df["Drawdown %"], y=dd_df["Metal"], orientation="h",
        marker=dict(color=dd_df["Drawdown %"], colorscale="RdYlGn", reversescale=True, cmin=-25, cmax=0),
        hovertemplate="%{y}: %{x:.1f}%<extra></extra>",
    ))
    fig_dd.update_layout(
        height=340, margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e6edf3"), xaxis_title="% from 52w peak",
    )
    st.plotly_chart(fig_dd, use_container_width=True)


# ===========================================================================
# TAB 3 — Metal Detail
# ===========================================================================
with tab_metal:
    if not scores:
        st.warning("No data loaded.")
    else:
        sel_sym = st.selectbox(
            "Select metal",
            options=[s.symbol for s in scores],
            format_func=lambda x: METALS_CFG[x]["display"],
        )
        s = scores_by_sym[sel_sym]
        df = histories[sel_sym]
        series = s.pack["_series"]
        cfg = METALS_CFG[sel_sym]

        # Header metrics
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric(f"{cfg['display']} price", fmt_inr(s.price), help=cfg["inr_unit"])
        c2.metric("Final score", f"{s.final:.0f}/100", help="Composite buy attractiveness")
        c3.metric("Dip category", s.dip_category.replace("_", " ").title())
        c4.metric("Risk", f"{s.risk:.0f}/100")
        c5.metric("Suggested horizon", f"{s.horizon_months} mo")

        # Price chart with MAs + Bollinger
        fig = make_subplots(
            rows=3, cols=1, shared_xaxes=True, row_heights=[0.55, 0.22, 0.23],
            vertical_spacing=0.04,
            subplot_titles=("Price · Moving averages · Bollinger", "RSI 14", "MACD"),
        )

        fig.add_trace(go.Scatter(
            x=df.index, y=series["bb"]["bb_upper"], name="BB Upper",
            line=dict(color="rgba(139,148,158,0.4)", width=1, dash="dot"), showlegend=False,
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=df.index, y=series["bb"]["bb_lower"], name="BB Lower",
            line=dict(color="rgba(139,148,158,0.4)", width=1, dash="dot"),
            fill="tonexty", fillcolor="rgba(139,148,158,0.05)", showlegend=False,
        ), row=1, col=1)

        fig.add_trace(go.Scatter(x=df.index, y=series["close"], name="Close",
                                 line=dict(color=cfg["accent"], width=2)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=series["sma50"], name="50DMA",
                                 line=dict(color="#3fb950", width=1.2)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=series["sma200"], name="200DMA",
                                 line=dict(color="#f85149", width=1.2)), row=1, col=1)

        # RSI
        fig.add_trace(go.Scatter(x=df.index, y=series["rsi"], name="RSI 14",
                                 line=dict(color="#d29922", width=1.5)), row=2, col=1)
        fig.add_hline(y=70, line_dash="dot", line_color="#f85149", row=2, col=1)
        fig.add_hline(y=30, line_dash="dot", line_color="#3fb950", row=2, col=1)

        # MACD
        fig.add_trace(go.Scatter(x=df.index, y=series["macd"]["macd"], name="MACD",
                                 line=dict(color="#58a6ff", width=1.5)), row=3, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=series["macd"]["signal"], name="Signal",
                                 line=dict(color="#d29922", width=1.2)), row=3, col=1)
        fig.add_trace(go.Bar(x=df.index, y=series["macd"]["hist"], name="Hist",
                             marker_color=np.where(series["macd"]["hist"] >= 0, "#3fb950", "#f85149")), row=3, col=1)

        fig.update_layout(
            height=720, margin=dict(l=10, r=10, t=40, b=10),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e6edf3"), legend=dict(orientation="h", y=1.04),
            hovermode="x unified",
        )
        for r in [1, 2, 3]:
            fig.update_xaxes(gridcolor="rgba(139,148,158,0.10)", row=r, col=1)
            fig.update_yaxes(gridcolor="rgba(139,148,158,0.10)", row=r, col=1)

        st.plotly_chart(fig, use_container_width=True)

        # Sub-score radar
        col_a, col_b = st.columns([1, 1])
        with col_a:
            st.subheader("Sub-score radar")
            radar = go.Figure()
            radar.add_trace(go.Scatterpolar(
                r=[s.dip, s.trend, s.rel_value, s.vol_adj_opp, s.portfolio_need, s.macro],
                theta=["Dip", "Trend", "Rel Value", "Vol-Adj Opp", "Portfolio Need", "Macro"],
                fill="toself",
                line=dict(color=cfg["accent"]),
                fillcolor=f"rgba(230,184,0,0.20)",
            ))
            radar.update_layout(
                polar=dict(
                    bgcolor="rgba(0,0,0,0)",
                    radialaxis=dict(visible=True, range=[0, 100], gridcolor="rgba(139,148,158,0.2)"),
                    angularaxis=dict(gridcolor="rgba(139,148,158,0.2)"),
                ),
                paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#e6edf3"),
                height=380, margin=dict(l=10, r=10, t=10, b=10), showlegend=False,
            )
            st.plotly_chart(radar, use_container_width=True)

        with col_b:
            st.subheader("Reasons & levels")
            for r in s.reasons:
                st.markdown(f"<div class='callout callout-info'>• {r}</div>", unsafe_allow_html=True)
            st.markdown(
                f"""
                <div class="metal-card" style="margin-top: 10px;">
                  <div style="color:var(--muted); font-size:12px;">Suggested stop / review price</div>
                  <div class="price">{fmt_inr(s.stop_review_price)}</div>
                  <div style="color:var(--muted); font-size:12px; margin-top:8px;">
                    Re-evaluate the thesis if price closes below this for several sessions.
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


# ===========================================================================
# TAB 4 — Allocation & Rebalance
# ===========================================================================
with tab_alloc:
    targets = {k: v for k, v in CFG["allocation_bands"][risk_profile].items() if k != "cash"}

    left, right = st.columns([1, 1])

    with left:
        st.subheader(f"Target allocation · {risk_profile.title()}")
        labels = [METALS_CFG[k]["display"] if k in METALS_CFG else k.title() for k in targets.keys()]
        values = list(targets.values())
        colors = [METALS_CFG[k]["accent"] if k in METALS_CFG else "#888" for k in targets.keys()]
        pie = go.Figure(go.Pie(
            labels=labels, values=values, hole=0.55, marker=dict(colors=colors),
            textinfo="label+percent",
            hovertemplate="%{label}: %{value}%<extra></extra>",
        ))
        pie.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#e6edf3"),
            height=360, margin=dict(l=10, r=10, t=10, b=10), showlegend=False,
        )
        st.plotly_chart(pie, use_container_width=True)

    with right:
        st.subheader("Current vs target")
        cur_pct = portfolio_obj.current_allocation_pct(current_prices)
        cmp_df = pd.DataFrame({
            "Metal": [METALS_CFG[k]["display"] for k in targets.keys()],
            "Symbol": list(targets.keys()),
            "Target %": list(targets.values()),
            "Current %": [cur_pct.get(k, 0) for k in targets.keys()],
        })
        cmp_df["Drift pp"] = cmp_df["Current %"] - cmp_df["Target %"]

        fig_cmp = go.Figure()
        fig_cmp.add_trace(go.Bar(name="Target",  x=cmp_df["Metal"], y=cmp_df["Target %"],
                                 marker_color="rgba(139,148,158,0.55)"))
        fig_cmp.add_trace(go.Bar(name="Current", x=cmp_df["Metal"], y=cmp_df["Current %"],
                                 marker_color=[ACCENT_BY_SYM.get(s, "#888") for s in cmp_df["Symbol"]]))
        fig_cmp.update_layout(
            barmode="group", height=360,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e6edf3"), legend=dict(orientation="h", y=1.05),
            margin=dict(l=10, r=10, t=10, b=10),
            yaxis=dict(title="%", gridcolor="rgba(139,148,158,0.1)"),
        )
        st.plotly_chart(fig_cmp, use_container_width=True)

    st.markdown("---")

    # Rebalance signals
    st.subheader("Rebalance signals")
    rb_signals = rebalance_signals(scores, user, CFG)
    if not rb_signals:
        st.info("Add holdings in the sidebar to see rebalance signals.")
    else:
        rb_df = pd.DataFrame([{
            "Metal": METALS_CFG[r.symbol]["display"],
            "Action": r.action.replace("_", " ").title(),
            "Drift (pp)": r.drift_pp,
            "Gain %": r.gain_pct,
            "Reason": r.reason,
        } for r in rb_signals])
        st.dataframe(
            rb_df.style.format({"Drift (pp)": "{:+.1f}", "Gain %": "{:+.1f}"}),
            hide_index=True, use_container_width=True,
        )

    # Risk-return scatter
    st.subheader("Risk vs opportunity (per metal)")
    sc_df = pd.DataFrame([{
        "Metal": METALS_CFG[s.symbol]["display"],
        "Risk": s.risk,
        "Final": s.final,
        "Vol": s.pack.get("vol_30", 0) * 100,
        "Symbol": s.symbol,
    } for s in scores])
    fig_sc = go.Figure(go.Scatter(
        x=sc_df["Risk"], y=sc_df["Final"], mode="markers+text",
        text=sc_df["Metal"], textposition="top center",
        marker=dict(size=14 + sc_df["Vol"], color=[ACCENT_BY_SYM[s] for s in sc_df["Symbol"]],
                    line=dict(color="#0d1117", width=1)),
        hovertemplate="%{text}<br>Risk: %{x:.0f}<br>Final: %{y:.0f}<extra></extra>",
    ))
    fig_sc.update_layout(
        height=360, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e6edf3"),
        xaxis=dict(title="Risk score", range=[0, 100], gridcolor="rgba(139,148,158,0.1)"),
        yaxis=dict(title="Final buy score", range=[0, 100], gridcolor="rgba(139,148,158,0.1)"),
        margin=dict(l=10, r=10, t=10, b=10),
    )
    st.plotly_chart(fig_sc, use_container_width=True)

    # Gold-Silver ratio box
    if gs_ratio is not None:
        st.subheader(f"Gold-Silver ratio: {gs_ratio:.1f}")
        thr = CFG["gs_ratio"]
        if gs_ratio >= thr["silver_attractive_above"]:
            verdict = ("Silver looks relatively attractive vs gold (ratio is elevated).", "callout-info")
        elif gs_ratio <= thr["gold_attractive_below"]:
            verdict = ("Gold looks relatively attractive vs silver (ratio is compressed).", "callout-info")
        else:
            verdict = ("Ratio in middle band — no strong tilt either way.", "callout-info")
        st.markdown(f"<div class='callout {verdict[1]}'>{verdict[0]}</div>", unsafe_allow_html=True)


# ===========================================================================
# TAB 5 — Backtest
# ===========================================================================
with tab_backtest:
    st.subheader("Backtest: SIP vs Smart Dip vs Gold-only vs Dynamic")
    st.caption(
        "Compares strategies on the loaded history. Results are gross of taxes, "
        "slippage, and fees — purely for relative comparison."
    )

    bt_amount = st.number_input("Monthly investment for backtest (₹)", min_value=1000, value=20000, step=1000)
    bt_symbols = st.multiselect(
        "Metals universe",
        SYMBOLS,
        default=[s for s in ["GOLD", "SILVER", "COPPER"] if s in SYMBOLS],
        format_func=lambda x: METALS_CFG[x]["display"],
    )

    if st.button("Run backtest", type="primary"):
        with st.spinner("Running backtests..."):
            results = run_all_strategies(histories, bt_amount, bt_symbols, CFG, METALS_CFG)

        # Equity curves
        fig = go.Figure()
        palette = {"normal_sip": "#8b949e", "smart_sip": "#58a6ff",
                   "dip_accumulator": "#3fb950", "gold_only": "#E6B800", "dynamic": "#d29922"}
        for name, res in results.items():
            if res is None or res.equity_curve.empty:
                continue
            fig.add_trace(go.Scatter(
                x=res.equity_curve.index, y=res.equity_curve.values,
                name=name.replace("_", " ").title(), mode="lines",
                line=dict(color=palette.get(name, "#888"), width=2),
            ))
        fig.update_layout(
            height=420, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e6edf3"), legend=dict(orientation="h", y=1.05),
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis=dict(gridcolor="rgba(139,148,158,0.1)"),
            yaxis=dict(title="Portfolio value (₹)", gridcolor="rgba(139,148,158,0.1)"),
        )
        st.plotly_chart(fig, use_container_width=True)

        # Metrics table
        rows = []
        for name, res in results.items():
            if res is None or not res.metrics:
                continue
            m = res.metrics
            rows.append({
                "Strategy": name.replace("_", " ").title(),
                "Invested": m["invested_total"],
                "Final value": m["final_value"],
                "TVPI": m["tvpi"],
                "Total P&L %": m["pnl_pct"],
                "CAGR (approx) %": m["cagr_approx"],
                "Vol (ann) %": m["vol_annual_pct"],
                "Max DD %": m["max_drawdown_pct"],
                "Sharpe-like": m["sharpe_like"],
                "Win rate %": m["win_rate_pct"],
            })
        if rows:
            metrics_df = pd.DataFrame(rows).set_index("Strategy")
            st.dataframe(
                metrics_df.style.format({
                    "Invested": "{:,.0f}", "Final value": "{:,.0f}",
                    "TVPI": "{:.2f}", "Total P&L %": "{:+.1f}",
                    "CAGR (approx) %": "{:+.1f}", "Vol (ann) %": "{:.1f}",
                    "Max DD %": "{:.1f}", "Sharpe-like": "{:.2f}", "Win rate %": "{:.1f}",
                }),
                use_container_width=True,
            )

            # Monthly deployments for the smart_sip / dip_accumulator strategies
            for name in ["smart_sip", "dip_accumulator", "dynamic"]:
                res = results.get(name)
                if not res or not res.deployments:
                    continue
                with st.expander(f"Monthly deployments — {name.replace('_',' ').title()}"):
                    rec = pd.DataFrame([
                        {"date": d["date"], "total": d["total"], **d["splits"]}
                        for d in res.deployments
                    ])
                    if not rec.empty:
                        st.dataframe(
                            rec.set_index("date").style.format("{:,.0f}"),
                            use_container_width=True,
                        )


# ===========================================================================
# TAB 6 — Thesis & Instruments
# ===========================================================================
with tab_thesis:
    st.subheader("Investment thesis by metal")
    sel = st.selectbox(
        "Metal",
        SYMBOLS,
        format_func=lambda x: METALS_CFG[x]["display"],
        key="thesis_select",
    )
    cfg = METALS_CFG[sel]
    th = THESIS.get(cfg.get("thesis_key", ""), {})

    c1, c2 = st.columns([1.2, 1])
    with c1:
        st.markdown(
            f"""
            <div class="metal-card" style="border-top: 3px solid {cfg['accent']};">
              <h3 style="margin-top:0;">{cfg['display']}</h3>
              <div style="color: var(--muted); font-size: 13px; margin-bottom: 12px;">{th.get('headline','')}</div>
              <b>Drivers</b>
              <ul style="margin-top:6px;">{''.join(f'<li>{d}</li>' for d in th.get('drivers', []))}</ul>
              <b>Best when</b><div style="color:var(--muted); margin: 4px 0 12px 0;">{th.get('best_when','—')}</div>
              <b>Watch for</b><div style="color:var(--muted);">{th.get('watch_for','—')}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c2:
        st.markdown("**India instrument routes**")
        for entry in INSTRUMENTS.get(sel, []):
            if isinstance(entry, tuple):
                name, blurb = entry
                st.markdown(f"<div class='callout callout-info'><b>{name}</b><br/><span style='color:var(--muted);'>{blurb}</span></div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='callout callout-info'>{entry}</div>", unsafe_allow_html=True)

        st.markdown(
            "<div class='disclaimer' style='margin-top:14px;'>"
            "Tax notes (India, indicative only): different instruments are taxed differently — "
            "ETFs, SGBs, physical, futures all differ. Consult a qualified tax advisor; "
            "this dashboard does not provide tax advice."
            "</div>", unsafe_allow_html=True,
        )


# ===========================================================================
# TAB 7 — Alerts
# ===========================================================================
with tab_alerts:
    st.subheader("Active alerts")
    if not alerts:
        st.info("No active alerts. Markets look quiet.")
    else:
        for a in alerts:
            cls = {"info": "callout-info", "warn": "callout-warn", "critical": "callout-danger"}[a.severity]
            st.markdown(
                f"""<div class='callout {cls}'>
                  <b>{a.title}</b><br/>
                  <span style="color: var(--muted);">{a.body}</span>
                </div>""",
                unsafe_allow_html=True,
            )

    st.markdown("---")
    st.caption(
        "Future-ready: alerts can be dispatched to Telegram / email / WhatsApp by "
        "implementing a new `AlertSink` in portfolio.py."
    )


# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown("---")
st.markdown(
    "<div class='disclaimer'>"
    "Metals Co-Pilot is an educational decision-support tool. Numbers shown are based on the data source "
    "selected (demo or live) and modeled heuristics, not guaranteed forecasts. Commodities are volatile. "
    "This is not financial, investment, legal, or tax advice. Always verify independently and consult a "
    "qualified advisor before deploying capital."
    "</div>",
    unsafe_allow_html=True,
)
