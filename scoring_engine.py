"""
scoring_engine.py
-----------------
Turns indicator packs into the dashboard's decision-grade scores:

    final_buy_score (0-100)
        = 0.30 * dip_score
        + 0.20 * trend_score
        + 0.15 * rel_value_score
        + 0.15 * vol_adj_opp_score
        + 0.10 * portfolio_need_score
        + 0.10 * macro_score

Plus auxiliary outputs every UI card needs:
    - dip_category   (no_dip | mild | good | strong | extreme | falling_knife)
    - buy_class      (avoid | watch | small_buy | buy | strong_buy | extreme)
    - risk_score     (0-100, higher = riskier)
    - confidence     (0-100, agreement of signals)
    - horizon_months (suggested holding period)
    - stop_review    (price level to re-evaluate at)
    - reasons        (list of short, plain-English bullet strings)

The math here is heuristic, not magic. Every weight, threshold, and clamp lives
in config.yaml so a strategist can tune it without touching code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Scaling helper: clamp + linearly map [lo, hi] -> [0, 100]
# ---------------------------------------------------------------------------
def _scale(x: float, lo: float, hi: float, invert: bool = False) -> float:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return 50.0
    if hi == lo:
        return 50.0
    v = (x - lo) / (hi - lo) * 100
    if invert:
        v = 100 - v
    return float(max(0.0, min(100.0, v)))


# ---------------------------------------------------------------------------
# Dip score
#   Drivers: drawdown from 52w high (deeper = better), RSI (lower = better),
#            distance below 50DMA (small = better dip, large = falling knife),
#            distance from 200DMA (must still be near/above for "healthy" dip)
# ---------------------------------------------------------------------------
def dip_score(pack: dict) -> float:
    if not pack:
        return 0.0

    # Drawdown: 0% -> 10 score, -15% -> 90 score, capped
    dd = pack.get("drawdown_pct", 0)  # negative
    s_dd = _scale(dd, -15, 0, invert=True)

    # RSI: 70 -> 0, 30 -> 100
    rsi_v = pack.get("rsi14", 50)
    s_rsi = _scale(rsi_v, 30, 70, invert=True)

    # Bollinger position: 0 (at lower band) -> 100, 1 (upper band) -> 0
    bb_pos = pack.get("bb_pos", 0.5)
    s_bb = _scale(bb_pos, 0, 1, invert=True)

    # 200DMA proximity: a "healthy dip" is near or just above 200DMA
    dist200 = pack.get("dist_from_200dma_pct", 0)
    if dist200 < -10:                  # well below 200DMA -> falling knife penalty
        s_200 = _scale(dist200, -25, -10, invert=False)  # -25 -> 0, -10 -> 100
        s_200 *= 0.5
    elif dist200 < 0:
        s_200 = 80
    elif dist200 < 5:
        s_200 = 90
    else:
        # Far above 200DMA -> not really a dip
        s_200 = _scale(dist200, 5, 25, invert=True)

    return float(0.35 * s_dd + 0.30 * s_rsi + 0.15 * s_bb + 0.20 * s_200)


# ---------------------------------------------------------------------------
# Trend score
#   Drivers: price vs 200DMA, 50DMA vs 200DMA, ADX strength, 6M return
# ---------------------------------------------------------------------------
def trend_score(pack: dict) -> float:
    if not pack:
        return 0.0

    price = pack.get("price", 0)
    sma50 = pack.get("sma50", price)
    sma200 = pack.get("sma200", price)
    adx_v = pack.get("adx14", 15)
    ret_6m = pack.get("ret_6m", 0)

    s_above_200 = 80 if price >= sma200 else 30
    s_50_vs_200 = 80 if sma50 >= sma200 else 30
    s_adx = _scale(adx_v, 10, 35)             # stronger trend -> higher score
    s_6m = _scale(ret_6m, -20, 25)            # stronger 6M return -> higher

    return float(0.30 * s_above_200 + 0.20 * s_50_vs_200 + 0.20 * s_adx + 0.30 * s_6m)


# ---------------------------------------------------------------------------
# Relative value (vs metal's own 1Y mean)
#   Drivers: 1Y z-score (more negative = better value)
# ---------------------------------------------------------------------------
def rel_value_score(pack: dict) -> float:
    z = pack.get("zscore_252", 0)
    return _scale(z, -2.0, 2.0, invert=True)


# ---------------------------------------------------------------------------
# Volatility-adjusted opportunity
#   Same drawdown means more for a low-vol metal than a high-vol metal.
# ---------------------------------------------------------------------------
def vol_adj_opp_score(pack: dict) -> float:
    dd = abs(pack.get("drawdown_pct", 0))
    vol = max(pack.get("vol_30", 0.2), 0.05)
    ratio = dd / (vol * 100)                 # dd-% per 1 vol-point
    return _scale(ratio, 0, 0.6)


# ---------------------------------------------------------------------------
# Portfolio need
#   How underweight is this metal vs its target % for the chosen risk profile?
# ---------------------------------------------------------------------------
def portfolio_need_score(symbol: str,
                         target_pct: float,
                         current_pct: float) -> float:
    """All inputs in % (0-100)."""
    if target_pct <= 0:
        return 30.0
    gap = target_pct - current_pct           # positive => underweight
    # Map gap as % points to 0-100. 0 gap -> 50, +20pp gap -> 100, -20pp -> 0
    return _scale(gap, -20, 20)


# ---------------------------------------------------------------------------
# Macro proxy
#   We don't fetch external macro data in demo mode. Use the metal's own 1Y
#   trend as a stand-in: positive long-term return => favourable macro for it.
#   Wire real CPI / USDINR / 10Y yield / PMI inputs via this function later.
# ---------------------------------------------------------------------------
def macro_score(pack: dict, category: str = "industrial") -> float:
    ret_1y = pack.get("ret_1y", 0)
    base = _scale(ret_1y, -25, 30)
    # Precious metals get a small floor (perma safe-haven utility)
    if category == "precious":
        base = max(base, 45)
    return base


# ---------------------------------------------------------------------------
# Final score + classification
# ---------------------------------------------------------------------------
@dataclass
class MetalScore:
    symbol: str
    price: float
    dip: float
    trend: float
    rel_value: float
    vol_adj_opp: float
    portfolio_need: float
    macro: float
    final: float
    risk: float
    confidence: float
    dip_category: str
    buy_class: str
    horizon_months: int
    stop_review_price: float
    reasons: list = field(default_factory=list)
    pack: dict = field(default_factory=dict)


def classify_buy(final: float) -> str:
    if final >= 91:
        return "extreme"
    if final >= 76:
        return "strong_buy"
    if final >= 61:
        return "buy"
    if final >= 46:
        return "small_buy"
    if final >= 31:
        return "watch"
    return "avoid"


def classify_dip(pack: dict, cfg_thresholds: dict) -> str:
    dd = pack.get("drawdown_pct", 0)         # negative
    rsi_v = pack.get("rsi14", 50)
    dist200 = pack.get("dist_from_200dma_pct", 0)
    macd_h = pack.get("macd_hist", 0)
    vol = pack.get("vol_30", 0.2)

    falling_knife = dist200 < -10 and macd_h < 0 and vol > 0.30
    if falling_knife:
        return "falling_knife"

    abs_dd = abs(dd)
    if abs_dd >= cfg_thresholds.get("extreme_pct", 15) and rsi_v < 35 and dist200 > -10:
        return "extreme"
    if abs_dd >= cfg_thresholds.get("strong_pct", 10) and dist200 > -8:
        return "strong"
    if abs_dd >= cfg_thresholds.get("good_pct", 6):
        return "good"
    if abs_dd >= cfg_thresholds.get("mild_pct", 3):
        return "mild"
    return "no_dip"


# ---------------------------------------------------------------------------
# Risk and confidence
# ---------------------------------------------------------------------------
def risk_score(pack: dict, dip_cat: str) -> float:
    vol = pack.get("vol_30", 0.2)
    dist200 = pack.get("dist_from_200dma_pct", 0)
    macd_h = pack.get("macd_hist", 0)

    s_vol = _scale(vol, 0.10, 0.45)                 # higher vol -> riskier
    # Far below 200DMA OR far above (overheated) both increase risk
    s_dist = _scale(abs(dist200), 0, 25)
    s_mom = 60 if macd_h < 0 else 30                # negative momentum -> riskier
    base = 0.45 * s_vol + 0.35 * s_dist + 0.20 * s_mom
    if dip_cat == "falling_knife":
        base = max(base, 85)
    return float(min(100, base))


def confidence_score(dip: float, trend: float, rel_val: float, macro: float) -> float:
    """Confidence is high when independent dimensions agree."""
    arr = np.array([dip, trend, rel_val, macro], dtype=float)
    mean = arr.mean()
    spread = arr.std()
    # Lower spread + higher mean -> higher confidence
    s_mean = _scale(mean, 30, 80)
    s_agree = _scale(spread, 5, 30, invert=True)
    return float(0.6 * s_agree + 0.4 * s_mean)


# ---------------------------------------------------------------------------
# Plain-English reason builder
# ---------------------------------------------------------------------------
def build_reasons(pack: dict, dip_cat: str, trend: float) -> list:
    reasons = []
    rsi_v = pack.get("rsi14", 50)
    dd = pack.get("drawdown_pct", 0)
    dist200 = pack.get("dist_from_200dma_pct", 0)
    z = pack.get("zscore_252", 0)
    macd_h = pack.get("macd_hist", 0)

    if dd <= -10:
        reasons.append(f"Down {abs(dd):.1f}% from 52-week peak.")
    elif dd <= -5:
        reasons.append(f"Mild pullback of {abs(dd):.1f}% from peak.")

    if rsi_v < 35:
        reasons.append(f"RSI at {rsi_v:.0f} — oversold zone.")
    elif rsi_v > 70:
        reasons.append(f"RSI at {rsi_v:.0f} — overbought, caution.")

    if dist200 > 2:
        reasons.append(f"Price {dist200:.1f}% above 200DMA — long-term trend intact.")
    elif dist200 < -5:
        reasons.append(f"Price {abs(dist200):.1f}% below 200DMA — long-term trend weak.")

    if z < -1.0:
        reasons.append("Trades >1σ below 1-year mean — relatively cheap.")
    elif z > 1.0:
        reasons.append("Trades >1σ above 1-year mean — relatively expensive.")

    if macd_h > 0 and trend >= 60:
        reasons.append("MACD turning positive with healthy uptrend.")
    elif macd_h < 0:
        reasons.append("MACD negative — momentum still soft.")

    if dip_cat == "falling_knife":
        reasons.append("⚠ Falling-knife pattern: trend, momentum, and volatility all hostile.")

    return reasons


# ---------------------------------------------------------------------------
# Holding horizon and stop/review price
# ---------------------------------------------------------------------------
def horizon_months(category: str, dip_cat: str, risk: float) -> int:
    base = {"precious": 36, "precious_industrial": 24, "industrial": 12}.get(category, 18)
    if dip_cat in {"strong", "extreme"}:
        base += 6
    if risk > 70:
        base = max(6, base - 6)
    return int(base)


def stop_review_price(price: float, sma200: float, dip_cat: str) -> float:
    """A simple level at which we want to re-evaluate the thesis."""
    if not sma200 or np.isnan(sma200):
        return float(price * 0.92)
    if dip_cat == "falling_knife":
        return float(min(sma200, price) * 0.93)
    if dip_cat in {"strong", "extreme"}:
        return float(min(sma200, price) * 0.95)
    return float(sma200 * 0.97)


# ---------------------------------------------------------------------------
# Top-level orchestrator
# ---------------------------------------------------------------------------
def score_metal(symbol: str,
                pack: dict,
                cfg: dict,
                category: str = "industrial",
                target_pct: float = 0.0,
                current_pct: float = 0.0) -> Optional[MetalScore]:
    if not pack:
        return None

    weights = cfg["scoring"]["weights"]
    thresholds = cfg["scoring"]["dip_thresholds"]

    s_dip = dip_score(pack)
    s_trend = trend_score(pack)
    s_rv = rel_value_score(pack)
    s_voa = vol_adj_opp_score(pack)
    s_pn = portfolio_need_score(symbol, target_pct, current_pct)
    s_mac = macro_score(pack, category)

    final = (
        weights["dip"] * s_dip
        + weights["trend"] * s_trend
        + weights["rel_value"] * s_rv
        + weights["vol_adj_opp"] * s_voa
        + weights["portfolio_need"] * s_pn
        + weights["macro"] * s_mac
    )
    final = float(max(0, min(100, final)))

    dip_cat = classify_dip(pack, thresholds)
    risk = risk_score(pack, dip_cat)
    conf = confidence_score(s_dip, s_trend, s_rv, s_mac)
    horizon = horizon_months(category, dip_cat, risk)
    stop_px = stop_review_price(pack.get("price", 0), pack.get("sma200", 0), dip_cat)
    reasons = build_reasons(pack, dip_cat, s_trend)

    # Hard guard: a falling knife should never produce a "buy" headline,
    # even if some sub-scores look attractive.
    if dip_cat == "falling_knife":
        final = min(final, 35.0)

    return MetalScore(
        symbol=symbol,
        price=pack.get("price", 0.0),
        dip=s_dip, trend=s_trend, rel_value=s_rv,
        vol_adj_opp=s_voa, portfolio_need=s_pn, macro=s_mac,
        final=final,
        risk=risk,
        confidence=conf,
        dip_category=dip_cat,
        buy_class=classify_buy(final),
        horizon_months=horizon,
        stop_review_price=stop_px,
        reasons=reasons,
        pack=pack,
    )
