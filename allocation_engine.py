"""
allocation_engine.py
--------------------
Converts (scores + budget + holdings + risk profile + strategy) into a
concrete "deploy ₹X today, split across these metals, here's why" answer.

Public entry point: `recommend_today(...)`

The engine respects four hard safety rails regardless of strategy:
    1. Never deploy 100% of available cash on a single day.
    2. Always keep a cash buffer (stricter if user is using loan money).
    3. Never recommend more than the per-metal cap.
    4. Falling-knife metals are excluded from buy lists.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from scoring_engine import MetalScore


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------
@dataclass
class Allocation:
    symbol: str
    amount: float
    weight: float           # share of today's deployment (0-1)
    reason: str


@dataclass
class Recommendation:
    action: str             # "buy" | "wait" | "hold_cash" | "rebalance"
    deploy_amount: float
    cash_remaining: float
    allocations: list[Allocation] = field(default_factory=list)
    headline: str = ""
    reasoning: list[str] = field(default_factory=list)
    risk_warnings: list[str] = field(default_factory=list)
    confidence: float = 0.0
    next_review_days: int = 7


# ---------------------------------------------------------------------------
# User inputs
# ---------------------------------------------------------------------------
@dataclass
class UserInputs:
    available_cash_today: float
    monthly_budget: float
    holdings_inr: dict          # {symbol: current INR value invested}
    avg_buy_price: dict         # {symbol: avg buy price}
    risk_profile: str           # conservative | balanced | aggressive
    horizon_years: int
    strategy: str               # smart_sip | dip_accumulator | momentum_dip | gs_ratio | industrial_cycle
    max_per_metal_pct: float    # 0-1, cap on a single metal's share of holdings
    using_loan_money: bool
    enable_futures: bool
    min_buy_inr: float = 500
    refresh_days: int = 7


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _current_allocation_pct(holdings: dict) -> dict:
    total = sum(holdings.values()) or 1.0
    return {k: (v / total) * 100 for k, v in holdings.items()}


def _target_bands(risk_profile: str, cfg: dict) -> dict:
    bands = cfg.get("allocation_bands", {}).get(risk_profile, {})
    return {k: float(v) for k, v in bands.items() if k != "cash"}


def _cap_per_metal(amount: float, total_after: float, max_pct: float, current_holding: float) -> float:
    """Clamp a proposed buy so the metal doesn't exceed `max_pct` of total holdings."""
    if total_after <= 0:
        return amount
    new_holding = current_holding + amount
    cap_value = max_pct * total_after
    if new_holding <= cap_value:
        return amount
    return max(0.0, cap_value - current_holding)


# ---------------------------------------------------------------------------
# Strategy filters & weightings
# Each strategy is a function: (eligible_scores, cfg, gs_ratio_pct, ratio_pctile)
#                              -> list of (score, strategy_weight_multiplier)
# ---------------------------------------------------------------------------
def _strategy_smart_sip(scores: list[MetalScore], **kwargs) -> list[tuple[MetalScore, float]]:
    return [(s, 1.0) for s in scores]


def _strategy_dip_accumulator(scores: list[MetalScore], **kwargs) -> list[tuple[MetalScore, float]]:
    # Only metals with strong/extreme dip OR final >= 65
    eligible = [s for s in scores
                if s.dip_category in {"good", "strong", "extreme"} or s.final >= 65]
    return [(s, 1.0) for s in eligible]


def _strategy_momentum_dip(scores: list[MetalScore], **kwargs) -> list[tuple[MetalScore, float]]:
    # Buy dips only when long-term trend is healthy (trend score >= 55)
    eligible = [s for s in scores if s.trend >= 55 and s.dip >= 50]
    return [(s, 1.0) for s in eligible]


def _strategy_gs_ratio(scores: list[MetalScore], **kwargs) -> list[tuple[MetalScore, float]]:
    cfg = kwargs.get("cfg", {})
    ratio = kwargs.get("gs_ratio")
    silver_thr = cfg.get("gs_ratio", {}).get("silver_attractive_above", 85)
    gold_thr = cfg.get("gs_ratio", {}).get("gold_attractive_below", 60)

    out = []
    for s in scores:
        mult = 1.0
        if s.symbol == "SILVER" and ratio and ratio >= silver_thr:
            mult = 1.5
        if s.symbol == "GOLD" and ratio and ratio <= gold_thr:
            mult = 1.5
        out.append((s, mult))
    return out


def _strategy_industrial_cycle(scores: list[MetalScore], **kwargs) -> list[tuple[MetalScore, float]]:
    industrial = {"COPPER", "ALUMINIUM", "ZINC", "NICKEL", "LEAD"}
    ranked = sorted(
        [s for s in scores if s.symbol in industrial],
        key=lambda x: (x.trend + x.macro) / 2,
        reverse=True,
    )
    weighted = []
    for i, s in enumerate(ranked):
        mult = 1.5 if i == 0 else (1.2 if i == 1 else 1.0)
        weighted.append((s, mult))
    # Always keep gold/silver as baseline
    for s in scores:
        if s.symbol in {"GOLD", "SILVER"}:
            weighted.append((s, 0.8))
    return weighted


_STRATEGY_FNS = {
    "smart_sip": _strategy_smart_sip,
    "dip_accumulator": _strategy_dip_accumulator,
    "momentum_dip": _strategy_momentum_dip,
    "gs_ratio": _strategy_gs_ratio,
    "industrial_cycle": _strategy_industrial_cycle,
}


# ---------------------------------------------------------------------------
# How much to deploy today
# ---------------------------------------------------------------------------
def _today_deploy_envelope(user: UserInputs, cfg: dict, top_score: float) -> float:
    """
    Decide how much of available cash we are willing to put to work today.
    Higher buy score = bigger fraction. Loan money = stricter cap.
    """
    if user.using_loan_money:
        max_frac = cfg["app"]["loan_money_max_single_day_deploy_pct"]
        buffer_frac = cfg["app"]["loan_money_min_cash_buffer_pct"]
    else:
        max_frac = cfg["app"]["max_single_day_deploy_pct"]
        buffer_frac = cfg["app"]["min_cash_buffer_pct"]

    # Map top_score -> envelope as fraction of available cash
    # 50 score -> 10%, 70 -> max_frac/2, 90+ -> max_frac
    if top_score < 46:
        frac = 0.0
    elif top_score < 60:
        frac = 0.05 + (top_score - 46) / 14 * 0.10
    elif top_score < 76:
        frac = 0.15 + (top_score - 60) / 16 * (max_frac / 2 - 0.15)
    else:
        frac = max_frac / 2 + (top_score - 76) / 14 * (max_frac - max_frac / 2)

    frac = min(frac, max_frac)

    # Always preserve buffer
    deployable = max(0.0, user.available_cash_today * (1 - buffer_frac))
    return float(min(deployable, user.available_cash_today * frac))


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def recommend_today(scores: list[MetalScore],
                    user: UserInputs,
                    cfg: dict,
                    gs_ratio: Optional[float] = None) -> Recommendation:
    if not scores:
        return Recommendation(
            action="wait",
            deploy_amount=0,
            cash_remaining=user.available_cash_today,
            headline="No data available — cannot recommend an action today.",
            reasoning=["Provider returned no usable data."],
        )

    # 1. Drop falling knives, sort by final score
    safe_scores = [s for s in scores if s.dip_category != "falling_knife"]
    safe_scores.sort(key=lambda x: x.final, reverse=True)

    if not safe_scores:
        return Recommendation(
            action="hold_cash",
            deploy_amount=0,
            cash_remaining=user.available_cash_today,
            headline="All metals showing falling-knife patterns. Hold cash today.",
            reasoning=["Every metal under coverage is below 200DMA with negative momentum and rising volatility."],
            risk_warnings=["Wait for stabilization before deploying capital."],
            next_review_days=3,
        )

    top = safe_scores[0]

    # 2. Apply strategy filter
    fn = _STRATEGY_FNS.get(user.strategy, _strategy_smart_sip)
    eligible = fn(safe_scores, cfg=cfg, gs_ratio=gs_ratio)

    # Within eligible, only keep metals scoring above watch threshold
    eligible = [(s, m) for s, m in eligible if s.final >= 46]

    if not eligible:
        return Recommendation(
            action="wait",
            deploy_amount=0,
            cash_remaining=user.available_cash_today,
            headline=f"No metal currently meets the {user.strategy.replace('_', ' ')} criteria.",
            reasoning=[f"Top score is {top.symbol} at {top.final:.0f}/100 — below the action threshold."],
            risk_warnings=["Consider waiting for clearer dip signals before deploying."],
            confidence=top.confidence,
            next_review_days=user.refresh_days,
        )

    # 3. Envelope of how much to deploy today
    envelope = _today_deploy_envelope(user, cfg, top.final)
    if envelope < user.min_buy_inr:
        return Recommendation(
            action="wait",
            deploy_amount=0,
            cash_remaining=user.available_cash_today,
            headline=f"Signal too soft for meaningful deployment today.",
            reasoning=[f"Computed deploy envelope ₹{envelope:,.0f} is below your min buy ₹{user.min_buy_inr:,.0f}."],
            confidence=top.confidence,
            next_review_days=user.refresh_days,
        )

    # 4. Compute weights: combine final score, strategy multiplier, and portfolio need
    holdings = dict(user.holdings_inr or {})
    total_holdings = sum(holdings.values())
    targets = _target_bands(user.risk_profile, cfg)

    raw_weights = []
    for s, mult in eligible:
        target_pct = targets.get(s.symbol, 0.0)
        current_pct = (holdings.get(s.symbol, 0) / total_holdings * 100) if total_holdings > 0 else 0.0
        underweight_bonus = max(0.0, target_pct - current_pct) / 20.0  # +1 per 20pp gap
        w = (s.final / 100.0) * mult * (1 + underweight_bonus)
        raw_weights.append((s, max(0.0, w)))

    weight_sum = sum(w for _, w in raw_weights) or 1.0
    raw_weights = [(s, w / weight_sum) for s, w in raw_weights]

    # 5. Allocate, applying per-metal cap
    allocations: list[Allocation] = []
    new_total = total_holdings + envelope
    deployed = 0.0

    for s, w in sorted(raw_weights, key=lambda x: -x[1]):
        amt = round(envelope * w, -1)        # round to nearest ₹10
        current_holding = holdings.get(s.symbol, 0)
        amt = _cap_per_metal(amt, new_total, user.max_per_metal_pct, current_holding)
        if amt < user.min_buy_inr:
            continue
        allocations.append(Allocation(
            symbol=s.symbol,
            amount=float(amt),
            weight=float(w),
            reason=_short_reason(s),
        ))
        deployed += amt

    # 6. Defensive: if nothing got allocated (e.g. all clipped by caps), force the top
    if not allocations and top.final >= 46:
        amt = max(user.min_buy_inr, min(envelope, user.available_cash_today * 0.10))
        allocations.append(Allocation(
            symbol=top.symbol,
            amount=float(amt),
            weight=1.0,
            reason=_short_reason(top),
        ))
        deployed = amt

    # 7. Build headline + warnings
    parts = [f"₹{a.amount:,.0f} {a.symbol.title()}" for a in allocations]
    headline = (
        f"Deploy ₹{deployed:,.0f} today: " + ", ".join(parts) + "."
        if allocations else
        "No deployment recommended today."
    )

    risk_warnings = _build_warnings(user, top, allocations, cfg)
    reasoning = [
        f"Top opportunity: {top.symbol.title()} at {top.final:.0f}/100 ({top.buy_class.replace('_',' ')}).",
        *top.reasons[:3],
        f"Strategy in use: {user.strategy.replace('_', ' ').title()}.",
    ]

    return Recommendation(
        action="buy" if allocations else "wait",
        deploy_amount=float(deployed),
        cash_remaining=float(user.available_cash_today - deployed),
        allocations=allocations,
        headline=headline,
        reasoning=reasoning,
        risk_warnings=risk_warnings,
        confidence=top.confidence,
        next_review_days=user.refresh_days,
    )


def _short_reason(s: MetalScore) -> str:
    if s.dip_category in {"strong", "extreme"}:
        return f"{s.dip_category.replace('_',' ')} dip, RSI {s.pack.get('rsi14', 50):.0f}"
    if s.trend >= 70 and s.dip >= 50:
        return f"Healthy uptrend with pullback (trend {s.trend:.0f})"
    if s.rel_value >= 70:
        return f"Trades cheap vs 1Y mean (z {s.pack.get('zscore_252', 0):+.2f})"
    if s.portfolio_need >= 65:
        return "Underweight vs target allocation"
    return f"Composite score {s.final:.0f}/100"


def _build_warnings(user: UserInputs,
                    top: MetalScore,
                    allocations: list[Allocation],
                    cfg: dict) -> list[str]:
    warnings = []
    if user.using_loan_money:
        warnings.append("Loan money mode: tighter cash buffer and per-day cap applied. Avoid leverage.")
    if any(a.symbol in {"COPPER", "ALUMINIUM", "ZINC", "NICKEL", "LEAD"} for a in allocations):
        warnings.append("Industrial metals are cyclical — keep position sizing modest and review monthly.")
    if user.enable_futures:
        warnings.append("Futures mode enabled. MCX commodity futures are leveraged. Default to ETFs/funds for investing.")
    if top.risk > 70:
        warnings.append(f"{top.symbol.title()} risk score {top.risk:.0f}/100 — size appropriately.")
    return warnings


# ---------------------------------------------------------------------------
# Sell / rebalance signals (used by rebalance UI tab)
# ---------------------------------------------------------------------------
@dataclass
class RebalanceSignal:
    symbol: str
    action: str           # take_profit | trim | hold | accumulate
    drift_pp: float       # current% - target%
    gain_pct: float
    reason: str


def rebalance_signals(scores: list[MetalScore],
                      user: UserInputs,
                      cfg: dict) -> list[RebalanceSignal]:
    targets = _target_bands(user.risk_profile, cfg)
    holdings = dict(user.holdings_inr or {})
    total = sum(holdings.values()) or 0.0
    if total <= 0:
        return []

    rb_cfg = cfg.get("rebalance", {})
    out: list[RebalanceSignal] = []
    score_by_sym = {s.symbol: s for s in scores}

    for sym, target in targets.items():
        cur_val = holdings.get(sym, 0)
        cur_pct = (cur_val / total) * 100 if total > 0 else 0
        drift = cur_pct - target

        avg = user.avg_buy_price.get(sym, 0)
        s = score_by_sym.get(sym)
        gain_pct = 0.0
        if s and avg > 0:
            gain_pct = (s.price / avg - 1) * 100

        action = "hold"
        reason = ""

        # Take profit
        if (gain_pct >= rb_cfg.get("take_profit_gain_pct", 20)
                and s and s.pack.get("rsi14", 50) > rb_cfg.get("take_profit_rsi", 70)):
            if sym == "GOLD":
                action = "hold"
                reason = "Gold up strongly but core holding — do not trim aggressively."
            else:
                action = "take_profit"
                reason = f"Up {gain_pct:.1f}% with RSI {s.pack.get('rsi14',50):.0f} — book partial profit."

        # Trim if drifted too far above target
        elif drift >= rb_cfg.get("trim_drift_pct_points", 10):
            action = "trim"
            reason = f"Allocation {cur_pct:.1f}% vs target {target:.0f}% — trim {drift:.1f}pp."

        # Accumulate if drifted below target
        elif drift <= -rb_cfg.get("trim_drift_pct_points", 10):
            action = "accumulate"
            reason = f"Underweight by {abs(drift):.1f}pp — direct fresh capital here."

        out.append(RebalanceSignal(
            symbol=sym,
            action=action,
            drift_pp=drift,
            gain_pct=gain_pct,
            reason=reason or f"Within tolerance of {target:.0f}% target.",
        ))

    return out
