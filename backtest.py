"""
backtest.py
-----------
Lightweight backtesting harness to compare strategies on historical data.

Strategies implemented:
    - "normal_sip"   : equal-split monthly SIP across selected metals
    - "smart_sip"    : monthly SIP, but deploy more on dips (RSI/drawdown)
    - "gold_only"    : 100% gold SIP
    - "dynamic"      : monthly rebalance using scoring_engine recommendations

Outputs per strategy:
    equity_curve : pd.Series indexed by date
    metrics      : dict with cagr, vol, max_dd, sharpe_like, win_rate, total_invested
    deployments  : list of {date, splits} actually invested

This is intentionally simple — no slippage, no taxes, no transaction costs.
The purpose is comparison, not P&L precision.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from indicators import compute_indicator_pack
from scoring_engine import score_metal


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------
@dataclass
class BacktestResult:
    name: str
    equity_curve: pd.Series
    metrics: dict
    deployments: list[dict] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Portfolio simulator
# ---------------------------------------------------------------------------
class _SimPortfolio:
    """Tracks units of each metal and cash."""

    def __init__(self, starting_cash: float = 0.0):
        self.units: dict[str, float] = {}
        self.cash = starting_cash
        self.invested_total = 0.0       # sum of money put in over time

    def buy(self, sym: str, amount_inr: float, price: float):
        if price <= 0 or amount_inr <= 0:
            return
        units = amount_inr / price
        self.units[sym] = self.units.get(sym, 0) + units
        self.invested_total += amount_inr

    def value(self, prices: dict[str, float]) -> float:
        total = self.cash
        for sym, u in self.units.items():
            px = prices.get(sym, 0)
            total += u * px
        return total


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _month_end_dates(index: pd.DatetimeIndex) -> list[pd.Timestamp]:
    """Last trading day of each month present in the index."""
    df = pd.DataFrame(index=index)
    df["m"] = df.index.to_period("M")
    return list(df.groupby("m").apply(lambda x: x.index.max()))


def _prices_on(date: pd.Timestamp, histories: dict[str, pd.DataFrame]) -> dict[str, float]:
    out = {}
    for sym, df in histories.items():
        if df is None or df.empty:
            continue
        try:
            row = df.loc[df.index <= date].tail(1)
            if not row.empty:
                out[sym] = float(row["close"].iloc[-1])
        except Exception:
            pass
    return out


def _history_up_to(df: pd.DataFrame, date: pd.Timestamp) -> pd.DataFrame:
    return df.loc[df.index <= date].copy()


# ---------------------------------------------------------------------------
# Strategy: equal-split monthly SIP
# ---------------------------------------------------------------------------
def _run_sip(histories: dict[str, pd.DataFrame],
             monthly_amount: float,
             symbols: list[str]) -> BacktestResult:
    if not symbols:
        return BacktestResult("normal_sip", pd.Series(dtype=float), {})

    common_idx = _common_index(histories, symbols)
    if common_idx is None or common_idx.empty:
        return BacktestResult("normal_sip", pd.Series(dtype=float), {})

    portfolio = _SimPortfolio()
    deployments = []
    equity = []

    months = _month_end_dates(common_idx)
    per_metal = monthly_amount / len(symbols)

    for d in common_idx:
        if d in months:
            prices = _prices_on(d, histories)
            split = {}
            for sym in symbols:
                px = prices.get(sym)
                if px:
                    portfolio.buy(sym, per_metal, px)
                    split[sym] = per_metal
            deployments.append({"date": d, "splits": split, "total": sum(split.values())})

        equity.append((d, portfolio.value(_prices_on(d, histories))))

    eq = pd.Series([v for _, v in equity], index=[d for d, _ in equity], name="normal_sip")
    metrics = _compute_metrics(eq, portfolio.invested_total)
    return BacktestResult("normal_sip", eq, metrics, deployments)


# ---------------------------------------------------------------------------
# Strategy: smart dip SIP
#   Each month, base allocation = monthly_amount / N. Then redistribute weight
#   toward metals showing larger drawdown / lower RSI. Total monthly outlay
#   is bounded between 0.5x and 2x the base when accumulator mode is used.
# ---------------------------------------------------------------------------
def _run_smart_sip(histories: dict[str, pd.DataFrame],
                   monthly_amount: float,
                   symbols: list[str],
                   accumulator: bool = False) -> BacktestResult:
    if not symbols:
        return BacktestResult("smart_sip", pd.Series(dtype=float), {})

    common_idx = _common_index(histories, symbols)
    if common_idx is None or common_idx.empty:
        return BacktestResult("smart_sip", pd.Series(dtype=float), {})

    portfolio = _SimPortfolio()
    deployments = []
    equity = []
    months = _month_end_dates(common_idx)

    for d in common_idx:
        if d in months:
            packs = {}
            for sym in symbols:
                hist = _history_up_to(histories[sym], d)
                if len(hist) >= 60:
                    packs[sym] = compute_indicator_pack(hist)
            if not packs:
                equity.append((d, portfolio.value(_prices_on(d, histories))))
                continue

            # Weights: dip-favouring score
            def _w(p):
                rsi_v = p.get("rsi14", 50)
                dd = p.get("drawdown_pct", 0)         # negative
                w = max(0.1, (50 - rsi_v) / 50 + abs(dd) / 30)
                # Falling-knife guard
                if (p.get("dist_from_200dma_pct", 0) < -10
                        and p.get("macd_hist", 0) < 0):
                    w *= 0.3
                return w

            ws = {sym: _w(p) for sym, p in packs.items()}
            wsum = sum(ws.values()) or 1.0

            # Total monthly amount
            month_total = monthly_amount
            if accumulator:
                # Strong dips => deploy more, calm markets => deploy less
                avg_rsi = np.mean([p.get("rsi14", 50) for p in packs.values()])
                avg_dd = np.mean([abs(p.get("drawdown_pct", 0)) for p in packs.values()])
                multiplier = 0.5 + (50 - avg_rsi) / 50 + avg_dd / 25
                multiplier = float(np.clip(multiplier, 0.5, 2.0))
                month_total = monthly_amount * multiplier

            prices = _prices_on(d, histories)
            split = {}
            for sym, w in ws.items():
                amt = month_total * (w / wsum)
                px = prices.get(sym)
                if px and amt > 0:
                    portfolio.buy(sym, amt, px)
                    split[sym] = amt
            deployments.append({"date": d, "splits": split, "total": sum(split.values())})

        equity.append((d, portfolio.value(_prices_on(d, histories))))

    name = "dip_accumulator" if accumulator else "smart_sip"
    eq = pd.Series([v for _, v in equity], index=[d for d, _ in equity], name=name)
    return BacktestResult(name, eq, _compute_metrics(eq, portfolio.invested_total), deployments)


# ---------------------------------------------------------------------------
# Strategy: gold only
# ---------------------------------------------------------------------------
def _run_gold_only(histories: dict[str, pd.DataFrame], monthly_amount: float) -> BacktestResult:
    return _run_sip(histories, monthly_amount, ["GOLD"]).__class__(
        name="gold_only",
        equity_curve=_run_sip(histories, monthly_amount, ["GOLD"]).equity_curve.rename("gold_only"),
        metrics=_run_sip(histories, monthly_amount, ["GOLD"]).metrics,
    )


# ---------------------------------------------------------------------------
# Strategy: dynamic allocation using scoring_engine
# ---------------------------------------------------------------------------
def _run_dynamic(histories: dict[str, pd.DataFrame],
                 monthly_amount: float,
                 symbols: list[str],
                 cfg: dict,
                 metals_cfg: dict) -> BacktestResult:
    if not symbols:
        return BacktestResult("dynamic", pd.Series(dtype=float), {})

    common_idx = _common_index(histories, symbols)
    if common_idx is None or common_idx.empty:
        return BacktestResult("dynamic", pd.Series(dtype=float), {})

    portfolio = _SimPortfolio()
    deployments = []
    equity = []
    months = _month_end_dates(common_idx)

    for d in common_idx:
        if d in months:
            scores = []
            for sym in symbols:
                hist = _history_up_to(histories[sym], d)
                if len(hist) < 60:
                    continue
                pack = compute_indicator_pack(hist)
                cat = metals_cfg.get(sym, {}).get("category", "industrial")
                ms = score_metal(sym, pack, cfg, category=cat)
                if ms and ms.dip_category != "falling_knife":
                    scores.append(ms)
            scores = [s for s in scores if s.final >= 46]
            scores.sort(key=lambda x: x.final, reverse=True)

            split = {}
            if scores:
                total_w = sum(s.final for s in scores) or 1.0
                prices = _prices_on(d, histories)
                for s in scores:
                    amt = monthly_amount * (s.final / total_w)
                    px = prices.get(s.symbol)
                    if px and amt > 0:
                        portfolio.buy(s.symbol, amt, px)
                        split[s.symbol] = amt
                deployments.append({"date": d, "splits": split, "total": sum(split.values())})
            else:
                deployments.append({"date": d, "splits": {}, "total": 0})

        equity.append((d, portfolio.value(_prices_on(d, histories))))

    eq = pd.Series([v for _, v in equity], index=[d for d, _ in equity], name="dynamic")
    return BacktestResult("dynamic", eq, _compute_metrics(eq, portfolio.invested_total), deployments)


# ---------------------------------------------------------------------------
# Common index / metrics
# ---------------------------------------------------------------------------
def _common_index(histories: dict[str, pd.DataFrame],
                  symbols: list[str]) -> Optional[pd.DatetimeIndex]:
    idx = None
    for sym in symbols:
        df = histories.get(sym)
        if df is None or df.empty:
            continue
        idx = df.index if idx is None else idx.intersection(df.index)
    return idx


def _compute_metrics(equity: pd.Series, invested_total: float) -> dict:
    if equity is None or equity.empty:
        return {}

    final_value = float(equity.iloc[-1])
    days = max(1, (equity.index[-1] - equity.index[0]).days)
    years = days / 365.25

    # CAGR vs invested capital is misleading because of staggered SIP. We use
    # money-weighted "TVPI" plus simple time-weighted return on the equity curve.
    tvpi = final_value / invested_total if invested_total > 0 else 0
    pnl_pct = (tvpi - 1) * 100

    rets = equity.pct_change().dropna()
    vol = float(rets.std() * np.sqrt(252)) if len(rets) > 1 else 0.0

    # Max drawdown of equity curve
    peaks = equity.cummax()
    dd = ((equity / peaks) - 1).min()
    max_dd = float(dd) * 100 if not np.isnan(dd) else 0.0

    sharpe_like = 0.0
    if vol > 0 and len(rets) > 1:
        # Risk-free ~ 6% nominal in INR; rough.
        excess = rets.mean() * 252 - 0.06
        sharpe_like = float(excess / vol)

    win_rate = float((rets > 0).mean() * 100) if len(rets) else 0.0

    cagr = (final_value / invested_total) ** (1 / years) - 1 if (invested_total > 0 and years > 0.5) else 0.0

    return {
        "final_value": final_value,
        "invested_total": invested_total,
        "tvpi": tvpi,
        "pnl_pct": pnl_pct,
        "cagr_approx": cagr * 100,
        "vol_annual_pct": vol * 100,
        "max_drawdown_pct": max_dd,
        "sharpe_like": sharpe_like,
        "win_rate_pct": win_rate,
        "years": years,
    }


# ---------------------------------------------------------------------------
# Public runner
# ---------------------------------------------------------------------------
def run_all_strategies(histories: dict[str, pd.DataFrame],
                       monthly_amount: float,
                       symbols: list[str],
                       cfg: dict,
                       metals_cfg: dict) -> dict[str, BacktestResult]:
    """Returns a dict of strategy_name -> BacktestResult."""
    results = {
        "normal_sip": _run_sip(histories, monthly_amount, symbols),
        "smart_sip": _run_smart_sip(histories, monthly_amount, symbols, accumulator=False),
        "dip_accumulator": _run_smart_sip(histories, monthly_amount, symbols, accumulator=True),
        "gold_only": _run_sip(histories, monthly_amount, ["GOLD"]) if "GOLD" in histories else None,
        "dynamic": _run_dynamic(histories, monthly_amount, symbols, cfg, metals_cfg),
    }
    # Tag the gold-only result name properly
    if results.get("gold_only"):
        results["gold_only"] = BacktestResult(
            name="gold_only",
            equity_curve=results["gold_only"].equity_curve.rename("gold_only"),
            metrics=results["gold_only"].metrics,
            deployments=results["gold_only"].deployments,
        )
    return {k: v for k, v in results.items() if v is not None}
