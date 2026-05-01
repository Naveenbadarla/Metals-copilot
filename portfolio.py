"""
portfolio.py
------------
Portfolio state container + alerts.

Responsibilities:
    - Hold the user's current holdings and average buy prices
    - Compute current allocation %, drift vs target, P/L
    - Emit alerts (buy / strong dip / overheated / rebalance / GS-ratio / concentration)

Alerts here are pure data; the UI decides where to show them. The architecture
is ready for Telegram/email/WhatsApp dispatch by adding a new alerts sink.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from scoring_engine import MetalScore


@dataclass
class Holding:
    symbol: str
    invested_inr: float    # total INR cost basis
    avg_buy_price: float   # in metal's price unit


@dataclass
class Alert:
    severity: str          # info | warn | critical
    kind: str              # buy | strong_dip | overheated | rebalance | gs_ratio | concentration
    symbol: Optional[str]
    title: str
    body: str


# ---------------------------------------------------------------------------
# Portfolio
# ---------------------------------------------------------------------------
class Portfolio:
    def __init__(self, holdings: dict[str, Holding] | None = None):
        self.holdings: dict[str, Holding] = holdings or {}

    def total_value(self, current_prices: dict[str, float]) -> float:
        """We track invested INR rather than units, so 'value' is the user's
        cost basis. For estimating MTM, multiply by (current/avg)."""
        total = 0.0
        for sym, h in self.holdings.items():
            px = current_prices.get(sym)
            if px and h.avg_buy_price > 0:
                total += h.invested_inr * (px / h.avg_buy_price)
            else:
                total += h.invested_inr
        return total

    def invested_total(self) -> float:
        return sum(h.invested_inr for h in self.holdings.values())

    def current_allocation_pct(self, current_prices: dict[str, float]) -> dict[str, float]:
        total = self.total_value(current_prices) or 1.0
        out = {}
        for sym, h in self.holdings.items():
            px = current_prices.get(sym)
            mtm = h.invested_inr * (px / h.avg_buy_price) if (px and h.avg_buy_price) else h.invested_inr
            out[sym] = (mtm / total) * 100
        return out

    def pnl_summary(self, current_prices: dict[str, float]) -> dict[str, dict]:
        out = {}
        for sym, h in self.holdings.items():
            px = current_prices.get(sym)
            if not (px and h.avg_buy_price):
                out[sym] = {"invested": h.invested_inr, "mtm": h.invested_inr,
                            "pnl": 0.0, "pnl_pct": 0.0}
                continue
            mtm = h.invested_inr * (px / h.avg_buy_price)
            pnl = mtm - h.invested_inr
            out[sym] = {
                "invested": h.invested_inr,
                "mtm": mtm,
                "pnl": pnl,
                "pnl_pct": (pnl / h.invested_inr) * 100 if h.invested_inr else 0.0,
            }
        return out


# ---------------------------------------------------------------------------
# Alert engine
# ---------------------------------------------------------------------------
def build_alerts(scores: list[MetalScore],
                 portfolio: Portfolio,
                 current_prices: dict[str, float],
                 target_bands: dict[str, float],
                 gs_ratio: Optional[float],
                 cfg: dict) -> list[Alert]:
    alerts: list[Alert] = []

    # 1. Per-metal buy / strong dip / overheated
    for s in scores:
        if s.dip_category in {"strong", "extreme"}:
            alerts.append(Alert(
                severity="critical" if s.dip_category == "extreme" else "warn",
                kind="strong_dip",
                symbol=s.symbol,
                title=f"{s.symbol.title()}: {s.dip_category.replace('_', ' ')} dip",
                body=" • ".join(s.reasons[:3]) or f"Dip score {s.dip:.0f}/100.",
            ))
        if s.buy_class in {"strong_buy", "extreme"}:
            alerts.append(Alert(
                severity="warn",
                kind="buy",
                symbol=s.symbol,
                title=f"{s.symbol.title()}: {s.buy_class.replace('_',' ')} signal ({s.final:.0f}/100)",
                body=" • ".join(s.reasons[:2]) or "Composite score crossed action threshold.",
            ))
        if s.pack.get("rsi14", 50) > 75 and s.pack.get("dist_from_200dma_pct", 0) > 15:
            alerts.append(Alert(
                severity="warn",
                kind="overheated",
                symbol=s.symbol,
                title=f"{s.symbol.title()}: overheated",
                body=f"RSI {s.pack.get('rsi14',50):.0f}, "
                     f"{s.pack.get('dist_from_200dma_pct',0):.1f}% above 200DMA.",
            ))

    # 2. Gold-Silver ratio
    if gs_ratio is not None:
        gs_cfg = cfg.get("gs_ratio", {})
        if gs_ratio >= gs_cfg.get("silver_attractive_above", 85):
            alerts.append(Alert(
                severity="info",
                kind="gs_ratio",
                symbol="SILVER",
                title=f"Gold-Silver ratio elevated ({gs_ratio:.1f})",
                body="Silver historically looks relatively attractive vs gold at these levels.",
            ))
        elif gs_ratio <= gs_cfg.get("gold_attractive_below", 60):
            alerts.append(Alert(
                severity="info",
                kind="gs_ratio",
                symbol="GOLD",
                title=f"Gold-Silver ratio compressed ({gs_ratio:.1f})",
                body="Gold historically looks relatively attractive vs silver at these levels.",
            ))

    # 3. Rebalance / concentration
    cur_pct = portfolio.current_allocation_pct(current_prices)
    rb_cfg = cfg.get("rebalance", {})
    drift_thr = rb_cfg.get("trim_drift_pct_points", 10)
    for sym, target in target_bands.items():
        actual = cur_pct.get(sym, 0)
        drift = actual - target
        if drift >= drift_thr:
            alerts.append(Alert(
                severity="warn",
                kind="rebalance",
                symbol=sym,
                title=f"{sym.title()} allocation drifted +{drift:.1f}pp",
                body=f"Currently {actual:.1f}% vs target {target:.0f}%. Consider trimming.",
            ))
    for sym, actual in cur_pct.items():
        if actual >= 70:
            alerts.append(Alert(
                severity="critical",
                kind="concentration",
                symbol=sym,
                title=f"Heavy concentration in {sym.title()} ({actual:.1f}%)",
                body="Diversify across at least 3 metals to reduce single-asset risk.",
            ))

    return alerts


# ---------------------------------------------------------------------------
# Future hooks: alert dispatchers
# ---------------------------------------------------------------------------
class AlertSink:
    """Base class. Replace with TelegramSink / EmailSink / WhatsAppSink later."""

    def push(self, alert: Alert) -> None:
        raise NotImplementedError


class InMemorySink(AlertSink):
    def __init__(self):
        self.log: list[Alert] = []

    def push(self, alert: Alert) -> None:
        self.log.append(alert)
