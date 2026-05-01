"""
Groww Trade API integration for Metals Co-Pilot.

PHASE 1 (this file): Read-only.
  - Fetches your real holdings -> auto-fills the Holdings sidebar
  - Fetches live LTP (Last Traded Price) for ETF tickers -> "lively" prices
  - Pulls order history -> real P&L tracking

PHASE 2 (next): place_order() with confirmation
PHASE 3 (last): scheduled auto-trade with hard safety limits

API docs: https://groww.in/trade-api/docs
SDK: pip install growwapi
Auth: generate API token at https://groww.in/trade-api -> Generate API Token

This file is designed so the rest of the dashboard works WITHOUT it.
If groww credentials are missing or growwapi isn't installed, it degrades to None.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List, Dict
import os
import time
import logging

log = logging.getLogger("groww_provider")

# ---------------------------------------------------------------------------
# Symbol mapping: dashboard metals -> Groww-tradeable instruments
# ---------------------------------------------------------------------------
# These are the Indian instruments the dashboard's "investment mode" recommends.
# All are NSE-listed equity/ETF symbols Groww's API can trade.
METAL_TO_NSE_SYMBOLS: Dict[str, List[str]] = {
    "GOLD": [
        "GOLDBEES",   # Nippon India ETF Gold BeES (most liquid gold ETF)
        "GOLDIETF",   # ICICI Prudential Gold ETF
    ],
    "SILVER": [
        "SILVERBEES", # Nippon India Silver ETF
        "SILVERIETF", # ICICI Prudential Silver ETF
    ],
    "COPPER": [
        "HINDCOPPER", # Hindustan Copper — pure-play
        "VEDL",       # Vedanta — diversified copper exposure
    ],
    "ALUMINIUM": [
        "HINDALCO",   # Hindalco Industries — primary aluminium producer
        "NATIONALUM", # National Aluminium (NALCO)
    ],
    "ZINC": [
        "HINDZINC",   # Hindustan Zinc — pure-play zinc
    ],
    "LEAD": [
        "HINDZINC",   # HZL also produces lead
    ],
    "NICKEL": [
        # No clean pure-play nickel listing in India.
        # User would need to use international ETFs via different routes.
    ],
    "PLATINUM": [
        # No platinum ETF in India yet — physical / international only.
    ],
}


@dataclass
class GrowwHolding:
    """One holding row pulled from Groww."""
    symbol: str          # NSE symbol e.g. "GOLDBEES"
    quantity: float
    avg_price: float
    ltp: float           # last traded price
    invested: float      # quantity * avg_price
    current_value: float # quantity * ltp
    pnl: float
    pnl_pct: float


@dataclass
class GrowwSnapshot:
    """Everything the dashboard needs from Groww in one shot."""
    holdings: List[GrowwHolding]
    total_invested: float
    total_value: float
    total_pnl: float
    timestamp: float
    source: str = "groww_live"


class GrowwProvider:
    """
    Read-only wrapper around the Groww trading API.

    Initialize with an API token (NOT your account password).
    Generate the token at https://groww.in/trade-api -> "Generate API Token".
    Store it in Streamlit secrets, NOT in your code.
    """

    def __init__(self, api_token: Optional[str] = None):
        self.api_token = api_token or os.getenv("GROWW_API_TOKEN")
        self._sdk = None
        self._client = None
        self._cache: Optional[GrowwSnapshot] = None
        self._cache_ttl_sec = 30  # refresh holdings at most every 30s

        if not self.api_token:
            log.warning("No Groww API token found. Provider will be inactive.")
            return

        try:
            from growwapi import GrowwAPI  # type: ignore
            self._sdk = GrowwAPI
            self._client = GrowwAPI(self.api_token)
            log.info("Groww client initialized.")
        except ImportError:
            log.warning(
                "growwapi package not installed. Run: pip install growwapi"
            )
        except Exception as e:
            log.error(f"Failed to initialize Groww client: {e}")

    @property
    def is_active(self) -> bool:
        return self._client is not None

    # -----------------------------------------------------------------------
    # Holdings
    # -----------------------------------------------------------------------
    def fetch_snapshot(self, force: bool = False) -> Optional[GrowwSnapshot]:
        """
        Fetch holdings + current values.
        Returns None if the client isn't active.
        Caches for 30s to avoid hammering the API.
        """
        if not self.is_active:
            return None

        now = time.time()
        if (
            not force
            and self._cache is not None
            and (now - self._cache.timestamp) < self._cache_ttl_sec
        ):
            return self._cache

        try:
            response = self._client.get_holdings_for_user(timeout=10)
        except Exception as e:
            log.error(f"Groww holdings fetch failed: {e}")
            return self._cache  # return stale cache rather than nothing

        # The API returns a dict; the structure as of late 2025:
        #   {"status": "SUCCESS", "payload": {"holdings": [ {...}, ... ]}}
        # We're defensive because the SDK may evolve.
        holdings_raw = []
        try:
            if isinstance(response, dict):
                payload = response.get("payload") or response.get("data") or response
                holdings_raw = payload.get("holdings", []) or []
            elif isinstance(response, list):
                holdings_raw = response
        except Exception:
            holdings_raw = []

        holdings: List[GrowwHolding] = []
        total_inv = 0.0
        total_val = 0.0

        for h in holdings_raw:
            try:
                # Field names vary — try a few likely keys defensively.
                symbol = (
                    h.get("trading_symbol")
                    or h.get("tradingsymbol")
                    or h.get("symbol", "")
                ).upper()
                qty = float(h.get("quantity", 0) or h.get("qty", 0))
                avg = float(
                    h.get("average_price", 0)
                    or h.get("avg_price", 0)
                    or h.get("buy_price", 0)
                )
                ltp = float(
                    h.get("ltp", 0)
                    or h.get("last_price", 0)
                    or h.get("close_price", avg)
                )
                if qty <= 0 or symbol == "":
                    continue

                inv = qty * avg
                val = qty * ltp
                pnl = val - inv
                pnl_pct = (pnl / inv * 100.0) if inv > 0 else 0.0

                holdings.append(GrowwHolding(
                    symbol=symbol,
                    quantity=qty,
                    avg_price=avg,
                    ltp=ltp,
                    invested=inv,
                    current_value=val,
                    pnl=pnl,
                    pnl_pct=pnl_pct,
                ))
                total_inv += inv
                total_val += val
            except Exception as e:
                log.debug(f"Skipping unparseable holding row: {e}")
                continue

        snap = GrowwSnapshot(
            holdings=holdings,
            total_invested=total_inv,
            total_value=total_val,
            total_pnl=total_val - total_inv,
            timestamp=now,
        )
        self._cache = snap
        return snap

    # -----------------------------------------------------------------------
    # Filter holdings to only metals-relevant tickers
    # -----------------------------------------------------------------------
    def metals_holdings_by_metal(self) -> Dict[str, List[GrowwHolding]]:
        """
        Group user's metals-relevant holdings by which metal they map to.
        Returns {metal_symbol: [GrowwHolding, ...]} — only metals with positions.
        """
        snap = self.fetch_snapshot()
        if snap is None:
            return {}

        # Reverse the metal -> symbols map
        symbol_to_metal: Dict[str, str] = {}
        for metal, syms in METAL_TO_NSE_SYMBOLS.items():
            for s in syms:
                # First mapping wins (so HINDZINC counts as ZINC, not LEAD)
                symbol_to_metal.setdefault(s, metal)

        out: Dict[str, List[GrowwHolding]] = {}
        for h in snap.holdings:
            metal = symbol_to_metal.get(h.symbol)
            if metal is None:
                continue
            out.setdefault(metal, []).append(h)
        return out

    def metals_holdings_aggregated(self) -> Dict[str, Dict[str, float]]:
        """
        Returns {metal: {"invested": ₹, "current_value": ₹, "avg_price": weighted-avg}}
        Used to auto-fill the dashboard's Holdings sidebar.
        """
        grouped = self.metals_holdings_by_metal()
        agg: Dict[str, Dict[str, float]] = {}
        for metal, rows in grouped.items():
            total_inv = sum(r.invested for r in rows)
            total_val = sum(r.current_value for r in rows)
            total_qty = sum(r.quantity for r in rows)
            avg_price = (total_inv / total_qty) if total_qty > 0 else 0.0
            agg[metal] = {
                "invested_inr": total_inv,
                "current_value_inr": total_val,
                "avg_price": avg_price,
                "quantity": total_qty,
                "pnl_inr": total_val - total_inv,
                "pnl_pct": ((total_val - total_inv) / total_inv * 100.0) if total_inv > 0 else 0.0,
            }
        return agg

    # -----------------------------------------------------------------------
    # Live LTP for any tickers (used to power "lively" prices)
    # -----------------------------------------------------------------------
    def get_ltp(self, nse_symbols: List[str]) -> Dict[str, float]:
        """
        Fetch live last-traded-price for a list of NSE symbols.
        Returns {symbol: price}; symbols that fail are silently skipped.
        """
        if not self.is_active:
            return {}
        out: Dict[str, float] = {}
        for sym in nse_symbols:
            try:
                # Try a few likely SDK method names for LTP
                if hasattr(self._client, "get_ltp"):
                    resp = self._client.get_ltp(
                        exchange="NSE", trading_symbol=sym, segment="CASH"
                    )
                elif hasattr(self._client, "get_quote"):
                    resp = self._client.get_quote(
                        exchange="NSE", trading_symbol=sym, segment="CASH"
                    )
                else:
                    continue
                # Defensive parse
                if isinstance(resp, dict):
                    price = (
                        resp.get("ltp")
                        or resp.get("last_price")
                        or (resp.get("payload") or {}).get("ltp")
                        or (resp.get("data") or {}).get("ltp")
                    )
                    if price is not None:
                        out[sym] = float(price)
            except Exception as e:
                log.debug(f"LTP fetch failed for {sym}: {e}")
                continue
        return out

    def get_metal_live_prices(self) -> Dict[str, float]:
        """
        For each dashboard metal, fetch the LTP of its primary ETF/equity proxy.
        Returns {metal_symbol: price_inr}.
        """
        if not self.is_active:
            return {}
        out: Dict[str, float] = {}
        for metal, syms in METAL_TO_NSE_SYMBOLS.items():
            if not syms:
                continue
            primary = syms[0]
            ltps = self.get_ltp([primary])
            if primary in ltps:
                out[metal] = ltps[primary]
        return out


# ---------------------------------------------------------------------------
# Streamlit-friendly factory
# ---------------------------------------------------------------------------
def make_groww_provider() -> Optional[GrowwProvider]:
    """
    Build a GrowwProvider from Streamlit secrets first, env vars second.
    Returns None if no credentials are configured.

    In Streamlit, store the token like this in .streamlit/secrets.toml:
        GROWW_API_TOKEN = "your_token_here"

    Or on Streamlit Cloud: Settings -> Secrets -> add the key.
    """
    token: Optional[str] = None
    try:
        import streamlit as st  # type: ignore
        token = st.secrets.get("GROWW_API_TOKEN")  # type: ignore
    except Exception:
        pass
    if not token:
        token = os.getenv("GROWW_API_TOKEN")
    if not token:
        return None
    provider = GrowwProvider(token)
    return provider if provider.is_active else None
