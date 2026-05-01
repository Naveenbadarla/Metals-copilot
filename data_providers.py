"""
data_providers.py
-----------------
Pluggable data adapters for the Metals Co-Pilot dashboard.

All providers implement the same interface:
    .get_history(symbol, lookback_days=730) -> pd.DataFrame
        index: DatetimeIndex (daily)
        columns: ["open", "high", "low", "close", "volume"] (close is the only required one)

Three providers ship out of the box:
    - DemoProvider       : synthetic but realistic price series, no internet needed
    - YahooProvider      : yfinance-backed, free, global futures
    - CSVProvider        : reads <symbol>.csv files from a folder

Adapters for paid feeds (Twelve Data, broker APIs, MCX licensed feeds) can be
added by subclassing BaseProvider.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------
class BaseProvider(ABC):
    """All providers return INR-denominated daily OHLCV for a metal symbol."""

    name: str = "base"

    @abstractmethod
    def get_history(self, symbol: str, lookback_days: int = 730) -> pd.DataFrame:
        ...

    def get_current_price(self, symbol: str) -> Optional[float]:
        df = self.get_history(symbol, lookback_days=10)
        if df is None or df.empty:
            return None
        return float(df["close"].iloc[-1])


# ---------------------------------------------------------------------------
# Demo provider
# ---------------------------------------------------------------------------
@dataclass
class _MetalGenSpec:
    """Parameters used to synthesize a realistic-looking price series."""
    start_price: float
    drift: float        # annualized
    vol: float          # annualized
    cycle_amp: float    # cyclical amplitude as fraction of price
    cycle_days: int     # cycle period
    shock_prob: float   # daily probability of a price shock
    shock_size: float   # std-dev of shock as fraction of price


# Calibrated to roughly match approximate Indian retail prices in 2024-2026
# (orders of magnitude only; for demo only).
_DEMO_SPECS: dict[str, _MetalGenSpec] = {
    "GOLD":      _MetalGenSpec(72000,   0.10, 0.14, 0.06, 220, 0.01, 0.012),
    "SILVER":    _MetalGenSpec(85000,   0.12, 0.28, 0.10, 180, 0.015, 0.020),
    "COPPER":    _MetalGenSpec(820,     0.06, 0.22, 0.08, 200, 0.012, 0.018),
    "ALUMINIUM": _MetalGenSpec(230,     0.04, 0.18, 0.06, 240, 0.010, 0.014),
    "ZINC":      _MetalGenSpec(255,     0.05, 0.24, 0.08, 200, 0.012, 0.020),
    "LEAD":      _MetalGenSpec(190,     0.03, 0.20, 0.06, 220, 0.010, 0.016),
    "NICKEL":    _MetalGenSpec(1450,    0.05, 0.32, 0.10, 180, 0.018, 0.030),
    "PLATINUM":  _MetalGenSpec(35000,   0.05, 0.20, 0.07, 220, 0.012, 0.018),
    "STEELREBAR":_MetalGenSpec(53000,   0.04, 0.16, 0.05, 240, 0.010, 0.012),
}


class DemoProvider(BaseProvider):
    """
    Generates deterministic-but-realistic synthetic histories per metal.
    Seeded so the dashboard is reproducible across reloads.
    """

    name = "demo"

    def __init__(self, seed: int = 42):
        self.seed = seed

    def get_history(self, symbol: str, lookback_days: int = 730) -> pd.DataFrame:
        spec = _DEMO_SPECS.get(symbol.upper())
        if spec is None:
            return pd.DataFrame()

        # Per-symbol seed so each metal is reproducible and independent
        rng = np.random.default_rng(self.seed + abs(hash(symbol)) % 10_000)

        n = lookback_days + 5
        dt = 1.0 / 252.0  # trading-day fraction of year

        # Geometric Brownian Motion with cyclical drift and rare shocks
        returns = rng.normal(loc=spec.drift * dt, scale=spec.vol * np.sqrt(dt), size=n)

        # Add a slow sine cycle so trend reversals appear naturally
        cycle = (spec.cycle_amp / spec.cycle_days) * np.sin(
            2 * np.pi * np.arange(n) / spec.cycle_days
        )
        returns += cycle

        # Occasional shocks (geopolitics, earnings, etc.)
        shocks = rng.binomial(1, spec.shock_prob, size=n) * rng.normal(0, spec.shock_size, size=n)
        returns += shocks

        prices = spec.start_price * np.exp(np.cumsum(returns))

        # Build OHLC around close using small intra-day noise
        close = prices
        noise = rng.normal(0, 0.004, size=n)
        open_ = close * (1 + noise)
        high = np.maximum(close, open_) * (1 + np.abs(rng.normal(0, 0.003, size=n)))
        low = np.minimum(close, open_) * (1 - np.abs(rng.normal(0, 0.003, size=n)))
        volume = rng.integers(8_000, 60_000, size=n)

        # Build calendar of trading days ending today
        end = pd.Timestamp.today().normalize()
        idx = pd.bdate_range(end=end, periods=n)

        df = pd.DataFrame(
            {
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
            },
            index=idx,
        )
        df.index.name = "date"
        return df.tail(lookback_days)


# ---------------------------------------------------------------------------
# Yahoo Finance provider (live, free, global)
# ---------------------------------------------------------------------------
class YahooProvider(BaseProvider):
    """
    Pulls global futures from Yahoo Finance and converts USD prices to INR
    per the metal's `conv_factor` and `inr_unit` configured in config.yaml.

    Falls back to the DemoProvider for any metal whose `yahoo` symbol is null
    or whose download failed (so the dashboard never breaks on a single
    missing series).
    """

    name = "yahoo"

    def __init__(self, metals_cfg: dict, fallback: Optional[BaseProvider] = None):
        self.metals_cfg = metals_cfg
        self.fallback = fallback or DemoProvider()
        try:
            import yfinance as yf  # noqa: F401
            self._yf_ok = True
        except Exception:
            self._yf_ok = False
        self._usdinr_cache: Optional[pd.Series] = None

    # --- helpers ------------------------------------------------------------
    def _fetch_yahoo(self, symbol: str, lookback_days: int) -> Optional[pd.DataFrame]:
        if not self._yf_ok:
            return None
        try:
            import yfinance as yf
            period_days = max(lookback_days + 30, 400)
            df = yf.download(
                symbol,
                period=f"{period_days}d",
                interval="1d",
                auto_adjust=False,
                progress=False,
                threads=False,
            )
            if df is None or df.empty:
                return None
            # yfinance >= 0.2.40 returns MultiIndex columns like ('Close', 'GC=F')
            # even for single tickers. Flatten to level-0 names so the rest of
            # the pipeline (which expects flat 'close', 'open', ... columns) works.
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df = df.rename(
                columns={"Open": "open", "High": "high", "Low": "low",
                         "Close": "close", "Volume": "volume",
                         "Adj Close": "adj_close"}
            )
            # Defensive: if duplicate column labels survived, keep the first.
            df = df.loc[:, ~df.columns.duplicated()]
            keep = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
            if not keep or "close" not in keep:
                return None
            df = df[keep].dropna()
            df.index = pd.to_datetime(df.index).tz_localize(None)
            return df
        except Exception:
            return None

    def _get_usdinr(self, lookback_days: int) -> pd.Series:
        if self._usdinr_cache is not None:
            return self._usdinr_cache
        df = self._fetch_yahoo("INR=X", lookback_days)
        if df is None or df.empty or "close" not in df.columns:
            # Reasonable fallback so industrial-metal numbers are still in the right zip code
            idx = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=lookback_days)
            self._usdinr_cache = pd.Series(83.0, index=idx, name="usdinr")
            return self._usdinr_cache
        # Defensive: ensure we end up with a Series, not a 1-col DataFrame
        close = df["close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        self._usdinr_cache = pd.Series(close.values, index=close.index, name="usdinr")
        return self._usdinr_cache

    # --- interface ----------------------------------------------------------
    def get_history(self, symbol: str, lookback_days: int = 730) -> pd.DataFrame:
        cfg = self.metals_cfg.get(symbol.upper())
        if cfg is None:
            return pd.DataFrame()

        ysym = cfg.get("yahoo")
        if not ysym:
            return self.fallback.get_history(symbol, lookback_days)

        df = self._fetch_yahoo(ysym, lookback_days)
        if df is None or df.empty:
            return self.fallback.get_history(symbol, lookback_days)

        # Convert USD-denominated futures to approximate INR retail price units
        usdinr = self._get_usdinr(lookback_days)
        usdinr_aligned = usdinr.reindex(df.index).ffill().bfill()
        conv = float(cfg.get("conv_factor", 1.0))

        for col in ["open", "high", "low", "close"]:
            df[col] = df[col] * usdinr_aligned * conv

        return df.tail(lookback_days)


# ---------------------------------------------------------------------------
# CSV provider (manual / paid-feed export)
# ---------------------------------------------------------------------------
class CSVProvider(BaseProvider):
    """
    Reads <SYMBOL>.csv from a folder. Each file should have columns
    date, open, high, low, close[, volume]. Prices must already be in the
    desired currency (INR).
    """

    name = "csv"

    def __init__(self, folder: str, fallback: Optional[BaseProvider] = None):
        self.folder = folder
        self.fallback = fallback or DemoProvider()

    def get_history(self, symbol: str, lookback_days: int = 730) -> pd.DataFrame:
        path = os.path.join(self.folder, f"{symbol.upper()}.csv")
        if not os.path.exists(path):
            return self.fallback.get_history(symbol, lookback_days)
        try:
            df = pd.read_csv(path)
            df.columns = [c.lower() for c in df.columns]
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date").sort_index()
            for col in ["open", "high", "low", "close", "volume"]:
                if col not in df.columns:
                    df[col] = np.nan
            df["close"] = df["close"].ffill()
            return df.tail(lookback_days)
        except Exception:
            return self.fallback.get_history(symbol, lookback_days)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
def make_provider(mode: str, metals_cfg: dict, csv_folder: Optional[str] = None) -> BaseProvider:
    mode = (mode or "demo").lower()
    if mode == "live" or mode == "yahoo":
        return YahooProvider(metals_cfg=metals_cfg, fallback=DemoProvider())
    if mode == "csv":
        return CSVProvider(folder=csv_folder or "./data", fallback=DemoProvider())
    return DemoProvider()
