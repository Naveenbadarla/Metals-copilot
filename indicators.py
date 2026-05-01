"""
indicators.py
-------------
Pure-pandas/numpy technical indicators. No external TA libs required so the
dashboard installs cleanly on any machine.

All functions take a price Series (close prices) and return a Series aligned
to the input index.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Moving averages
# ---------------------------------------------------------------------------
def sma(close: pd.Series, n: int) -> pd.Series:
    return close.rolling(n, min_periods=max(2, n // 2)).mean()


def ema(close: pd.Series, n: int) -> pd.Series:
    return close.ewm(span=n, adjust=False, min_periods=max(2, n // 2)).mean()


# ---------------------------------------------------------------------------
# RSI (14)
# ---------------------------------------------------------------------------
def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    # Wilder's smoothing
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return (100 - (100 / (1 + rs))).fillna(50)


# ---------------------------------------------------------------------------
# MACD (12, 26, 9). Returns macd_line, signal_line, histogram
# ---------------------------------------------------------------------------
def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    fast_ema = ema(close, fast)
    slow_ema = ema(close, slow)
    macd_line = fast_ema - slow_ema
    sig = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    hist = macd_line - sig
    return pd.DataFrame({"macd": macd_line, "signal": sig, "hist": hist})


# ---------------------------------------------------------------------------
# Bollinger bands (20, 2). Returns upper, mid, lower, position (0..1)
# ---------------------------------------------------------------------------
def bollinger(close: pd.Series, n: int = 20, k: float = 2.0) -> pd.DataFrame:
    mid = sma(close, n)
    sd = close.rolling(n, min_periods=max(2, n // 2)).std()
    upper = mid + k * sd
    lower = mid - k * sd
    width = (upper - lower).replace(0, np.nan)
    pos = ((close - lower) / width).clip(0, 1)
    return pd.DataFrame({"bb_upper": upper, "bb_mid": mid, "bb_lower": lower, "bb_pos": pos})


# ---------------------------------------------------------------------------
# Average True Range and ADX
# ---------------------------------------------------------------------------
def atr(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low).abs(),
         (high - prev_close).abs(),
         (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def adx(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    """Average Directional Index. >25 generally indicates a trending market."""
    up = high.diff()
    down = -low.diff()
    plus_dm = ((up > down) & (up > 0)) * up
    minus_dm = ((down > up) & (down > 0)) * down

    a = atr(high, low, close, n).replace(0, np.nan)
    plus_di = 100 * plus_dm.ewm(alpha=1 / n, adjust=False, min_periods=n).mean() / a
    minus_di = 100 * minus_dm.ewm(alpha=1 / n, adjust=False, min_periods=n).mean() / a
    dx = (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan) * 100
    return dx.ewm(alpha=1 / n, adjust=False, min_periods=n).mean().fillna(0)


# ---------------------------------------------------------------------------
# Drawdown from rolling peak (in %)
# ---------------------------------------------------------------------------
def drawdown_from_peak(close: pd.Series, lookback: int = 252) -> pd.Series:
    peak = close.rolling(lookback, min_periods=10).max()
    return ((close / peak) - 1) * 100  # negative = below peak


# ---------------------------------------------------------------------------
# Z-score versus rolling mean
# ---------------------------------------------------------------------------
def zscore(close: pd.Series, lookback: int = 252) -> pd.Series:
    m = close.rolling(lookback, min_periods=20).mean()
    s = close.rolling(lookback, min_periods=20).std().replace(0, np.nan)
    return ((close - m) / s).fillna(0)


# ---------------------------------------------------------------------------
# Annualised volatility (std of log returns * sqrt(252))
# ---------------------------------------------------------------------------
def realized_vol(close: pd.Series, lookback: int = 30) -> pd.Series:
    log_ret = np.log(close / close.shift(1))
    return log_ret.rolling(lookback, min_periods=10).std() * np.sqrt(252)


# ---------------------------------------------------------------------------
# Convenience: full indicator pack for a metal in one call
# ---------------------------------------------------------------------------
def compute_indicator_pack(df: pd.DataFrame) -> dict:
    """
    Given an OHLCV DataFrame, return a dict of latest scalar indicator values
    plus the underlying series for charts.
    """
    if df is None or df.empty or "close" not in df.columns:
        return {}

    close = df["close"].astype(float)
    high = df.get("high", close).astype(float)
    low = df.get("low", close).astype(float)

    sma20 = sma(close, 20)
    sma50 = sma(close, 50)
    sma100 = sma(close, 100)
    sma200 = sma(close, 200)
    rsi14 = rsi(close, 14)
    macd_df = macd(close)
    bb = bollinger(close, 20, 2)
    adx14 = adx(high, low, close, 14)
    dd_252 = drawdown_from_peak(close, 252)
    z_252 = zscore(close, 252)
    vol_30 = realized_vol(close, 30)

    last = -1
    price = float(close.iloc[last])

    def _safe(s, i=last, default=np.nan):
        try:
            v = float(s.iloc[i])
            return v if not np.isnan(v) else default
        except Exception:
            return default

    high_252 = float(close.tail(252).max()) if len(close) > 0 else price
    low_252 = float(close.tail(252).min()) if len(close) > 0 else price

    pack = {
        # Latest scalars
        "price": price,
        "sma20": _safe(sma20),
        "sma50": _safe(sma50),
        "sma100": _safe(sma100),
        "sma200": _safe(sma200),
        "rsi14": _safe(rsi14, default=50),
        "macd": _safe(macd_df["macd"]),
        "macd_signal": _safe(macd_df["signal"]),
        "macd_hist": _safe(macd_df["hist"]),
        "bb_pos": _safe(bb["bb_pos"], default=0.5),
        "adx14": _safe(adx14, default=15),
        "drawdown_pct": _safe(dd_252, default=0),
        "zscore_252": _safe(z_252, default=0),
        "vol_30": _safe(vol_30, default=0.2),
        "high_52w": high_252,
        "low_52w": low_252,
        "dist_from_high_pct": (price / high_252 - 1) * 100 if high_252 else 0,
        "dist_from_200dma_pct": (price / _safe(sma200) - 1) * 100 if _safe(sma200) else 0,

        # % changes
        "ret_1d": _pct_change(close, 1),
        "ret_1w": _pct_change(close, 5),
        "ret_1m": _pct_change(close, 21),
        "ret_3m": _pct_change(close, 63),
        "ret_6m": _pct_change(close, 126),
        "ret_1y": _pct_change(close, 252),

        # Series for plotting
        "_series": {
            "close": close, "sma20": sma20, "sma50": sma50,
            "sma100": sma100, "sma200": sma200,
            "rsi": rsi14, "macd": macd_df, "bb": bb,
            "adx": adx14, "drawdown": dd_252, "vol": vol_30,
        },
    }
    return pack


def _pct_change(s: pd.Series, n: int) -> float:
    if len(s) <= n:
        return 0.0
    try:
        return float((s.iloc[-1] / s.iloc[-1 - n] - 1) * 100)
    except Exception:
        return 0.0
