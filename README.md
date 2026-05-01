# Metals Co-Pilot — India Metals Investment Dashboard

A production-grade Streamlit dashboard for **intelligent dip-based investing** across gold, silver, copper, aluminium, zinc, lead, nickel, and platinum — built with India (MCX / INR) as the primary market.

> **This is a decision-support tool, not financial advice.** It surfaces signals, confidence, risk, and reasoning. You make the call.

---

## What it does

Every day it answers seven questions:

1. Which metal is most attractive to buy right now?
2. Is today a dip worth deploying capital into?
3. How much should I deploy today?
4. Should I buy gold, silver, copper, aluminium — or stay in cash?
5. Should I sell or rebalance anything?
6. What is the risk of the signal?
7. What is the expected holding horizon?

It does this through a 6-factor scoring model, a daily allocation engine, a rebalance engine, five strategy modes, and a backtest module — all wrapped in a premium dark UI.

---

## Quick start

```bash
# 1. Clone / unzip into a folder, then:
cd metals_dashboard

# 2. (recommended) create a virtualenv
python -m venv .venv
source .venv/bin/activate          # macOS / Linux
# .venv\Scripts\activate           # Windows

# 3. install deps
pip install -r requirements.txt

# 4. run
streamlit run app.py
```

Open the URL Streamlit prints (usually `http://localhost:8501`). The app launches in **Demo mode** — it generates realistic synthetic price histories so you can use the full dashboard without any API keys.

---

## Folder structure

```
metals_dashboard/
├── app.py                 # Streamlit UI — 7 tabs, dark theme, all charts
├── config.yaml            # All tunables: metal universe, weights, bands, colors
├── requirements.txt       # Minimal deps (no TA-Lib needed)
├── data_providers.py      # DemoProvider, YahooProvider, CSVProvider + factory
├── indicators.py          # SMA, EMA, RSI, MACD, BB, ATR, ADX, drawdown (pure pandas)
├── scoring_engine.py      # 6-factor scoring + classification + reasons
├── allocation_engine.py   # Daily deployment + 5 strategy modes + safety rails
├── portfolio.py           # Holdings, P&L, allocation drift, alerts
├── backtest.py            # SIP vs smart-dip vs dynamic, CAGR/Sharpe/drawdown
├── thesis.py              # Per-metal investment thesis + India instrument routes
└── README.md              # This file
```

---

## Three data modes

Configurable from the **sidebar → Data Mode**. The same UI / engines / charts work in all three.

### 1. Demo mode (default)
Synthetic GBM-based price histories with cyclical drift and rare shocks per metal, calibrated to recent INR price levels. Perfect for evaluating the dashboard.

### 2. Live mode (Yahoo Finance — free)
Pulls global futures (`GC=F`, `SI=F`, `HG=F`, etc.), converts USD → INR using `INR=X`, and applies retail-unit conversion factors from `config.yaml`. No API key required, but coverage is limited (gold/silver/copper are reliable; aluminium/zinc/lead/nickel may fall back to demo).

### 3. CSV mode
Drop OHLC CSVs into a folder (one file per metal: `GOLD.csv`, `SILVER.csv`, ...) with columns `date, open, high, low, close, volume`. Point the sidebar to the folder. This is how you plug in **MCX paid feeds, broker exports, Investing.com downloads, or Twelve Data dumps**.

### Plugging in a real-time API
To wire a real provider (MCX paid feed, broker WebSocket, Twelve Data, Alpha Vantage, etc.):

1. Open `data_providers.py`.
2. Subclass `BaseProvider`:
   ```python
   class MyMCXProvider(BaseProvider):
       def get_history(self, symbol: str, lookback_days: int = 800) -> pd.DataFrame:
           # call your API, return a DataFrame with index=date and columns
           # ['open','high','low','close','volume']
           ...
       def get_spot(self, symbol: str) -> float:
           ...
   ```
3. Register it in the `make_provider()` factory at the bottom of the file.
4. Add a new option in the sidebar `Data Mode` selectbox in `app.py`.

The rest of the app — indicators, scoring, allocation, backtest, UI — works unchanged.

---

## Scoring logic (the heart of the app)

Every metal gets a **Final Buy Score from 0–100**, computed as a weighted blend of six sub-scores:

| Sub-score | Weight | What it captures |
|---|---|---|
| **Dip Score** | 30% | Drawdown from recent peak, distance below 20/50/200 DMA, RSI zone, Bollinger position |
| **Trend Quality** | 20% | Long-term trend health: price vs 200DMA, MACD direction, ADX strength |
| **Relative Valuation** | 15% | Z-score vs 1-year mean, price percentile in 1y range, gold-silver ratio context |
| **Volatility-Adjusted Opportunity** | 15% | Dip magnitude divided by realized vol — separates "real opportunity" from "normal noise" |
| **Portfolio Need** | 10% | How underweight this metal is vs your target allocation |
| **Macro Proxy** | 10% | Coarse macro tilt (precious vs industrial) — placeholder for real macro feeds |

**Final classification bands:**

| Score | Label |
|---|---|
| 0–30 | Avoid |
| 31–45 | Watch |
| 46–60 | Small buy |
| 61–75 | Buy |
| 76–90 | Strong buy |
| 91–100 | Extreme opportunity (verify risk) |

**Falling-knife guard:** if price is below the 200DMA *and* MACD is negative *and* volatility is rising, the final score is **hard-capped at 35** regardless of how cheap it looks. This is enforced in both `scoring_engine.py` *and* `allocation_engine.py` (defense in depth).

For each metal the engine also returns: **risk score, confidence score, suggested holding horizon (months), suggested stop/review price, and a plain-English reasons list.** All of this is visible in the *Metal Detail* tab.

---

## Daily allocation logic

The allocation engine takes your sidebar inputs (cash, budget, risk profile, holdings, strategy, loan-money toggle) and produces a single recommendation. It runs in this order:

1. **Drop falling knives** from the buy universe.
2. **Apply strategy filter** (SIP / Dip Accumulator / Momentum+Dip / Gold-Silver Ratio / Industrial Cycle).
3. **Compute today's deployment envelope** — a fraction of available cash based on the *best* score in the universe. Loan-money mode tightens this from 40% max to 20% max.
4. **Weight remaining metals** by `final_score × strategy_multiplier × portfolio_underweight_bonus`.
5. **Clip to per-metal caps** from the sidebar.
6. **Build the headline + reasoning + warnings** — including "what would change this recommendation".

**Hardcoded safety rails** (cannot be overridden):
- Never deploy 100% of available cash in one day.
- Always preserve a minimum cash buffer (10% normal, 25% loan-money mode).
- Never exceed your per-metal allocation cap.
- Falling knives are always excluded from buy lists.

---

## Five strategy modes

| Mode | Logic |
|---|---|
| **Smart SIP** | Default. Invest monthly, but deploy *more* during dips and *less* when overheated. |
| **Dip Accumulator** | Hold cash; deploy only when dip score crosses a threshold. |
| **Momentum + Dip** | Buy dips, but only when long-term trend (200DMA + MACD) is still positive. |
| **Gold-Silver Ratio** | Tilts incremental capital toward whichever is cheap on the GSR percentile. |
| **Industrial Cycle** | Ranks copper, aluminium, zinc, nickel by momentum × valuation × inverse-vol. |

---

## Rebalance engine

Triggers per holding:
- **Take profit:** unrealized gain ≥ 20% AND RSI > 70 → suggest trimming 25–40%.
- **Trim:** allocation drift > 10 percentage points above target → trim down.
- **Accumulate:** drift > 10pp below target AND score ≥ 60 → top up on the next dip.
- **Hold:** otherwise.
- **Special rule:** core gold is never trimmed unless you flag a liquidity need.

---

## Backtest module

The *Backtest* tab compares four strategies side-by-side over your chosen window:

1. **Plain SIP** — equal split, monthly.
2. **Smart SIP** — weights monthly capital by dip metric per metal.
3. **Gold-only SIP** — control benchmark.
4. **Dynamic** — runs the full scoring engine each month-end and deploys per its recommendation.

Output: equity curves, CAGR, volatility, max drawdown, Sharpe-like ratio (rf = 6%), win rate, TVPI, and a per-month deployment log.

> **Backtest caveat:** demo mode runs against synthetic data, so backtest results are illustrative of *behavior*, not historical performance. Switch to Live or CSV mode and re-run for real-data backtests.

---

## Alerts

In-app alerts are shown in the *Alerts* tab and surface in the headline card:
- Buy / Strong dip / Extreme opportunity
- Overheated (RSI > 75 + above upper Bollinger)
- Rebalance needed (drift > 10pp)
- Gold-Silver ratio extreme
- Portfolio concentration (single metal > 50%)

The `AlertSink` interface in `portfolio.py` is a stub for plugging in **Telegram, email, or WhatsApp delivery later** — implement `send(alert)` and register your sink.

---

## India-specific instrument routes

The *Thesis & Instruments* tab lays out practical investment routes per metal — **SGBs, Gold ETFs, Gold MFs, digital gold, Silver ETFs, sector equity proxies, MCX futures (with leverage warnings), commodity MFs**. Defaults to "investment mode" — futures are off unless you explicitly enable them in the sidebar.

A placeholder for **Indian capital gains tax notes** is included. *No tax/legal advice is provided.*

---

## Configuration

Almost everything is tunable via `config.yaml` without touching code:
- Metal universe (enable/disable platinum, steel/rebar)
- Yahoo Finance symbols + USD→INR retail unit conversion factors
- Scoring weights (currently 30/20/15/15/10/10)
- Classification bands
- Dip thresholds (mild=3%, good=6%, strong=10%, extreme=15%)
- Allocation bands per risk profile
- Rebalance rules
- Gold-silver ratio thresholds
- App defaults

Edit and restart Streamlit.

---

## Future roadmap

- **Real-time alerts:** Telegram bot, email (SMTP), WhatsApp Business API
- **MCX paid feed integration** with bid/ask + volume from official source
- **Broker API integration** (Zerodha Kite, Upstox, Groww) for one-tap execution
- **Macro data feeds** — DXY, US 10Y real yield, INR/USD, central bank gold buying — to replace the placeholder macro score with real signals
- **Mobile-first responsive layout** + PWA wrapper
- **Multi-currency support** (USD, AED for NRI users)
- **Tax computation module** for India CGT (LTCG/STCG on physical, ETFs, SGBs)
- **Custom alert rules engine** — "ping me when silver drops 5% in a week and RSI < 30"
- **Portfolio import** from CAS / Coin / broker export
- **AI thesis assistant** — natural-language Q&A over the current state of the dashboard

---

## Disclaimer

This dashboard is an **educational and decision-support tool**. It does not constitute investment, financial, legal, or tax advice. Commodities are volatile. Past performance — synthetic or real — does not guarantee future results. Verify all signals independently and consult a SEBI-registered advisor before deploying capital. The authors accept no liability for investment decisions made using this tool.
