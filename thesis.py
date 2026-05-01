"""
thesis.py
---------
Metal-specific investment narratives and India-specific instrument routes.
This is plain content — no calculations — so it lives separately from logic.
"""

THESIS = {
    "gold": {
        "headline": "Safe-haven core, INR-depreciation hedge",
        "drivers": [
            "Hedges global risk-off events, geopolitical stress, and equity drawdowns.",
            "Long-term hedge against INR depreciation vs USD.",
            "Inflation hedge over multi-year horizons.",
            "Sensitive to central-bank gold buying (notably RBI, PBOC).",
            "Inversely correlated with real US interest rates.",
        ],
        "best_when": "Real yields falling, USD weakening, equity volatility rising.",
        "watch_for": "Aggressive Fed tightening or sharp USD strength can cap upside.",
    },
    "silver": {
        "headline": "Precious + industrial hybrid, higher beta than gold",
        "drivers": [
            "Solar PV demand has become a major secular driver.",
            "Electronics, EV, and 5G manufacturing demand.",
            "Tracks gold in safe-haven episodes but with 1.5–2x volatility.",
            "Tends to outperform gold in late stages of precious-metal bull cycles.",
        ],
        "best_when": "Gold is rising and global growth expectations are stable to improving.",
        "watch_for": "Silver corrects faster than gold in risk-off events.",
    },
    "copper": {
        "headline": "The electrification metal",
        "drivers": [
            "Grid expansion, EVs, and renewables drive structural demand.",
            "Highly cyclical — tracks global manufacturing PMIs and China activity.",
            "Supply growth constrained by ore-grade decline at major mines.",
        ],
        "best_when": "Global PMIs expanding, China stimulus active, USD softening.",
        "watch_for": "Recession risk and China property weakness drive sharp drawdowns.",
    },
    "aluminium": {
        "driver": "industrial",
        "headline": "Lightweighting, packaging, infra",
        "drivers": [
            "Auto lightweighting and EV bodies.",
            "Packaging and construction.",
            "Renewable infrastructure.",
            "Production cost is heavily energy-driven (smelters)."
        ],
        "best_when": "Energy prices firming and global manufacturing expanding.",
        "watch_for": "Cheap energy + Chinese smelter expansion can cap prices.",
    },
    "zinc": {
        "headline": "Galvanizing demand, infra-linked",
        "drivers": [
            "Mostly used to galvanize steel — tied to construction/infra cycles.",
            "More volatile than gold/silver; more tactical exposure.",
        ],
        "best_when": "Infra and construction cycles accelerating.",
        "watch_for": "Property/construction slowdowns hit demand directly.",
    },
    "nickel": {
        "headline": "Battery + stainless steel exposure",
        "drivers": [
            "EV battery cathodes (NMC chemistry).",
            "Stainless steel — the larger demand bucket today.",
            "High volatility; supply shocks (Indonesia policy) drive sharp moves.",
        ],
        "best_when": "EV demand accelerating and Indonesian supply stable.",
        "watch_for": "Indonesian export rule changes and battery-chemistry shifts.",
    },
    "lead": {
        "headline": "Battery (lead-acid) demand",
        "drivers": [
            "Lead-acid batteries for ICE vehicles, backup power, telecom.",
            "Mature, lower-growth metal; minor portfolio role only.",
        ],
        "best_when": "Stable demand environment.",
        "watch_for": "Long-term EV transition pressures lead-acid demand.",
    },
    "platinum": {
        "headline": "Auto-catalyst + emerging hydrogen role",
        "drivers": [
            "Diesel auto-catalysts (declining), gasoline catalysts (substituting palladium).",
            "Potential hydrogen-economy demand long-term.",
        ],
        "best_when": "Palladium-platinum substitution accelerating, hydrogen capex rising.",
        "watch_for": "Auto demand structurally falling and ETF flows reversing.",
    },
    "steel": {
        "headline": "India infrastructure proxy",
        "drivers": [
            "Infrastructure spend, real estate, capex cycle.",
            "Highly cyclical and policy-sensitive.",
        ],
        "best_when": "Government capex accelerating and credit conditions easy.",
        "watch_for": "Cheap Chinese exports can pressure margins and prices.",
    },
}


# ---------------------------------------------------------------------------
# India instrument routes per metal — practical, retail-friendly
# ---------------------------------------------------------------------------
INSTRUMENTS = {
    "GOLD": [
        ("Sovereign Gold Bonds (SGB)", "Best for long-term holders. Look for secondary-market SGBs at discount to NAV."),
        ("Gold ETFs", "Liquid, tax-efficient post-2023; e.g. Nippon, HDFC, ICICI Pru, SBI Gold ETF."),
        ("Gold Mutual Funds", "Good for SIP automation; slight expense over ETFs."),
        ("Digital Gold (MMTC-PAMP, Augmont)", "Convenient for small tactical buys; check platform & spread."),
        ("Physical (coins/bars)", "Emotional/long-term holding only — high spread + storage cost."),
    ],
    "SILVER": [
        ("Silver ETFs", "Cleanest exposure: Nippon, ICICI Pru, HDFC, Aditya Birla SL Silver ETF."),
        ("Silver Mutual Funds / FoF", "For SIP automation."),
        ("Digital Silver", "Small allocations only — assess platform risk."),
        ("Physical silver", "Avoid for investment-size — storage and spread are punitive."),
    ],
    "COPPER": [
        ("Metal sector stocks (e.g. Hindalco, Vedanta)", "Equity proxy with company-specific risk."),
        ("MCX Copper futures", "Sophisticated investors only — leveraged."),
        ("Global commodity ETFs (US-listed via LRS)", "If accessible and within LRS limits."),
    ],
    "ALUMINIUM": [
        ("Hindalco / Nalco / Vedanta equity", "India-listed proxies."),
        ("MCX Aluminium futures", "Leveraged — investors should default to equity proxies."),
    ],
    "ZINC": [
        ("Hindustan Zinc equity", "Direct India-listed proxy."),
        ("MCX Zinc futures", "Leveraged."),
    ],
    "NICKEL": [
        ("Global mining equity ETFs", "Limited India-listed pure plays."),
        ("MCX Nickel futures", "Highly volatile and leveraged — caution."),
    ],
    "LEAD": [
        ("Tactical only — limited dedicated instruments in India."),
    ],
    "PLATINUM": [
        ("Global PGM ETFs (US-listed via LRS)", "Limited India retail options."),
        ("Platinum coins", "High spread, niche use case."),
    ],
    "STEELREBAR": [
        ("Tata Steel, JSW Steel, Jindal Steel & Power equity", "India-listed proxies — cyclical."),
        ("MCX Steel Rebar futures", "Leveraged."),
    ],
}
