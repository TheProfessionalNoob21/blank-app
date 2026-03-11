"""
Trading Strategy App — SMA, EMA, RSI & ATR
Run with: streamlit run trading_app.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import yfinance as yf
import streamlit as st
import warnings
import time
warnings.filterwarnings("ignore")
from streamlit_autorefresh import st_autorefresh


# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="Whaleberg",
    page_icon="🐋",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;800&family=JetBrains+Mono:wght@300;400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'JetBrains Mono', monospace;
    background-color: #05080f;
    color: #e2e8f0;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #080c17;
    border-right: 1px solid #0e1e35;
}
section[data-testid="stSidebar"] * {
    color: #e2e8f0 !important;
}

/* Main header */
.main-header {
    font-family: 'Syne', sans-serif;
    font-size: 2.8rem;
    font-weight: 800;
    color: #00c9a7;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 0.1rem;
    text-shadow: 0 0 40px rgba(0,201,167,0.25);
}
.main-sub {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    color: #2a4a5e;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    margin-bottom: 2rem;
}

/* Metric cards */
.metric-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1rem;
    margin: 1.5rem 0;
}
.metric-card {
    background: #080c17;
    border: 1px solid #0e1e35;
    border-radius: 8px;
    padding: 1rem 1.2rem;
}
.metric-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    color: #475569;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 0.4rem;
}
.metric-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.4rem;
    font-weight: 600;
    color: #f1f5f9;
}
.metric-value.positive { color: #4ade80; }
.metric-value.negative { color: #f87171; }
.metric-value.neutral  { color: #00c9a7; }

/* Divider */
.section-divider {
    border: none;
    border-top: 1px solid #0e1e35;
    margin: 1.5rem 0;
}

/* Tag */
.ticker-tag {
    display: inline-block;
    background: #0e1e35;
    border: 1px solid #00c9a7;
    border-radius: 4px;
    padding: 0.2rem 0.6rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    color: #00c9a7;
    margin-right: 0.5rem;
}

/* Bloomberg panel */
.bbg-panel {
    background: #080c17;
    border: 1px solid #0e1e35;
    border-radius: 10px;
    padding: 1.4rem 1.6rem;
    margin: 1rem 0 1.5rem 0;
}
.bbg-name {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.25rem;
    font-weight: 600;
    color: #f1f5f9;
    margin-bottom: 0.15rem;
}
.bbg-meta {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    color: #475569;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 1rem;
}
.bbg-price {
    font-family: 'JetBrains Mono', monospace;
    font-size: 2rem;
    font-weight: 600;
    color: #f1f5f9;
    line-height: 1;
}
.bbg-chg-pos { font-family:'JetBrains Mono',monospace; font-size:0.9rem; color:#4ade80; margin-left:0.6rem; }
.bbg-chg-neg { font-family:'JetBrains Mono',monospace; font-size:0.9rem; color:#f87171; margin-left:0.6rem; }
.bbg-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 0.6rem 2rem;
    margin-top: 1.2rem;
    border-top: 1px solid #0e1e35;
    padding-top: 1rem;
}
.bbg-kv { display: flex; flex-direction: column; }
.bbg-k  { font-family:'JetBrains Mono',monospace; font-size:0.6rem; color:#475569; text-transform:uppercase; letter-spacing:0.08em; }
.bbg-v  { font-family:'JetBrains Mono',monospace; font-size:0.85rem; color:#e2e8f0; margin-top:0.15rem; }
.bbg-desc {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    color: #64748b;
    line-height: 1.6;
    margin-top: 1rem;
    border-top: 1px solid #0e1e35;
    padding-top: 0.9rem;
}

/* Streamlit overrides */
.stSelectbox label, .stTextInput label, .stSlider label, .stDateInput label {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.7rem !important;
    color: #64748b !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
}
div[data-testid="stMetric"] {
    background: #080c17;
    border: 1px solid #0e1e35;
    border-radius: 8px;
    padding: 1rem;
}
div[data-testid="stMetric"] label {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.65rem !important;
    color: #475569 !important;
    text-transform: uppercase !important;
}
div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
    font-family: 'IBM Plex Mono', monospace !important;
    color: #f1f5f9 !important;
}
.stButton > button {
    background: #00c9a7;
    color: #05080f;
    font-family: 'JetBrains Mono', monospace;
    font-weight: 600;
    font-size: 0.8rem;
    letter-spacing: 0.05em;
    border: none;
    border-radius: 6px;
    padding: 0.6rem 1.5rem;
    width: 100%;
    transition: background 0.2s;
}
.stButton > button:hover {
    background: #5eead4;
    color: #05080f;
}
.stDataFrame {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.8rem;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  INDICATOR FUNCTIONS
# ─────────────────────────────────────────────

def add_sma(df, fast, slow):
    df[f"sma_{fast}"] = df["close"].rolling(fast).mean()
    df[f"sma_{slow}"] = df["close"].rolling(slow).mean()
    return df

def add_ema(df, fast, slow):
    df[f"ema_{fast}"] = df["close"].ewm(span=fast, adjust=False).mean()
    df[f"ema_{slow}"] = df["close"].ewm(span=slow, adjust=False).mean()
    return df

def add_rsi(df, period):
    delta = df["close"].diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss.replace(0, np.nan)
    df["rsi"] = 100 - (100 / (1 + rs))
    return df

def add_atr(df, period):
    hl = df["high"] - df["low"]
    hc = (df["high"] - df["close"].shift()).abs()
    lc = (df["low"]  - df["close"].shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    df["atr"] = tr.rolling(period).mean()
    return df

def compute_indicators(df, cfg):
    df = add_sma(df, cfg["sma_fast"], cfg["sma_slow"])
    df = add_ema(df, cfg["ema_fast"], cfg["ema_slow"])
    df = add_rsi(df, cfg["rsi_period"])
    df = add_atr(df, cfg["atr_period"])
    return df


# ─────────────────────────────────────────────
#  SIGNAL + BACKTEST
# ─────────────────────────────────────────────

def generate_signals(df, cfg):
    sf, ss = f"sma_{cfg['sma_fast']}", f"sma_{cfg['sma_slow']}"
    ef, es = f"ema_{cfg['ema_fast']}", f"ema_{cfg['ema_slow']}"
    trend_up   = (df[sf] > df[ss]) & (df[ef] > df[es])
    rsi_ok     = df["rsi"] < cfg["rsi_overbought"]
    rsi_exit   = df["rsi"] > cfg["rsi_overbought"]
    trend_down = (df[sf] < df[ss]) | (df[ef] < df[es])
    df["signal"] = 0
    df.loc[trend_up & rsi_ok,     "signal"] =  1
    df.loc[trend_down | rsi_exit, "signal"] = -1
    df["position"] = df["signal"].replace(-1, 0).replace(0, np.nan)
    df["position"] = df["position"].ffill().fillna(0)
    df.loc[df["signal"] == -1, "position"] = 0
    return df

def compute_position_size(row, portfolio_value, cfg):
    if row["close"] == 0 or np.isnan(row["close"]) or row["atr"] == 0 or np.isnan(row["atr"]):
        return 0.0
    # Volatility targeting: ATR as % of price (normalised volatility)
    # Higher ATR  → smaller position  (more volatile, be cautious)
    # Lower ATR   → larger position   (calmer market, deploy more)
    atr_pct    = row["atr"] / row["close"]                      # e.g. 0.02 = 2% daily range
    target_vol = cfg["risk_per_trade"] * cfg["atr_multiplier"]  # target portfolio risk per trade
    deploy_pct = min(target_vol / atr_pct, 0.95)                # cap at 95% of portfolio
    return (portfolio_value * deploy_pct) / row["close"]

def run_backtest(df, cfg):
    cash, shares_held, equity, prev_pos = cfg["initial_capital"], 0.0, [], 0
    for _, row in df.iterrows():
        cur_pos = row["position"]
        if cur_pos == 1 and prev_pos == 0:
            size = compute_position_size(row, cash, cfg)
            cost = size * row["close"]
            if cost <= cash:
                shares_held = size
                cash -= cost
        elif cur_pos == 0 and prev_pos == 1 and shares_held > 0:
            cash += shares_held * row["close"]
            shares_held = 0.0
        equity.append(cash + shares_held * row["close"])
        prev_pos = cur_pos
    if shares_held > 0:
        equity[-1] = cash + shares_held * df.iloc[-1]["close"]
    df = df.copy()
    df["equity"]  = equity
    df["returns"] = df["equity"].pct_change().fillna(0)
    return df

def run_benchmark(df, cfg):
    initial_shares = cfg["initial_capital"] / df["close"].iloc[0]
    return initial_shares * df["close"]

def compute_metrics(equity, label="Strategy"):
    returns   = equity.pct_change().dropna()
    total_ret = (equity.iloc[-1] / equity.iloc[0]) - 1
    ann_ret   = (1 + total_ret) ** (252 / len(equity)) - 1
    ann_vol   = returns.std() * np.sqrt(252)
    sharpe    = ann_ret / ann_vol if ann_vol != 0 else 0
    rolling_max = equity.cummax()
    drawdown  = (equity - rolling_max) / rolling_max
    max_dd    = drawdown.min()
    calmar    = ann_ret / abs(max_dd) if max_dd != 0 else 0
    win_rate  = (returns > 0).sum() / len(returns)
    return {
        "Label":           label,
        "Total Return":    total_ret,
        "Ann. Return":     ann_ret,
        "Ann. Volatility": ann_vol,
        "Sharpe Ratio":    sharpe,
        "Max Drawdown":    max_dd,
        "Calmar Ratio":    calmar,
        "Win Rate":        win_rate,
    }

def get_stock_info(ticker):
    try:
        t    = yf.Ticker(ticker)
        info = t.info
        hist = t.history(period="5d")
        prev_close = hist["Close"].iloc[-2] if len(hist) >= 2 else None
        curr_price = hist["Close"].iloc[-1]  if len(hist) >= 1 else None
        day_chg    = ((curr_price - prev_close) / prev_close * 100) if (curr_price and prev_close) else None

        # OHLC today
        today_open  = hist["Open"].iloc[-1]  if len(hist) >= 1 else None
        today_high  = hist["High"].iloc[-1]  if len(hist) >= 1 else None
        today_low   = hist["Low"].iloc[-1]   if len(hist) >= 1 else None
        today_vol   = hist["Volume"].iloc[-1] if len(hist) >= 1 else None

        def fmt_large(v):
            if v is None: return "—"
            try: v = float(v)
            except: return "—"
            if v >= 1e12: return f"{v/1e12:.2f}T"
            if v >= 1e9:  return f"{v/1e9:.2f}B"
            if v >= 1e6:  return f"{v/1e6:.2f}M"
            return f"{v:,.0f}"

        def safe(key, fmt=None, suffix=""):
            v = info.get(key)
            if v is None or v == "N/A": return "—"
            try:
                if fmt: return fmt.format(float(v)) + suffix
            except: return "—"
            return str(v) + suffix

        def pct(key):
            v = info.get(key)
            if v is None: return "—"
            try: return f"{float(v)*100:.2f}%"
            except: return "—"

        # Analyst recommendations
        try:
            rec = t.recommendations
            if rec is not None and not rec.empty:
                # latest 90 days summary
                cols = [c for c in rec.columns if c.lower() in ["strongbuy","buy","hold","sell","strongsell"]]
                if cols:
                    latest = rec.iloc[-1]
                    analyst_summary = {c: int(latest.get(c, 0)) for c in cols}
                else:
                    analyst_summary = {}
            else:
                analyst_summary = {}
        except:
            analyst_summary = {}

        # News
        try:
            news_raw = t.news or []
            news = []
            for n in news_raw[:6]:
                content = n.get("content", {})
                title = content.get("title") or n.get("title", "")
                pub = content.get("pubDate") or n.get("providerPublishTime", "")
                provider = content.get("provider", {}).get("displayName", "") if isinstance(content.get("provider"), dict) else ""
                link_obj = content.get("canonicalUrl", {})
                link = link_obj.get("url", "") if isinstance(link_obj, dict) else n.get("link", "")
                if title:
                    news.append({"title": title, "publisher": provider, "link": link, "time": str(pub)[:10]})
        except:
            news = []

        return {
            "name":         info.get("longName", ticker),
            "sector":       info.get("sector", "—"),
            "industry":     info.get("industry", "—"),
            "country":      info.get("country", "—"),
            "currency":     info.get("currency", ""),
            "exchange":     info.get("exchange", "—"),
            "curr_price":   f"{curr_price:.2f}" if curr_price else "—",
            "day_chg":      day_chg,
            "prev_close":   f"{prev_close:.2f}" if prev_close else "—",
            "open":         f"{today_open:.2f}" if today_open else "—",
            "high":         f"{today_high:.2f}" if today_high else "—",
            "low":          f"{today_low:.2f}"  if today_low  else "—",
            "volume":       fmt_large(today_vol),
            "week52_high":  safe("fiftyTwoWeekHigh", "{:.2f}"),
            "week52_low":   safe("fiftyTwoWeekLow",  "{:.2f}"),
            "mkt_cap":      fmt_large(info.get("marketCap")),
            "enterprise_val": fmt_large(info.get("enterpriseValue")),
            "pe_ratio":     safe("trailingPE", "{:.2f}"),
            "fwd_pe":       safe("forwardPE",  "{:.2f}"),
            "peg":          safe("pegRatio",   "{:.2f}"),
            "ps_ratio":     safe("priceToSalesTrailing12Months", "{:.2f}"),
            "pb_ratio":     safe("priceToBook", "{:.2f}"),
            "ev_ebitda":    safe("enterpriseToEbitda", "{:.2f}"),
            "eps":          safe("trailingEps", "{:.2f}"),
            "fwd_eps":      safe("forwardEps",  "{:.2f}"),
            "beta":         safe("beta", "{:.2f}"),
            "div_yield":    pct("dividendYield"),
            "div_rate":     safe("dividendRate", "{:.2f}"),
            "payout_ratio": pct("payoutRatio"),
            "avg_volume":   fmt_large(info.get("averageVolume")),
            "float_shares": fmt_large(info.get("floatShares")),
            "shares_out":   fmt_large(info.get("sharesOutstanding")),
            "short_pct":    pct("shortPercentOfFloat"),
            "inst_own":     pct("heldPercentInstitutions"),
            "insider_own":  pct("heldPercentInsiders"),
            # Financials
            "revenue":      fmt_large(info.get("totalRevenue")),
            "gross_margin": pct("grossMargins"),
            "op_margin":    pct("operatingMargins"),
            "net_margin":   pct("profitMargins"),
            "ebitda":       fmt_large(info.get("ebitda")),
            "free_cashflow":fmt_large(info.get("freeCashflow")),
            "total_cash":   fmt_large(info.get("totalCash")),
            "total_debt":   fmt_large(info.get("totalDebt")),
            "debt_equity":  safe("debtToEquity", "{:.2f}"),
            "current_ratio":safe("currentRatio", "{:.2f}"),
            "roe":          pct("returnOnEquity"),
            "roa":          pct("returnOnAssets"),
            "rev_growth":   pct("revenueGrowth"),
            "earn_growth":  pct("earningsGrowth"),
            # Analyst
            "target_mean":  safe("targetMeanPrice",   "{:.2f}"),
            "target_high":  safe("targetHighPrice",   "{:.2f}"),
            "target_low":   safe("targetLowPrice",    "{:.2f}"),
            "recommendation": info.get("recommendationKey", "—").replace("_", " ").title(),
            "analyst_count":  safe("numberOfAnalystOpinions", "{:.0f}"),
            "analyst_summary": analyst_summary,
            # News
            "news": news,
            "description": info.get("longBusinessSummary", ""),
        }
    except Exception:
        return None


def parameter_sweep(df_raw, cfg):
    results = []
    for sf in [20, 50, 100]:
        for ss in [100, 150, 200]:
            if sf >= ss:
                continue
            c = {**cfg, "sma_fast": sf, "sma_slow": ss}
            d = compute_indicators(df_raw.copy(), c)
            d = generate_signals(d, c)
            d = run_backtest(d, c)
            ret    = (d["equity"].iloc[-1] / d["equity"].iloc[0]) - 1
            vol    = d["returns"].std() * np.sqrt(252)
            sharpe = ret / vol if vol else 0
            dd     = ((d["equity"] - d["equity"].cummax()) / d["equity"].cummax()).min()
            results.append({
                "SMA Fast": sf, "SMA Slow": ss,
                "Total Return": f"{ret:.2%}",
                "Sharpe": round(sharpe, 3),
                "Max Drawdown": f"{dd:.2%}",
            })
    return pd.DataFrame(results).sort_values("Sharpe", ascending=False)


# ─────────────────────────────────────────────
#  CHART
# ─────────────────────────────────────────────

def make_chart(df, benchmark, cfg):
    fig = plt.figure(figsize=(14, 10), facecolor="#05080f")
    gs  = gridspec.GridSpec(4, 1, figure=fig, hspace=0.06, height_ratios=[3, 1, 1, 1])

    P = {
        "bg": "#05080f", "text": "#e2e8f0", "grid": "#0e1e35",
        "price": "#94a3b8", "sma_f": "#00c9a7", "sma_s": "#f97316",
        "ema_f": "#5eead4", "ema_s": "#fdba74",
        "buy": "#4ade80", "sell": "#f87171",
        "equity": "#00c9a7", "bench": "#4ade80",
        "rsi": "#c084fc", "atr": "#fb923c",
    }

    def sx(ax):
        ax.set_facecolor(P["bg"])
        ax.tick_params(colors=P["text"], labelsize=7)
        ax.yaxis.label.set_color(P["text"])
        ax.grid(color=P["grid"], linewidth=0.5, alpha=0.7)
        for s in ax.spines.values():
            s.set_edgecolor(P["grid"])

    sf = f"sma_{cfg['sma_fast']}"; ss = f"sma_{cfg['sma_slow']}"
    ef = f"ema_{cfg['ema_fast']}"; es = f"ema_{cfg['ema_slow']}"

    ax1 = fig.add_subplot(gs[0])
    sx(ax1)
    ax1.plot(df.index, df["close"], color=P["price"], lw=0.9, alpha=0.7, label="Close")
    ax1.plot(df.index, df[sf], color=P["sma_f"], lw=1.3, label=f"SMA {cfg['sma_fast']}")
    ax1.plot(df.index, df[ss], color=P["sma_s"], lw=1.3, label=f"SMA {cfg['sma_slow']}")
    ax1.plot(df.index, df[ef], color=P["ema_f"], lw=0.8, ls="--", alpha=0.6, label=f"EMA {cfg['ema_fast']}")
    ax1.plot(df.index, df[es], color=P["ema_s"], lw=0.8, ls="--", alpha=0.6, label=f"EMA {cfg['ema_slow']}")
    entries = df[df["signal"] == 1]
    exits   = df[(df["signal"] == -1) & (df["position"].shift() == 1)]
    ax1.scatter(entries.index, entries["close"], marker="^", s=35, color=P["buy"],  zorder=5, label="Entry")
    ax1.scatter(exits.index,   exits["close"],   marker="v", s=35, color=P["sell"], zorder=5, label="Exit")
    ax1.set_ylabel("Price", color=P["text"], fontsize=8)
    ax1.set_title(f"{cfg['ticker']}  ·  SMA/EMA/RSI/ATR Strategy  |  {cfg['start']} → {cfg['end']}",
                  color=P["text"], fontsize=10, pad=8, fontfamily="monospace")
    ax1.legend(loc="upper left", fontsize=6, facecolor="#080c17", labelcolor=P["text"], framealpha=0.9)
    ax1.set_xticklabels([])

    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    sx(ax2)
    ax2.plot(df.index, df["equity"], color=P["equity"], lw=1.4, label="Strategy")
    ax2.plot(df.index, benchmark,    color=P["bench"],  lw=1.1, ls="--", label="Buy & Hold")
    ax2.set_ylabel("Portfolio", color=P["text"], fontsize=8)
    ax2.legend(loc="upper left", fontsize=6, facecolor="#080c17", labelcolor=P["text"], framealpha=0.9)
    ax2.set_xticklabels([])

    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    sx(ax3)
    ax3.plot(df.index, df["rsi"], color=P["rsi"], lw=0.9)
    ax3.axhline(cfg["rsi_overbought"], color=P["sell"], lw=0.8, ls="--", alpha=0.6)
    ax3.axhline(cfg["rsi_oversold"],   color=P["buy"],  lw=0.8, ls="--", alpha=0.6)
    ax3.set_ylim(0, 100)
    ax3.set_ylabel("RSI", color=P["text"], fontsize=8)
    ax3.set_xticklabels([])

    ax4 = fig.add_subplot(gs[3], sharex=ax1)
    sx(ax4)
    ax4.fill_between(df.index, df["atr"], color=P["atr"], alpha=0.4)
    ax4.plot(df.index, df["atr"], color=P["atr"], lw=0.9)
    ax4.set_ylabel("ATR", color=P["text"], fontsize=8)
    ax4.tick_params(axis="x", colors=P["text"])

    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────
#  AUTO REFRESH (live quote every 30s)
# ─────────────────────────────────────────────

REFRESH_INTERVAL = 30  # seconds
# autorefresh will be set after sidebar reads interval

# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────

with st.sidebar:
    st.markdown("### 🐋 Whaleberg")
    st.markdown("---")

    exchange = st.selectbox("Exchange", ["NYSE", "NSE"])
    raw_ticker = st.text_input(
        "Ticker Symbol",
        value="AAPL" if exchange == "NYSE" else "RELIANCE",
        placeholder="e.g. AAPL or RELIANCE"
    ).upper().strip()

    st.markdown("**Date Range**")
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.text_input("Start", value="2018-01-01")
    with col2:
        end_date = st.text_input("End", value="2024-01-01")

    st.markdown("---")
    st.markdown("**Trend Indicators**")
    sma_fast = st.slider("SMA Fast", 10, 100, 50, step=5)
    sma_slow = st.slider("SMA Slow", 50, 300, 200, step=10)
    ema_fast = st.slider("EMA Fast", 5, 50, 12)
    ema_slow = st.slider("EMA Slow", 10, 100, 26)

    st.markdown("**Momentum**")
    rsi_period     = st.slider("RSI Period", 7, 21, 14)
    rsi_overbought = st.slider("RSI Overbought", 60, 85, 70)
    rsi_oversold   = st.slider("RSI Oversold", 15, 40, 30)

    st.markdown("**Position Sizing**")
    atr_period     = st.slider("ATR Period", 7, 21, 14)
    risk_per_trade = st.slider("Risk per Trade (%)", 0.5, 3.0, 1.0, step=0.1) / 100
    atr_multiplier = st.slider("ATR Multiplier", 1.0, 4.0, 2.0, step=0.5)

    st.markdown("**Capital**")
    initial_capital = st.number_input("Initial Capital", value=100000, step=10000)

    st.markdown("---")
    run_btn = st.button("▶ Run Strategy")
    st.markdown("---")
    refresh_interval = st.selectbox("Live Quote Refresh", [15, 30, 60, 120], index=1, format_func=lambda x: f"Every {x}s")
    REFRESH_INTERVAL = refresh_interval


# ─────────────────────────────────────────────
#  MAIN PANEL
# ─────────────────────────────────────────────

# Set autorefresh with user-selected interval
st_autorefresh(interval=refresh_interval * 1000, key="live_refresh")

st.markdown('<div class="main-header">WHALEBERG</div>', unsafe_allow_html=True)
st.markdown('<div class="main-sub">🐋 &nbsp; Deep market intelligence · SMA · EMA · RSI · ATR</div>', unsafe_allow_html=True)

if not run_btn and "backtest_results" not in st.session_state:
    st.markdown("""
    <div style="background:#080c17; border:1px solid #0e1e35; border-radius:10px; padding:2rem; margin-top:1rem;">
        <p style="font-family:'JetBrains Mono',monospace; font-size:0.85rem; color:#64748b; margin:0;">
            ← Configure your strategy in the sidebar and hit <strong style="color:#00c9a7;">Run Strategy</strong> to begin.
        </p>
        <br/>
        <p style="font-family:'JetBrains Mono',monospace; font-size:0.75rem; color:#334155; margin:0;">
            🌊 &nbsp; NYSE · NSE · Dive deeper than the surface.
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()


# ─────────────────────────────────────────────
#  RUN
# ─────────────────────────────────────────────

suffix     = {"NYSE": "",      "NSE": ".NS"}
BENCHMARKS = {"NYSE": "^GSPC", "NSE": "^NSEI"}

TICKER    = raw_ticker + suffix[exchange]
BENCHMARK = BENCHMARKS[exchange]

cfg = {
    "ticker":          TICKER,
    "benchmark":       BENCHMARK,
    "start":           start_date,
    "end":             end_date,
    "sma_fast":        sma_fast,
    "sma_slow":        sma_slow,
    "ema_fast":        ema_fast,
    "ema_slow":        ema_slow,
    "rsi_period":      rsi_period,
    "rsi_overbought":  rsi_overbought,
    "rsi_oversold":    rsi_oversold,
    "atr_period":      atr_period,
    "risk_per_trade":  risk_per_trade,
    "atr_multiplier":  atr_multiplier,
    "initial_capital": initial_capital,
}

# ── Run backtest only when button is clicked, cache results ──
if run_btn:
    with st.spinner(f"Fetching {TICKER} data..."):
        try:
            df_raw = yf.download(TICKER, start=start_date, end=end_date, progress=False, auto_adjust=True)
            if isinstance(df_raw.columns, pd.MultiIndex):
                df_raw.columns = df_raw.columns.droplevel(1)
            df_raw.columns = [c.lower() for c in df_raw.columns]
            df_raw.dropna(inplace=True)
            if len(df_raw) == 0:
                st.error(f"No data found for **{TICKER}**. Check the ticker symbol and try again.")
                st.stop()
        except Exception as e:
            st.error(f"Failed to fetch data: {e}")
            st.stop()

    with st.spinner("Running backtest..."):
        df = compute_indicators(df_raw.copy(), cfg)
        df = generate_signals(df, cfg)
        df = run_backtest(df, cfg)
        benchmark = run_benchmark(df, cfg)
        m_strat = compute_metrics(df["equity"],  "Strategy")
        m_bench = compute_metrics(benchmark,     "Buy & Hold")

    st.session_state["backtest_results"] = {
        "df": df, "df_raw": df_raw, "benchmark": benchmark,
        "m_strat": m_strat, "m_bench": m_bench,
        "cfg": cfg, "TICKER": TICKER, "exchange": exchange,
        "start_date": start_date, "end_date": end_date,
    }

# ── Load from cache (also used on auto-refresh) ──
if "backtest_results" not in st.session_state:
    st.stop()

cached      = st.session_state["backtest_results"]
df          = cached["df"]
df_raw      = cached["df_raw"]
benchmark   = cached["benchmark"]
m_strat     = cached["m_strat"]
m_bench     = cached["m_bench"]
cfg         = cached["cfg"]
TICKER      = cached["TICKER"]
exchange    = cached["exchange"]
start_date  = cached["start_date"]
end_date    = cached["end_date"]

# ── Live indicator + last updated ──
last_updated = time.strftime("%H:%M:%S")
st.markdown(f"""
<div style="display:flex; align-items:center; gap:0.6rem; margin-bottom:0.5rem;">
    <span style="width:8px; height:8px; border-radius:50%; background:#4ade80; display:inline-block; animation:pulse 2s infinite;"></span>
    <span style="font-family:'JetBrains Mono',monospace; font-size:0.65rem; color:#4ade80;">LIVE</span>
    <span style="font-family:'JetBrains Mono',monospace; font-size:0.65rem; color:#334155;">· Quote refreshes every {refresh_interval}s · Last updated {last_updated} UTC · Backtest pinned to last run</span>
</div>
<style>
@keyframes pulse {{
    0%, 100% {{ opacity: 1; }}
    50% {{ opacity: 0.3; }}
}}
</style>
""", unsafe_allow_html=True)

# ── Ticker tag + period ──
st.markdown(f"""
<div style="margin-bottom:1rem;">
    <span class="ticker-tag">{TICKER}</span>
    <span class="ticker-tag">{exchange}</span>
    <span style="font-family:'JetBrains Mono',monospace; font-size:0.75rem; color:#475569;">
        {start_date} → {end_date} · {len(df_raw)} trading days
    </span>
</div>
""", unsafe_allow_html=True)

# ── Bloomberg Info Panel ──
with st.spinner("Loading company data..."):
    info = get_stock_info(TICKER)

# Fallback so tabs always render even if yfinance returns nothing
if not info:
    info = {
        "name": TICKER, "sector": "—", "industry": "—", "country": "—",
        "exchange": "—", "currency": "", "curr_price": "—", "day_chg": None,
        "prev_close": "—", "open": "—", "high": "—", "low": "—", "volume": "—",
        "week52_high": "—", "week52_low": "—", "mkt_cap": "—", "enterprise_val": "—",
        "pe_ratio": "—", "fwd_pe": "—", "peg": "—", "ps_ratio": "—", "pb_ratio": "—",
        "ev_ebitda": "—", "eps": "—", "fwd_eps": "—", "beta": "—",
        "div_yield": "—", "div_rate": "—", "payout_ratio": "—",
        "avg_volume": "—", "float_shares": "—", "shares_out": "—",
        "short_pct": "—", "inst_own": "—", "insider_own": "—",
        "revenue": "—", "gross_margin": "—", "op_margin": "—", "net_margin": "—",
        "ebitda": "—", "free_cashflow": "—", "total_cash": "—", "total_debt": "—",
        "debt_equity": "—", "current_ratio": "—", "roe": "—", "roa": "—",
        "rev_growth": "—", "earn_growth": "—",
        "target_mean": "—", "target_high": "—", "target_low": "—",
        "recommendation": "—", "analyst_count": "—", "analyst_summary": {},
        "news": [], "description": "Company data unavailable — yfinance may be rate-limiting. Try again shortly.",
    }

if info:
    chg_val  = info["day_chg"]
    chg_html = ""
    if chg_val is not None:
        sign     = "+" if chg_val >= 0 else ""
        cls      = "bbg-chg-pos" if chg_val >= 0 else "bbg-chg-neg"
        chg_html = f'<span class="{cls}">{sign}{chg_val:.2f}%</span>'

    # ── Header bar (always visible) ──
    st.markdown(f"""
    <div class="bbg-panel" style="padding-bottom:1rem;">
        <div style="display:flex; justify-content:space-between; align-items:flex-start; flex-wrap:wrap; gap:1rem;">
            <div>
                <div class="bbg-name">{info["name"]}</div>
                <div class="bbg-meta">{info["sector"]} &nbsp;·&nbsp; {info["industry"]} &nbsp;·&nbsp; {info["country"]} &nbsp;·&nbsp; {info["exchange"]}</div>
                <div style="margin-top:0.5rem;">
                    <span class="bbg-price">{info["currency"]} {info["curr_price"]}</span>
                    {chg_html}
                </div>
            </div>
            <div style="display:grid; grid-template-columns:repeat(4,1fr); gap:0.5rem 2rem; align-self:flex-end;">
                <div class="bbg-kv"><span class="bbg-k">Open</span>    <span class="bbg-v">{info["open"]}</span></div>
                <div class="bbg-kv"><span class="bbg-k">High</span>    <span class="bbg-v">{info["high"]}</span></div>
                <div class="bbg-kv"><span class="bbg-k">Low</span>     <span class="bbg-v">{info["low"]}</span></div>
                <div class="bbg-kv"><span class="bbg-k">Volume</span>  <span class="bbg-v">{info["volume"]}</span></div>
                <div class="bbg-kv"><span class="bbg-k">Prev Close</span><span class="bbg-v">{info["prev_close"]}</span></div>
                <div class="bbg-kv"><span class="bbg-k">52W High</span><span class="bbg-v">{info["week52_high"]}</span></div>
                <div class="bbg-kv"><span class="bbg-k">52W Low</span> <span class="bbg-v">{info["week52_low"]}</span></div>
                <div class="bbg-kv"><span class="bbg-k">Avg Vol</span> <span class="bbg-v">{info["avg_volume"]}</span></div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Tabs ──
    tab_ov, tab_fin, tab_val, tab_own, tab_analyst, tab_news = st.tabs([
        "📊 Overview", "💰 Financials", "📐 Valuation", "🏦 Ownership", "🎯 Analysts", "📰 News"
    ])

    with tab_ov:
        desc = info["description"]
        if len(desc) > 600:
            desc = desc[:600].rsplit(" ", 1)[0] + "…"
        st.markdown(f"""
        <div class="bbg-grid" style="grid-template-columns:repeat(4,1fr); margin-top:1rem;">
            <div class="bbg-kv"><span class="bbg-k">Market Cap</span>     <span class="bbg-v">{info["mkt_cap"]}</span></div>
            <div class="bbg-kv"><span class="bbg-k">Enterprise Val</span> <span class="bbg-v">{info["enterprise_val"]}</span></div>
            <div class="bbg-kv"><span class="bbg-k">Beta</span>           <span class="bbg-v">{info["beta"]}</span></div>
            <div class="bbg-kv"><span class="bbg-k">Float Shares</span>   <span class="bbg-v">{info["float_shares"]}</span></div>
            <div class="bbg-kv"><span class="bbg-k">Shares Out</span>     <span class="bbg-v">{info["shares_out"]}</span></div>
            <div class="bbg-kv"><span class="bbg-k">Div Yield</span>      <span class="bbg-v">{info["div_yield"]}</span></div>
            <div class="bbg-kv"><span class="bbg-k">Div Rate</span>       <span class="bbg-v">{info["div_rate"]}</span></div>
            <div class="bbg-kv"><span class="bbg-k">Payout Ratio</span>   <span class="bbg-v">{info["payout_ratio"]}</span></div>
        </div>
        {"<div class='bbg-desc' style='margin-top:1rem;'>" + desc + "</div>" if desc else ""}
        """, unsafe_allow_html=True)

    with tab_fin:
        st.markdown(f"""
        <div class="bbg-grid" style="grid-template-columns:repeat(4,1fr); margin-top:1rem;">
            <div class="bbg-kv"><span class="bbg-k">Revenue (TTM)</span>   <span class="bbg-v">{info["revenue"]}</span></div>
            <div class="bbg-kv"><span class="bbg-k">EBITDA</span>          <span class="bbg-v">{info["ebitda"]}</span></div>
            <div class="bbg-kv"><span class="bbg-k">Free Cash Flow</span>  <span class="bbg-v">{info["free_cashflow"]}</span></div>
            <div class="bbg-kv"><span class="bbg-k">Total Cash</span>      <span class="bbg-v">{info["total_cash"]}</span></div>
            <div class="bbg-kv"><span class="bbg-k">Total Debt</span>      <span class="bbg-v">{info["total_debt"]}</span></div>
            <div class="bbg-kv"><span class="bbg-k">Debt / Equity</span>   <span class="bbg-v">{info["debt_equity"]}</span></div>
            <div class="bbg-kv"><span class="bbg-k">Current Ratio</span>   <span class="bbg-v">{info["current_ratio"]}</span></div>
            <div class="bbg-kv"><span class="bbg-k">Rev Growth (YoY)</span><span class="bbg-v">{info["rev_growth"]}</span></div>
            <div class="bbg-kv"><span class="bbg-k">Earn Growth (YoY)</span><span class="bbg-v">{info["earn_growth"]}</span></div>
            <div class="bbg-kv"><span class="bbg-k">Gross Margin</span>    <span class="bbg-v">{info["gross_margin"]}</span></div>
            <div class="bbg-kv"><span class="bbg-k">Op. Margin</span>      <span class="bbg-v">{info["op_margin"]}</span></div>
            <div class="bbg-kv"><span class="bbg-k">Net Margin</span>      <span class="bbg-v">{info["net_margin"]}</span></div>
            <div class="bbg-kv"><span class="bbg-k">ROE</span>             <span class="bbg-v">{info["roe"]}</span></div>
            <div class="bbg-kv"><span class="bbg-k">ROA</span>             <span class="bbg-v">{info["roa"]}</span></div>
        </div>
        """, unsafe_allow_html=True)

    with tab_val:
        st.markdown(f"""
        <div class="bbg-grid" style="grid-template-columns:repeat(4,1fr); margin-top:1rem;">
            <div class="bbg-kv"><span class="bbg-k">P/E (TTM)</span>    <span class="bbg-v">{info["pe_ratio"]}</span></div>
            <div class="bbg-kv"><span class="bbg-k">Fwd P/E</span>      <span class="bbg-v">{info["fwd_pe"]}</span></div>
            <div class="bbg-kv"><span class="bbg-k">PEG Ratio</span>    <span class="bbg-v">{info["peg"]}</span></div>
            <div class="bbg-kv"><span class="bbg-k">P/S (TTM)</span>    <span class="bbg-v">{info["ps_ratio"]}</span></div>
            <div class="bbg-kv"><span class="bbg-k">P/B</span>          <span class="bbg-v">{info["pb_ratio"]}</span></div>
            <div class="bbg-kv"><span class="bbg-k">EV/EBITDA</span>    <span class="bbg-v">{info["ev_ebitda"]}</span></div>
            <div class="bbg-kv"><span class="bbg-k">EPS (TTM)</span>    <span class="bbg-v">{info["eps"]}</span></div>
            <div class="bbg-kv"><span class="bbg-k">Fwd EPS</span>      <span class="bbg-v">{info["fwd_eps"]}</span></div>
        </div>
        """, unsafe_allow_html=True)

    with tab_own:
        st.markdown(f"""
        <div class="bbg-grid" style="grid-template-columns:repeat(4,1fr); margin-top:1rem;">
            <div class="bbg-kv"><span class="bbg-k">Institutional Own</span><span class="bbg-v">{info["inst_own"]}</span></div>
            <div class="bbg-kv"><span class="bbg-k">Insider Own</span>      <span class="bbg-v">{info["insider_own"]}</span></div>
            <div class="bbg-kv"><span class="bbg-k">Short % of Float</span> <span class="bbg-v">{info["short_pct"]}</span></div>
            <div class="bbg-kv"><span class="bbg-k">Float Shares</span>     <span class="bbg-v">{info["float_shares"]}</span></div>
            <div class="bbg-kv"><span class="bbg-k">Shares Out</span>       <span class="bbg-v">{info["shares_out"]}</span></div>
        </div>
        """, unsafe_allow_html=True)

    with tab_analyst:
        rec_key = info["recommendation"]
        rec_color = {"Strong Buy":"#4ade80","Buy":"#86efac","Hold":"#fbbf24","Sell":"#f87171","Strong Sell":"#ef4444"}.get(rec_key, "#94a3b8")
        summary = info["analyst_summary"]

        # Stats row
        st.markdown(f"""
        <div style="display:flex; align-items:center; gap:1.5rem; margin:1rem 0 1.2rem 0;">
            <div>
                <span style="font-family:'JetBrains Mono',monospace; font-size:0.6rem; color:#475569; text-transform:uppercase;">Consensus</span><br/>
                <span style="font-family:'JetBrains Mono',monospace; font-size:1.4rem; font-weight:600; color:{rec_color};">{rec_key}</span>
            </div>
            <div>
                <span style="font-family:'JetBrains Mono',monospace; font-size:0.6rem; color:#475569; text-transform:uppercase;">Analysts</span><br/>
                <span style="font-family:'JetBrains Mono',monospace; font-size:1.4rem; font-weight:600; color:#f1f5f9;">{info["analyst_count"]}</span>
            </div>
            <div>
                <span style="font-family:'JetBrains Mono',monospace; font-size:0.6rem; color:#475569; text-transform:uppercase;">Price Target</span><br/>
                <span style="font-family:'JetBrains Mono',monospace; font-size:1.4rem; font-weight:600; color:#00c9a7;">{info["currency"]} {info["target_mean"]}</span>
            </div>
            <div>
                <span style="font-family:'JetBrains Mono',monospace; font-size:0.6rem; color:#475569; text-transform:uppercase;">Range</span><br/>
                <span style="font-family:'JetBrains Mono',monospace; font-size:0.9rem; color:#94a3b8;">{info["target_low"]} – {info["target_high"]}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Bars — each rendered separately to avoid f-string rendering issues
        if summary:
            total = sum(summary.values()) or 1
            label_map = {"strongBuy":"Strong Buy","buy":"Buy","hold":"Hold","sell":"Sell","strongSell":"Strong Sell"}
            color_map = {"strongBuy":"#4ade80","buy":"#86efac","hold":"#fbbf24","sell":"#f87171","strongSell":"#ef4444"}
            for k, v in summary.items():
                pct_w = round(v / total * 100)
                lbl   = label_map.get(k, k)
                col   = color_map.get(k, "#94a3b8")
                st.markdown(
                    f'<div style="display:flex;align-items:center;gap:0.8rem;margin-bottom:0.5rem;">'
                    f'<span style="font-family:\'IBM Plex Mono\',monospace;font-size:0.7rem;color:#64748b;width:80px;text-align:right;">{lbl}</span>'
                    f'<div style="flex:1;background:#0e1e35;border-radius:3px;height:14px;">'
                    f'<div style="width:{pct_w}%;background:{col};height:14px;border-radius:3px;"></div>'
                    f'</div>'
                    f'<span style="font-family:\'IBM Plex Mono\',monospace;font-size:0.7rem;color:#e2e8f0;width:20px;">{v}</span>'
                    f'</div>',
                    unsafe_allow_html=True
                )

    with tab_news:
        news_items = info.get("news", [])
        if news_items:
            for n in news_items:
                title     = n.get("title", "")
                publisher = n.get("publisher", "")
                link      = n.get("link", "")
                time_str  = n.get("time", "")
                link_html = f'<a href="{link}" target="_blank" style="color:#00c9a7; text-decoration:none;">{title}</a>' if link else f'<span style="color:#e2e8f0;">{title}</span>'
                st.markdown(f"""
                <div style="padding:0.8rem 0; border-bottom:1px solid #0e1e35;">
                    <div style="font-family:'Syne',sans-serif; font-size:0.85rem; margin-bottom:0.25rem;">{link_html}</div>
                    <div style="font-family:'JetBrains Mono',monospace; font-size:0.65rem; color:#475569;">{publisher} &nbsp;·&nbsp; {time_str}</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown('<p style="font-family:\'IBM Plex Mono\',monospace; font-size:0.8rem; color:#475569;">No recent news available.</p>', unsafe_allow_html=True)

st.markdown("<hr style='border-color:#0e1e35; margin: 0.5rem 0 1rem 0;'>", unsafe_allow_html=True)

# ── Metrics row ──
c1, c2, c3, c4, c5, c6, c7 = st.columns(7)

def fmt_pct(v): return f"{v:.2%}"
def fmt_2f(v):  return f"{v:.2f}"
def delta_color(v): return "normal" if v >= 0 else "inverse"

with c1: st.metric("Total Return (S)",  fmt_pct(m_strat["Total Return"]),  delta=fmt_pct(m_strat["Total Return"] - m_bench["Total Return"]))
with c2: st.metric("Total Return (B&H)", fmt_pct(m_bench["Total Return"]))
with c3: st.metric("Sharpe (S)",        fmt_2f(m_strat["Sharpe Ratio"]),   delta=fmt_2f(m_strat["Sharpe Ratio"] - m_bench["Sharpe Ratio"]))
with c4: st.metric("Sharpe (B&H)",      fmt_2f(m_bench["Sharpe Ratio"]))
with c5: st.metric("Max DD (S)",        fmt_pct(m_strat["Max Drawdown"]))
with c6: st.metric("Max DD (B&H)",      fmt_pct(m_bench["Max Drawdown"]))
with c7: st.metric("Win Rate",          fmt_pct(m_strat["Win Rate"]))

st.markdown("<hr style='border-color:#0e1e35; margin: 1rem 0;'>", unsafe_allow_html=True)

# ── Chart ──
fig = make_chart(df, benchmark, cfg)
st.pyplot(fig, use_container_width=True)

st.markdown("<hr style='border-color:#0e1e35; margin: 1rem 0;'>", unsafe_allow_html=True)

# ── Parameter sweep ──
st.markdown("#### 🔍 Parameter Sweep — SMA Grid")
with st.spinner("Running parameter sweep..."):
    sweep = parameter_sweep(df_raw, cfg)

st.dataframe(
    sweep.reset_index(drop=True),
    use_container_width=True,
    hide_index=True,
)

st.markdown("""
<p style="font-family:'JetBrains Mono',monospace; font-size:0.65rem; color:#334155; margin-top:1rem;">
    🐋 Whaleberg · For educational purposes only. Not financial advice. The market has no bottom.
</p>
""", unsafe_allow_html=True)
