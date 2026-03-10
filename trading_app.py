"""
Trading Strategy App — SMA, EMA, RSI & ATR
Run with: streamlit run trading_app.py
"""





import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package, "-q"])

required = ["streamlit", "yfinance", "pandas", "numpy", "matplotlib"]

for package in required:
    try:
        __import__(package)
    except ImportError:
        print(f"Installing {package}...")
        install(package)


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import yfinance as yf
import streamlit as st
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="AlgoTrader",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
    background-color: #0a0e1a;
    color: #e2e8f0;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #0d1224;
    border-right: 1px solid #1e2a45;
}
section[data-testid="stSidebar"] * {
    color: #e2e8f0 !important;
}

/* Main header */
.main-header {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 2.2rem;
    font-weight: 600;
    color: #38bdf8;
    letter-spacing: -0.02em;
    margin-bottom: 0.1rem;
}
.main-sub {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.8rem;
    color: #475569;
    letter-spacing: 0.1em;
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
    background: #0d1224;
    border: 1px solid #1e2a45;
    border-radius: 8px;
    padding: 1rem 1.2rem;
}
.metric-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.65rem;
    color: #475569;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 0.4rem;
}
.metric-value {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.4rem;
    font-weight: 600;
    color: #f1f5f9;
}
.metric-value.positive { color: #4ade80; }
.metric-value.negative { color: #f87171; }
.metric-value.neutral  { color: #38bdf8; }

/* Divider */
.section-divider {
    border: none;
    border-top: 1px solid #1e2a45;
    margin: 1.5rem 0;
}

/* Tag */
.ticker-tag {
    display: inline-block;
    background: #1e2a45;
    border: 1px solid #38bdf8;
    border-radius: 4px;
    padding: 0.2rem 0.6rem;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem;
    color: #38bdf8;
    margin-right: 0.5rem;
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
    background: #0d1224;
    border: 1px solid #1e2a45;
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
    background: #38bdf8;
    color: #0a0e1a;
    font-family: 'IBM Plex Mono', monospace;
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
    background: #7dd3fc;
    color: #0a0e1a;
}
.stDataFrame {
    font-family: 'IBM Plex Mono', monospace;
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
    if row["atr"] == 0 or np.isnan(row["atr"]):
        return 0.0
    risk_dollars  = portfolio_value * cfg["risk_per_trade"]
    stop_distance = row["atr"] * cfg["atr_multiplier"]
    shares        = risk_dollars / stop_distance
    max_shares    = portfolio_value / row["close"]
    return min(shares, max_shares)

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
    fig = plt.figure(figsize=(14, 10), facecolor="#0a0e1a")
    gs  = gridspec.GridSpec(4, 1, figure=fig, hspace=0.06, height_ratios=[3, 1, 1, 1])

    P = {
        "bg": "#0a0e1a", "text": "#e2e8f0", "grid": "#1e2a45",
        "price": "#94a3b8", "sma_f": "#38bdf8", "sma_s": "#f97316",
        "ema_f": "#7dd3fc", "ema_s": "#fdba74",
        "buy": "#4ade80", "sell": "#f87171",
        "equity": "#38bdf8", "bench": "#4ade80",
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
    ax1.legend(loc="upper left", fontsize=6, facecolor="#0d1224", labelcolor=P["text"], framealpha=0.9)
    ax1.set_xticklabels([])

    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    sx(ax2)
    ax2.plot(df.index, df["equity"], color=P["equity"], lw=1.4, label="Strategy")
    ax2.plot(df.index, benchmark,    color=P["bench"],  lw=1.1, ls="--", label="Buy & Hold")
    ax2.set_ylabel("Portfolio", color=P["text"], fontsize=8)
    ax2.legend(loc="upper left", fontsize=6, facecolor="#0d1224", labelcolor=P["text"], framealpha=0.9)
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
#  SIDEBAR
# ─────────────────────────────────────────────

with st.sidebar:
    st.markdown("### ⚙️ Configuration")
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


# ─────────────────────────────────────────────
#  MAIN PANEL
# ─────────────────────────────────────────────

st.markdown('<div class="main-header">Project WhiteSand</div>', unsafe_allow_html=True)
st.markdown('<div class="main-sub">SMA · EMA · RSI · ATR — Rule-Based Strategy Backtester</div>', unsafe_allow_html=True)

if not run_btn:
    st.markdown("""
    <div style="background:#0d1224; border:1px solid #1e2a45; border-radius:10px; padding:2rem; margin-top:1rem;">
        <p style="font-family:'IBM Plex Mono',monospace; font-size:0.85rem; color:#64748b; margin:0;">
            ← Configure your strategy in the sidebar and hit <strong style="color:#38bdf8;">Run Strategy</strong> to begin.
        </p>
        <br/>
        <p style="font-family:'IBM Plex Mono',monospace; font-size:0.75rem; color:#334155; margin:0;">
            Supports NYSE (e.g. AAPL, TSLA, NVDA) and NSE (e.g. RELIANCE, TCS, INFY)
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

# ── Ticker tag + period ──
st.markdown(f"""
<div style="margin-bottom:1rem;">
    <span class="ticker-tag">{TICKER}</span>
    <span class="ticker-tag">{exchange}</span>
    <span style="font-family:'IBM Plex Mono',monospace; font-size:0.75rem; color:#475569;">
        {start_date} → {end_date} · {len(df_raw)} trading days
    </span>
</div>
""", unsafe_allow_html=True)

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

st.markdown("<hr style='border-color:#1e2a45; margin: 1rem 0;'>", unsafe_allow_html=True)

# ── Chart ──
fig = make_chart(df, benchmark, cfg)
st.pyplot(fig, use_container_width=True)

st.markdown("<hr style='border-color:#1e2a45; margin: 1rem 0;'>", unsafe_allow_html=True)

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
<p style="font-family:'IBM Plex Mono',monospace; font-size:0.65rem; color:#334155; margin-top:1rem;">
    Strategy backtests are for educational purposes only. Not financial advice.
</p>
""", unsafe_allow_html=True)


