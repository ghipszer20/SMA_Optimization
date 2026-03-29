import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime

# ---------------- INPUT HELPERS ----------------
def get_date(prompt):
    while True:
        try:
            return datetime.strptime(input(prompt), "%m/%d/%Y")
        except ValueError:
            print("Invalid format. Please enter MM/DD/YYYY")

def get_float(prompt, default):
    raw = input(f"{prompt} [default {default}]: ").strip()
    if raw == "":
        return float(default)
    return float(raw)

def get_choice(prompt, choices, default=None):
    choices_lower = [c.lower() for c in choices]
    while True:
        raw = input(prompt).strip().lower()
        if raw == "" and default is not None:
            return default.lower()
        if raw in choices_lower:
            return raw
        print(f"Choose one of: {choices}")

# ---------------- USER INPUT ----------------
TICKER = input("Enter ticker symbol: ").strip().upper()
ASSET_TYPE = get_choice("Asset type? (crypto/stock) [default stock]: ", ["crypto", "stock"], default="stock")
start_date = get_date("Enter start date (MM/DD/YYYY): ")
end_date   = get_date("Enter end date (MM/DD/YYYY): ")

# ---------------- SETTINGS ----------------
INITIAL_CAPITAL = 10000.0
SMA_MIN = 1
SMA_MAX = 200

PLOT_TOP_10 = True
SAVE_RESULTS_OPTION = True
SHOW_TOP_10_TABLE = True

# ---------------- COST MODEL ----------------
# Robinhood-style assumptions:
# - stocks/ETFs: no commission, but small slippage/spread
# - crypto: wider effective spread
if ASSET_TYPE == "crypto":
    default_spread = 0.0086   # 0.86%
    annualization_factor = 365
else:
    default_spread = 0.0005   # 0.05%
    annualization_factor = 252

RH_SPREAD = get_float("Enter total spread cost as decimal", default_spread)
EXTRA_SLIPPAGE = get_float("Enter extra slippage per side as decimal", 0.0)

# Per-side execution penalty
HALF_COST = (RH_SPREAD / 2.0) + EXTRA_SLIPPAGE

# ---------------- DOWNLOAD DATA ----------------
data = yf.download(
    TICKER,
    start=start_date,
    end=end_date,
    auto_adjust=True,
    progress=False
).dropna()

data.columns = [c[0] if isinstance(c, tuple) else c for c in data.columns]
data = data.loc[:, ~data.columns.duplicated()].copy()

needed = {"Open", "Close"}
missing = needed - set(data.columns)
if missing:
    raise ValueError(f"Missing columns {missing}. Available: {list(data.columns)}")

if len(data) < SMA_MAX + 5:
    raise ValueError("Not enough data for SMA testing up to 200.")

# ---------------- BENCHMARK ----------------
first_open = float(data["Open"].iloc[0])
bh_entry_fill = first_open * (1.0 + HALF_COST)
bh_units = INITIAL_CAPITAL / bh_entry_fill
benchmark_curve = bh_units * data["Close"]
benchmark_pct = benchmark_curve / INITIAL_CAPITAL * 100.0

# ---------------- KELLY HELPER ----------------
def compute_kelly(trade_returns):
    """
    Kelly formula used:
        k = p - (1 - p) / w
    where
        p = win probability
        w = avg win size / avg loss size

    Returns decimals, not percentages.
    """
    if len(trade_returns) == 0:
        return np.nan, np.nan, np.nan, np.nan, np.nan

    wins = [r for r in trade_returns if r > 0]
    losses = [r for r in trade_returns if r < 0]

    p = len(wins) / len(trade_returns)

    avg_win = np.mean(wins) if wins else np.nan
    avg_loss = abs(np.mean(losses)) if losses else np.nan

    if not wins or not losses or avg_loss == 0 or np.isnan(avg_loss):
        return p, avg_win, avg_loss, np.nan, np.nan

    w = avg_win / avg_loss
    kelly = p - (1 - p) / w
    half_kelly = kelly / 2.0

    return p, avg_win, avg_loss, kelly, half_kelly

# ---------------- BACKTEST FUNCTION ----------------
def run_sma_backtest(base_data, sma_period):
    df = base_data.copy()

    sma_col = f"SMA{sma_period}"
    df[sma_col] = df["Close"].rolling(sma_period).mean()

    # Signal from close vs SMA
    df["Signal"] = (df["Close"] > df[sma_col]).astype(int)

    # Execute at next day's open
    df["Position"] = df["Signal"].shift(1)

    df = df.dropna().copy()
    if df.empty:
        return None

    df["Position"] = df["Position"].astype(int)

    cash = INITIAL_CAPITAL
    units = 0.0
    portfolio = []

    in_pos = False
    entry_fill = None
    trade_returns = []

    for _, row in df.iterrows():
        open_price = float(row["Open"])
        close_price = float(row["Close"])
        pos = int(row["Position"])

        buy_fill = open_price * (1.0 + HALF_COST)
        sell_fill = open_price * (1.0 - HALF_COST)

        # Enter at next open
        if pos == 1 and not in_pos:
            units = cash / buy_fill
            cash = 0.0
            in_pos = True
            entry_fill = buy_fill

        # Exit at next open
        elif pos == 0 and in_pos:
            cash = units * sell_fill
            units = 0.0
            in_pos = False
            trade_returns.append((sell_fill - entry_fill) / entry_fill)
            entry_fill = None

        total = cash + units * close_price
        portfolio.append(total)

    df["Portfolio"] = portfolio
    df["Strategy %"] = df["Portfolio"] / INITIAL_CAPITAL * 100.0

    final_value = float(df["Portfolio"].iloc[-1])
    total_return = final_value / INITIAL_CAPITAL - 1.0

    days = (df.index[-1] - df.index[0]).days
    years = days / 365.25 if days > 0 else np.nan
    cagr = (final_value / INITIAL_CAPITAL) ** (1 / years) - 1 if years and years > 0 else np.nan

    daily_returns = df["Portfolio"].pct_change().dropna()

    if len(daily_returns) > 1 and daily_returns.std() != 0:
        sharpe = np.sqrt(annualization_factor) * daily_returns.mean() / daily_returns.std()
    else:
        sharpe = 0.0

    downside = daily_returns[daily_returns < 0]
    if len(downside) > 1 and downside.std() != 0:
        sortino = np.sqrt(annualization_factor) * daily_returns.mean() / downside.std()
    else:
        sortino = 0.0

    rolling_max = df["Portfolio"].cummax()
    drawdown = df["Portfolio"] / rolling_max - 1.0
    max_dd = float(drawdown.min())

    win_rate, avg_win, avg_loss, kelly, half_kelly = compute_kelly(trade_returns)

    result = {
        "SMA": sma_period,
        "Final Value": final_value,
        "Total Return %": total_return * 100.0,
        "CAGR %": cagr * 100.0 if pd.notna(cagr) else np.nan,
        "Sharpe": sharpe,
        "Sortino": sortino,
        "Max Drawdown %": max_dd * 100.0,
        "Trades": len(trade_returns),
        "Win Rate %": win_rate * 100.0 if pd.notna(win_rate) else np.nan,
        "Avg Win %": avg_win * 100.0 if pd.notna(avg_win) else np.nan,
        "Avg Loss %": avg_loss * 100.0 if pd.notna(avg_loss) else np.nan,
        "Kelly %": kelly * 100.0 if pd.notna(kelly) else np.nan,
        "Half Kelly %": half_kelly * 100.0 if pd.notna(half_kelly) else np.nan,
    }

    return result, df[["Strategy %"]].copy()

# ---------------- RUN ALL SMA TESTS ----------------
results_list = []
equity_curves = {}

for sma in range(SMA_MIN, SMA_MAX + 1):
    output = run_sma_backtest(data, sma)
    if output is None:
        continue

    metrics, curve = output
    results_list.append(metrics)
    equity_curves[sma] = curve["Strategy %"]

results = pd.DataFrame(results_list)

if results.empty:
    raise ValueError("No valid SMA results were produced.")

results = results.sort_values("SMA").reset_index(drop=True)

# ---------------- PRINT RESULTS ----------------
pd.set_option("display.max_rows", None)
pd.set_option("display.width", 240)
pd.set_option("display.max_columns", None)

print("\n========== SETTINGS ==========")
print(f"Ticker: {TICKER}")
print(f"Asset type: {ASSET_TYPE}")
print(f"Spread assumption: {RH_SPREAD * 100:.4f}% total")
print(f"Extra slippage: {EXTRA_SLIPPAGE * 100:.4f}% per side")
print(f"Per-side trading cost used: {HALF_COST * 100:.4f}%")
print("==============================")

print("\n========== ALL SMA RESULTS ==========")
print(results.to_string(index=False))
print("=====================================")

best_row = results.loc[results["Total Return %"].idxmax()]
best_sma = int(best_row["SMA"])
best_curve = equity_curves[best_sma]

print("\n========== BEST SMA BY TOTAL RETURN ==========")
print(best_row.to_frame().T.to_string(index=False))
print("==============================================")

if SHOW_TOP_10_TABLE:
    print("\n========== TOP 10 SMAS BY TOTAL RETURN ==========")
    print(results.sort_values("Total Return %", ascending=False).head(10).to_string(index=False))
    print("=================================================")

# ---------------- PLOT: BEST STRATEGY VS BUY & HOLD ----------------
plt.figure(figsize=(14, 8))

plt.plot(
    best_curve.index,
    best_curve.values,
    linewidth=3,
    label=f"Best Strategy (SMA{best_sma})"
)

plt.plot(
    benchmark_pct.index,
    benchmark_pct.values,
    linewidth=2.5,
    linestyle="--",
    label=f"Buy & Hold {TICKER}"
)

plt.title(f"{TICKER} Best SMA Strategy vs Buy & Hold")
plt.xlabel("Date")
plt.ylabel("Value (% of Start)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# ---------------- OPTIONAL PLOT: TOP 10 EQUITY CURVES ----------------
if PLOT_TOP_10:
    top_n = 10
    top_smas = results.sort_values("Total Return %", ascending=False).head(top_n)["SMA"].tolist()

    plt.figure(figsize=(14, 8))
    for sma in top_smas:
        curve = equity_curves[sma]
        plt.plot(curve.index, curve.values, linewidth=1.8, label=f"SMA{sma}")

    plt.plot(
        benchmark_pct.index,
        benchmark_pct.values,
        linewidth=2.8,
        linestyle="--",
        label=f"Buy & Hold {TICKER}"
    )

    plt.title(f"{TICKER} Top {top_n} SMA Strategies vs Buy & Hold")
    plt.xlabel("Date")
    plt.ylabel("Value (% of Start)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

# ---------------- OPTIONAL SAVE ----------------
if SAVE_RESULTS_OPTION:
    save_csv = input("\nSave results to CSV? (y/n): ").strip().lower()
    if save_csv == "y":
        filename = f"{TICKER.replace('-', '_')}_sma_{SMA_MIN}_{SMA_MAX}_results.csv"
        results.to_csv(filename, index=False)
        print(f"Saved results to {filename}")