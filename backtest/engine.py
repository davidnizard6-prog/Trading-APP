import pandas as pd
import numpy as np


def run_backtest(df: pd.DataFrame, initial_capital: float = 10_000.0, commission: float = 0.001) -> dict:
    """
    Simple event-driven backtest on a dataframe with 'close' and 'signal' columns.
    commission: fraction per trade (e.g. 0.001 = 0.1%)
    """
    cash = initial_capital
    shares = 0.0
    equity_curve = []
    trades = []

    for date, row in df.iterrows():
        price = row["close"]

        if row["signal"] == 1.0 and cash > 0:  # Golden Cross — buy
            shares = (cash * (1 - commission)) / price
            cost = shares * price
            cash -= cost * (1 + commission)
            trades.append({"date": date, "type": "BUY", "price": price, "shares": shares})

        elif row["signal"] == -1.0 and shares > 0:  # Death Cross — sell
            proceeds = shares * price * (1 - commission)
            cash += proceeds
            trades.append({"date": date, "type": "SELL", "price": price, "shares": shares})
            shares = 0.0

        equity = cash + shares * price
        equity_curve.append({"date": date, "equity": equity, "cash": cash, "shares": shares})

    equity_df = pd.DataFrame(equity_curve).set_index("date")
    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame(columns=["date", "type", "price", "shares"])

    # Buy & hold benchmark
    buy_hold_return = (df["close"].iloc[-1] / df["close"].iloc[0] - 1) * 100

    # Strategy metrics
    total_return = (equity_df["equity"].iloc[-1] / initial_capital - 1) * 100
    daily_returns = equity_df["equity"].pct_change().dropna()
    sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252) if daily_returns.std() > 0 else 0
    rolling_max = equity_df["equity"].cummax()
    drawdown = (equity_df["equity"] - rolling_max) / rolling_max * 100
    max_drawdown = drawdown.min()
    win_trades = _count_winning_trades(trades_df)

    return {
        "equity_curve": equity_df,
        "trades": trades_df,
        "drawdown": drawdown,
        "metrics": {
            "total_return": round(total_return, 2),
            "buy_hold_return": round(buy_hold_return, 2),
            "sharpe_ratio": round(sharpe, 2),
            "max_drawdown": round(max_drawdown, 2),
            "num_trades": len(trades_df),
            "win_rate": win_trades,
            "final_equity": round(equity_df["equity"].iloc[-1], 2),
        },
    }


def _count_winning_trades(trades_df: pd.DataFrame) -> float:
    if trades_df.empty or "type" not in trades_df.columns:
        return 0.0
    buys = trades_df[trades_df["type"] == "BUY"]["price"].values
    sells = trades_df[trades_df["type"] == "SELL"]["price"].values
    pairs = min(len(buys), len(sells))
    if pairs == 0:
        return 0.0
    wins = sum(sells[i] > buys[i] for i in range(pairs))
    return round(wins / pairs * 100, 1)
