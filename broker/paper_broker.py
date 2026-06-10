"""
Paper broker — simulates order execution with no real money.
All state is in-memory; persisted to a JSON file for the session.
"""

import json
import os
from datetime import datetime
from pathlib import Path

STATE_FILE = Path(__file__).parent.parent / "paper_portfolio.json"


def _load() -> dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {"cash": 10_000.0, "positions": {}, "history": []}


def _save(state: dict):
    STATE_FILE.write_text(json.dumps(state, indent=2, default=str))


def get_portfolio() -> dict:
    return _load()


def place_order(ticker: str, action: str, price: float, quantity: float | None = None) -> dict:
    """action: 'buy' | 'sell'. quantity=None → use all available cash / sell full position."""
    state = _load()
    commission_rate = 0.001
    ts = datetime.now().isoformat()

    if action == "buy":
        budget = state["cash"]
        if budget <= 0:
            return {"ok": False, "error": "Insufficient cash"}
        qty = quantity or (budget * (1 - commission_rate)) / price
        cost = qty * price * (1 + commission_rate)
        if cost > state["cash"]:
            return {"ok": False, "error": "Not enough cash"}
        state["cash"] -= cost
        state["positions"][ticker] = state["positions"].get(ticker, 0) + qty
        order = {"date": ts, "ticker": ticker, "action": "BUY", "price": price, "qty": round(qty, 4), "cost": round(cost, 2)}

    elif action == "sell":
        held = state["positions"].get(ticker, 0)
        if held <= 0:
            return {"ok": False, "error": f"No position in {ticker}"}
        qty = quantity or held
        proceeds = qty * price * (1 - commission_rate)
        state["cash"] += proceeds
        state["positions"][ticker] = held - qty
        if state["positions"][ticker] <= 1e-6:
            del state["positions"][ticker]
        order = {"date": ts, "ticker": ticker, "action": "SELL", "price": price, "qty": round(qty, 4), "proceeds": round(proceeds, 2)}

    else:
        return {"ok": False, "error": f"Unknown action: {action}"}

    state["history"].append(order)
    _save(state)
    return {"ok": True, "order": order}


def reset_portfolio(initial_cash: float = 10_000.0):
    _save({"cash": initial_cash, "positions": {}, "history": []})
