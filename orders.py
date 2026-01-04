"""
orders.py

Phase 5+ : Pure order-factory functions that build ib_insync
Contracts + Orders for Day and Swing brackets.

This module knows nothing about IB connections or fills; it only
constructs bracket "shapes" from signal objects.

Now integrates with market_calendar for holiday and early close awareness.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, date, time
from typing import Optional

from ib_insync import Contract, Stock, Order, IB

import market_calendar as mc


# ---------------------------------------------------------------------- #
# Dataclasses to hold bracket "shapes"
# ---------------------------------------------------------------------- #


@dataclass
class DayBracket:
    contract: Contract
    parent: Order
    stop: Order
    timed: Order


@dataclass
class SwingBracket:
    contract: Contract
    parent: Order
    stop: Order
    timed: Order


# ---------------------------------------------------------------------- #
# Contract helpers
# ---------------------------------------------------------------------- #


def make_stock_contract(symbol: str) -> Stock:
    """
    Per spec we trade only US stocks for now, via SMART routing.
    """
    contract = Stock(symbol, "SMART", "USD")
    # We let ib_insync qualify it; no conId here.
    return contract


# ---------------------------------------------------------------------- #
# Time helpers for GTD / GAT fields
# ---------------------------------------------------------------------- #


def _format_pt_datetime(dt: datetime) -> str:
    """
    Format a datetime into IB's 'YYYYMMDD HH:MM:SS' string.
    """
    return dt.strftime("%Y%m%d %H:%M:%S")


def _format_pt_gtd(trade_date: date, hour: int, minute: int) -> str:
    """
    Format a PT date + (hour:minute) into IB's 'YYYYMMDD HH:MM:SS' string
    for goodTillDate.

    For Day stops, spec wants ~12:55 PT on the same PT date.
    """
    dt = datetime(
        trade_date.year, trade_date.month, trade_date.day,
        hour, minute, 0
    )
    return dt.strftime("%Y%m%d %H:%M:%S")


def _format_pt_gat(trade_date: date, hour: int, minute: int) -> str:
    """
    Format a PT date + (hour:minute) into IB's 'YYYYMMDD HH:MM:SS' string
    for goodAfterTime (GAT).

    For Day timed exits, spec wants ~12:58 PT on the same PT date.
    """
    dt = datetime(
        trade_date.year, trade_date.month, trade_date.day,
        hour, minute, 0
    )
    return dt.strftime("%Y%m%d %H:%M:%S")


def _get_day_stop_gtd(trade_date: date) -> str:
    """
    Get GTD string for day stop orders, accounting for early close.

    Normal days: 12:55 PT
    Early close days: 9:55 PT
    """
    stop_time = mc.get_day_stop_time_pt(trade_date)
    dt = datetime(
        trade_date.year, trade_date.month, trade_date.day,
        stop_time.hour, stop_time.minute, 0
    )
    return _format_pt_datetime(dt)


def _get_day_exit_gat(trade_date: date) -> str:
    """
    Get GAT string for day timed exit orders, accounting for early close.

    Normal days: 12:58 PT
    Early close days: 9:58 PT
    """
    exit_time = mc.get_day_exit_time_pt(trade_date)
    dt = datetime(
        trade_date.year, trade_date.month, trade_date.day,
        exit_time.hour, exit_time.minute, 0
    )
    return _format_pt_datetime(dt)


# ---------------------------------------------------------------------- #
# Day brackets
# ---------------------------------------------------------------------- #


def build_day_bracket(
    signal,
    trade_date: date,
) -> DayBracket:
    """
    Build a Day bracket for a single DaySignal, for the given PT trade_date.

    Supports both LONG and SHORT directions:

    LONG:
      - Parent: BUY LMT @ Entry
      - Stop child: SELL STP @ Stop (below entry), GTD to ~12:55 PT
      - Timed exit child: SELL MKT, DAY, GAT ~12:58 PT

    SHORT:
      - Parent: SELL LMT @ Entry (short entry)
      - Stop child: BUY STP @ Stop (above entry), GTD to ~12:55 PT
      - Timed exit child: BUY MKT, DAY, GAT ~12:58 PT

    The DaySignal is assumed to expose:
      - symbol
      - shares
      - entry_price
      - stop_price
      - direction ("LONG" or "SHORT")
    """
    symbol = signal.symbol
    qty = signal.shares
    entry = signal.entry_price
    stop_price = signal.stop_price
    direction = getattr(signal, "direction", "LONG").upper()

    contract = make_stock_contract(symbol)

    # Determine actions based on direction
    if direction == "SHORT":
        entry_action = "SELL"
        exit_action = "BUY"
    else:  # LONG (default)
        entry_action = "BUY"
        exit_action = "SELL"

    # ----- Parent entry -----
    parent = Order()
    parent.action = entry_action
    parent.totalQuantity = qty
    parent.orderType = "LMT"
    parent.lmtPrice = entry
    parent.tif = "DAY"

    # ----- Stop child -----
    # GTD adjusted for early close days (9:55 PT vs 12:55 PT)
    stop = Order()
    stop.action = exit_action
    stop.totalQuantity = qty
    stop.orderType = "STP"
    stop.auxPrice = stop_price
    stop.tif = "GTD"
    stop.goodTillDate = _get_day_stop_gtd(trade_date)

    # ----- Timed exit child -----
    # GAT adjusted for early close days (9:58 PT vs 12:58 PT)
    timed = Order()
    timed.action = exit_action
    timed.totalQuantity = qty
    timed.orderType = "MKT"
    timed.tif = "DAY"
    timed.goodAfterTime = _get_day_exit_gat(trade_date)

    return DayBracket(
        contract=contract,
        parent=parent,
        stop=stop,
        timed=timed,
    )


# ---------------------------------------------------------------------- #
# Swing brackets
# ---------------------------------------------------------------------- #

from datetime import timedelta  # placed here to keep top imports minimal


def _get_swing_exit_times(trade_date: date) -> tuple:
    """
    Get the GTD and GAT times for swing exit orders.

    Returns (week_ending_day, stop_gtd_string, timed_gat_string)

    Swing order timing:
    - Limit entry: GTD Friday close (stays active all week)
    - Stop: GTD Friday close (protects through the trading week)
    - Timed exit GAT: Next Monday @ 6:30 AM PT (exits at new week open)
    """
    # Week-ending day for stop/limit GTD (Friday, or Thursday if Friday is holiday)
    week_end = mc.get_week_ending_day(trade_date)
    stop_time = mc.get_day_stop_time_pt(week_end)  # 5 min before close

    # Stop/Limit GTD: week-ending day @ (close - 5 min)
    stop_dt = datetime(
        week_end.year, week_end.month, week_end.day,
        stop_time.hour, stop_time.minute, 0
    )

    # Timed exit GAT: next Monday @ market open (6:30 PT)
    current_monday = trade_date - timedelta(days=trade_date.weekday())
    next_monday = current_monday + timedelta(days=7)

    # If next Monday is a holiday, use the next trading day
    if not mc.is_trading_day(next_monday):
        next_monday = mc.get_next_trading_day(next_monday)

    timed_dt = datetime(
        next_monday.year, next_monday.month, next_monday.day,
        6, 30, 0
    )

    return week_end, _format_pt_datetime(stop_dt), _format_pt_datetime(timed_dt)


def build_swing_bracket(
    signal,
    trade_date: date,
    *,
    market_entry: Optional[bool] = None,
) -> SwingBracket:
    """
    Build a Swing bracket for a single SwingSignal, for the given PT trade_date.

    Normal (non-gap) behavior per spec, LONG-only:
      - Parent: BUY LMT @ Entry, GTD to Friday close (stays active all week).
      - Stop child: SELL STP @ Stop, GTD to Friday close.
      - Timed exit child: SELL MKT, DAY, GAT next Monday 06:30 PT.

    Swing positions exit at the beginning of the next trading week.

    `market_entry` is kept for gap-at-open logic and re-entry (MKT entry).
    In the normal path, pass None (=> LIMIT entry for all strategies).

    Now uses market_calendar for:
    - Week-ending day detection (Friday, or Thursday if Friday is holiday)
    - Next Monday detection (or next trading day if Monday is holiday)
    - Early close time adjustment (10:00 PT close on early close days)
    """
    symbol = signal.symbol
    qty = signal.shares
    entry = signal.entry_price
    stop_price = signal.stop_price

    contract = make_stock_contract(symbol)

    # Determine whether to use a market entry (e.g. gap, re-entry) or limit
    use_mkt = False
    if market_entry is True:
        use_mkt = True
    elif market_entry is None:
        # Default to LIMIT for all swing entries per spec
        use_mkt = False

    # Get week-ending day and exit times from market calendar
    week_end, stop_gtd, timed_gat = _get_swing_exit_times(trade_date)

    # ----- Parent entry (LONG-only swings) -----
    parent = Order()
    parent.action = "BUY"
    parent.totalQuantity = qty

    if use_mkt:
        parent.orderType = "MKT"
        parent.tif = "DAY"
    else:
        parent.orderType = "LMT"
        parent.lmtPrice = entry
        parent.tif = "GTD"
        parent.goodTillDate = stop_gtd  # Same GTD as stop (Friday close)

    # ----- Stop child -----
    # GTD to week-ending day, adjusted for early close
    stop = Order()
    stop.action = "SELL"
    stop.totalQuantity = qty
    stop.orderType = "STP"
    stop.auxPrice = stop_price
    stop.tif = "GTD"
    stop.goodTillDate = stop_gtd

    # ----- Timed exit child -----
    # GAT at market open on week-ending day
    timed = Order()
    timed.action = "SELL"
    timed.totalQuantity = qty
    timed.orderType = "MKT"
    timed.tif = "DAY"
    timed.goodAfterTime = timed_gat

    return SwingBracket(
        contract=contract,
        parent=parent,
        stop=stop,
        timed=timed,
    )


# ---------------------------------------------------------------------- #
# Bracket linking (OCA only; IDs are handled in execution layer)
# ---------------------------------------------------------------------- #


def link_bracket(
    bracket: DayBracket | SwingBracket, *, oca_group: str
) -> DayBracket | SwingBracket:
    """
    Attach OCA group tags to the exit legs of a Day or Swing bracket.

    IMPORTANT:
    - This function no longer assigns orderIds or parentIds.
    - Order ID sequencing and parent-child wiring are handled by the
      execution layer (OrderExecutor and any test harnesses) when the
      orders are actually placed with IBKR.

    This avoids manual management of TWS orderId sequences and the
    associated IB errors such as:
      * 105: "Order being modified does not match original order"
      * 10341: "Parent order id cannot be modified"
      * 201: "Order rejected - reason:Parent order is being cancelled."
      * 104: "Cannot modify a filled order."

    Returns the same bracket object for convenience.
    """
    # Parent usually does not need an OCA group, but it's harmless if set.
    # The key is that both exit legs share the same OCA group so that when
    # one fills or is triggered, the other is cancelled automatically.
    bracket.stop.ocaGroup = oca_group
    bracket.timed.ocaGroup = oca_group
    return bracket
