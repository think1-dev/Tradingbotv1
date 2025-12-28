"""
executor.py

Phase 7: TradeExecutor turns StrategyEngine triggers into IBKR
bracket orders.

- Uses StateManager to enforce caps again at submit-time.
- Uses orders.build_* + orders.link_bracket() to create full
  Day / Swing brackets.
- Respects a boolean flag enable_trading:
    * False -> DRYRUN: only logs what it WOULD do.
    * True  -> actually calls ib.placeOrder().

This module never decides WHEN to trade; it only executes requests
from StrategyEngine.
"""

from __future__ import annotations

from typing import Tuple

from ib_insync import IB, Trade

from state_manager import StateManager
from signals import DaySignal, SwingSignal
from orders import build_day_bracket, build_swing_bracket, link_bracket


class TradeExecutor:
    def __init__(self, ib: IB, logger, state_mgr: StateManager, enable_trading: bool) -> None:
        self.ib = ib
        self.logger = logger
        self.state_mgr = state_mgr
        self.enable_trading = enable_trading

    # --- internal helper ---

    def _place_bracket(self, tag: str, bracket) -> Tuple[Trade, Trade, Trade]:
        """
        Actually submit the parent + stop + timed orders to IBKR.
        Assumes link_bracket() has already assigned orderIds.
        """
        parent_trade = self.ib.placeOrder(bracket.contract, bracket.parent)
        stop_trade = self.ib.placeOrder(bracket.contract, bracket.stop)
        timed_trade = self.ib.placeOrder(bracket.contract, bracket.timed)

        self.logger.info(
            "[TRADE][%s] Submitted bracket: parentId=%s stopId=%s timedId=%s",
            tag,
            bracket.parent.orderId,
            bracket.stop.orderId,
            bracket.timed.orderId,
        )
        return parent_trade, stop_trade, timed_trade

    # --- DAY entries ---

    def submit_day_entry(self, sig: DaySignal, direction: str, trigger_price: float) -> None:
        """
        Execute a Day entry for the given signal, if caps allow.

        direction: "LONG" or "SHORT"
        trigger_price: bid/ask level that caused the trigger (for logging).
        """
        allowed, reason = self.state_mgr.can_open_day(sig.strategy_id, sig.trade_date)
        if not allowed:
            self.logger.info(
                "[TRADE][DAY][SKIP_CAP] %s strategy=%s trigger=%.4f cap_block=%s",
                sig.symbol,
                sig.strategy_id,
                trigger_price,
                reason,
            )
            return

        tag = f"DAY_{sig.symbol}_{sig.strategy_id}"
        bracket = build_day_bracket(sig, sig.trade_date)

        if not self.enable_trading:
            self.logger.info(
                "[TRADE][DAY][DRYRUN] Would place %s bracket for %s strategy=%s "
                "shares=%s entry=%s stop=%s trigger=%.4f tag=%s",
                direction,
                sig.symbol,
                sig.strategy_id,
                sig.shares,
                sig.entry_price,
                sig.stop_price,
                trigger_price,
                tag,
            )
            return

        bracket = link_bracket(bracket, oca_group=tag)
        self._place_bracket(tag, bracket)
        # For now we count on submit; later we can refine to first-fill based.
        self.state_mgr.register_day_entry(sig.strategy_id, sig.trade_date)

    # --- SWING entries ---

    def submit_swing_entry(self, sig: SwingSignal, label: str, trigger_price: float) -> None:
        """
        Execute a Swing entry for the given signal, if caps allow.

        label: "MOMO" or "PULLBACK" (for logging).
        trigger_price: ask level that caused the trigger.
        """
        allowed, reason = self.state_mgr.can_open_swing(sig.strategy_id, sig.trade_date)
        if not allowed:
            self.logger.info(
                "[TRADE][SWING][%s][SKIP_CAP] %s strategy=%s trigger=%.4f cap_block=%s",
                label,
                sig.symbol,
                sig.strategy_id,
                trigger_price,
                reason,
            )
            return

        tag = f"SWING_{sig.symbol}_{sig.strategy_id}"
        bracket = build_swing_bracket(sig, sig.trade_date)

        if not self.enable_trading:
            self.logger.info(
                "[TRADE][SWING][%s][DRYRUN] Would place bracket for %s strategy=%s "
                "shares=%s entry=%s stop=%s trigger=%.4f tag=%s",
                label,
                sig.symbol,
                sig.strategy_id,
                sig.shares,
                sig.entry_price,
                sig.stop_price,
                trigger_price,
                tag,
            )
            return

        bracket = link_bracket(bracket, oca_group=tag)
        self._place_bracket(tag, bracket)
        self.state_mgr.register_swing_entry(sig.strategy_id, sig.trade_date)
