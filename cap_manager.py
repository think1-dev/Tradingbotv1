"""
cap_manager.py

Phase 8.1: Centralized cap management for Day and Swing trades.

This module wraps StateManager cap logic and provides a cleaner interface
for the StrategyEngine to check whether new positions can be opened.

Caps (from spec):
- Day: 5 entries per strategy per day, 20 global open positions
- Swing: 5 entries per strategy per week, 15 global open positions
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Tuple, Optional

if TYPE_CHECKING:
    from state_manager import StateManager
    from signals import DaySignal, SwingSignal


class CapManager:
    """
    Centralized cap checking for Day and Swing trades.

    Delegates to StateManager for actual state tracking, but provides
    a signal-centric API that the StrategyEngine can call directly.
    """

    def __init__(self, logger: logging.Logger, state_mgr: "StateManager") -> None:
        self.logger = logger
        self.state_mgr = state_mgr

    def can_open_day(self, sig: "DaySignal") -> Tuple[bool, Optional[str]]:
        """
        Check if a Day trade can be opened for the given signal.

        Returns:
            (True, None) if allowed
            (False, reason) if blocked by caps
        """
        return self.state_mgr.can_open_day(sig.strategy_id, sig.trade_date)

    def can_open_swing(self, sig: "SwingSignal") -> Tuple[bool, Optional[str]]:
        """
        Check if a Swing trade can be opened for the given signal.

        Returns:
            (True, None) if allowed
            (False, reason) if blocked by caps
        """
        # SwingSignal may have trade_date as an attribute set dynamically
        trade_date = getattr(sig, "trade_date", None)
        if trade_date is None:
            from time_utils import now_pt
            trade_date = now_pt().date()
        return self.state_mgr.can_open_swing(sig.strategy_id, trade_date)

    def register_day_entry(self, sig: "DaySignal") -> None:
        """
        Register a Day entry with the state manager.
        Called after a bracket order is successfully placed.
        """
        self.state_mgr.register_day_entry(sig.strategy_id, sig.trade_date)
        self.logger.info(
            "[CAP] Registered Day entry: %s strategy=%s date=%s",
            sig.symbol,
            sig.strategy_id,
            sig.trade_date.isoformat(),
        )

    def register_swing_entry(self, sig: "SwingSignal") -> None:
        """
        Register a Swing entry with the state manager.
        Called after a bracket order is successfully placed.
        """
        trade_date = getattr(sig, "trade_date", None)
        if trade_date is None:
            from time_utils import now_pt
            trade_date = now_pt().date()
        self.state_mgr.register_swing_entry(sig.strategy_id, trade_date)
        self.logger.info(
            "[CAP] Registered Swing entry: %s strategy=%s",
            sig.symbol,
            sig.strategy_id,
        )

    def register_day_exit(self, sig_or_date) -> None:
        """
        Register a Day exit (decrement open count).

        Accepts either a signal object with trade_date attribute, or a date directly.
        """
        if hasattr(sig_or_date, "trade_date"):
            trade_date = sig_or_date.trade_date
        else:
            trade_date = sig_or_date
        self.state_mgr.register_day_exit(trade_date)

    def register_swing_exit(self, sig_or_date) -> None:
        """
        Register a Swing exit (decrement open count).

        Accepts either a signal object with trade_date attribute, or a date directly.
        """
        if hasattr(sig_or_date, "trade_date"):
            trade_date = sig_or_date.trade_date
        else:
            trade_date = sig_or_date
        self.state_mgr.register_swing_exit(trade_date)
