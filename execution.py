"""
execution.py

Phase 6+: OrderExecutor handles bracket order placement to IBKR.

This module receives signals from StrategyEngine and:
- Builds bracket orders using order factory functions
- Links exit legs with OCA groups
- Places orders via ib_insync
- Registers entries with CapManager for state tracking
- Registers pending orders with FillTracker for fill-based cap management

The OrderExecutor does not decide WHEN to trade; it only executes
placement requests from the StrategyEngine.
"""

from __future__ import annotations

import logging
import time
import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING, Tuple, Optional

from ib_insync import IB, Trade, Order


# IBKR error codes indicating "no shortable shares available"
SHORTABLE_ERROR_CODES = {201, 10147, 162}


@dataclass
class PlacementResult:
    """
    Result of a bracket order placement attempt.

    Attributes:
        order_id: Parent order ID if successful, None otherwise
        success: True if order was accepted by IBKR
        rejection_code: IBKR error code if rejected
        rejection_message: IBKR error message if rejected
    """
    order_id: Optional[int] = None
    success: bool = False
    rejection_code: Optional[int] = None
    rejection_message: Optional[str] = None

    @property
    def is_shortable_rejection(self) -> bool:
        """Check if rejection was due to no shortable shares."""
        return self.rejection_code in SHORTABLE_ERROR_CODES

from orders import build_day_bracket, build_swing_bracket, link_bracket
from time_utils import now_pt, is_rth

# Flatten retry configuration
FLATTEN_MAX_RETRIES = 10
FLATTEN_INITIAL_DELAY = 1.0  # seconds
FLATTEN_MAX_DELAY = 30.0  # seconds
FLATTEN_BACKOFF_MULTIPLIER = 2.0

if TYPE_CHECKING:
    from state_manager import StateManager
    from cap_manager import CapManager
    from fill_tracker import FillTracker, FilledPosition
    from signals import DaySignal, SwingSignal
    from conflict_resolver import FlattenInstruction

# Import FilledPosition for runtime use (needed for handle_timed_exit_cancel)
from fill_tracker import FilledPosition


class OrderExecutor:
    """
    Handles bracket order placement for Day and Swing trades.

    Responsibilities:
    - Build bracket orders from signals
    - Place orders with IBKR via ib_insync
    - Register successful entries with CapManager
    - Register pending orders with FillTracker for fill-based caps
    - Log all placement activity
    """

    def __init__(
        self,
        ib: IB,
        logger: logging.Logger,
        state_mgr: "StateManager",
        cap_manager: "CapManager",
        fill_tracker: Optional["FillTracker"] = None,
    ) -> None:
        self.ib = ib
        self.logger = logger
        self.state_mgr = state_mgr
        self.cap_manager = cap_manager
        self.fill_tracker = fill_tracker

        # Error tracking for order rejections
        self._error_lock = threading.Lock()
        self._last_error_code: Optional[int] = None
        self._last_error_message: Optional[str] = None
        self._last_error_order_id: Optional[int] = None

        # Subscribe to IBKR error events
        self.ib.errorEvent += self._on_error

    def _on_error(self, reqId: int, errorCode: int, errorString: str, contract) -> None:
        """
        Handle IBKR error events.

        Captures error code and message for order rejections.
        """
        with self._error_lock:
            self._last_error_order_id = reqId
            self._last_error_code = errorCode
            self._last_error_message = errorString

        # Log all errors for visibility
        self.logger.warning(
            "[EXEC] IBKR error: reqId=%s code=%s msg=%s",
            reqId, errorCode, errorString
        )

    def _get_last_error(self) -> Tuple[Optional[int], Optional[str]]:
        """Get and clear the last error."""
        with self._error_lock:
            code = self._last_error_code
            msg = self._last_error_message
            self._last_error_code = None
            self._last_error_message = None
            self._last_error_order_id = None
            return code, msg

    def _get_next_order_id(self) -> int:
        """
        Request the next valid order ID from IBKR.
        """
        return self.ib.client.getReqId()

    def _place_bracket_orders(
        self, tag: str, bracket, contract
    ) -> Tuple[Optional[Trade], Optional[Trade], Optional[Trade]]:
        """
        Place parent + child orders for a bracket with retry for child orders.

        Assigns order IDs and parent-child relationships, then submits to IBKR.
        If child orders fail, retries with backoff. If all retries fail, cancels parent.
        Returns tuple of (parent_trade, stop_trade, timed_trade) or Nones on failure.
        """
        CHILD_MAX_RETRIES = 3
        CHILD_RETRY_DELAYS = [0.5, 1.0, 2.0]  # seconds

        try:
            # Assign order IDs
            parent_id = self._get_next_order_id()
            stop_id = self._get_next_order_id()
            timed_id = self._get_next_order_id()

            bracket.parent.orderId = parent_id
            bracket.stop.orderId = stop_id
            bracket.stop.parentId = parent_id
            bracket.timed.orderId = timed_id
            bracket.timed.parentId = parent_id

            # Transmit parent first, then children
            bracket.parent.transmit = False
            bracket.stop.transmit = False
            bracket.timed.transmit = True  # Last child transmits the whole group

            # Place parent order first
            parent_trade = self.ib.placeOrder(contract, bracket.parent)
            if parent_trade is None:
                self.logger.error("[EXEC][%s] Parent order placement returned None", tag)
                return None, None, None

            # Place child orders with retry
            stop_trade = None
            timed_trade = None

            for attempt in range(CHILD_MAX_RETRIES):
                try:
                    if stop_trade is None:
                        stop_trade = self.ib.placeOrder(contract, bracket.stop)
                    if timed_trade is None:
                        timed_trade = self.ib.placeOrder(contract, bracket.timed)

                    if stop_trade is not None and timed_trade is not None:
                        break  # Both children placed successfully

                except Exception as child_exc:
                    self.logger.warning(
                        "[EXEC][%s] Child order attempt %d failed: %s",
                        tag, attempt + 1, child_exc,
                    )

                if attempt < CHILD_MAX_RETRIES - 1:
                    time.sleep(CHILD_RETRY_DELAYS[attempt])

            # Check if children were placed successfully
            if stop_trade is None or timed_trade is None:
                self.logger.error(
                    "[EXEC][%s] Child orders failed after %d retries - cancelling parent",
                    tag, CHILD_MAX_RETRIES,
                )
                # Cancel the parent order to avoid unprotected position
                try:
                    self.ib.cancelOrder(bracket.parent)
                    self.logger.info("[EXEC][%s] Parent order cancelled after child failure", tag)
                except Exception as cancel_exc:
                    self.logger.critical(
                        "[EXEC][%s] CRITICAL: Failed to cancel parent after child failure: %s. "
                        "Position may be unprotected! Manual intervention required.",
                        tag, cancel_exc,
                    )
                return None, None, None

            self.logger.info(
                "[EXEC][%s] Bracket submitted: parentId=%s stopId=%s timedId=%s",
                tag,
                parent_id,
                stop_id,
                timed_id,
            )

            return parent_trade, stop_trade, timed_trade

        except Exception as exc:
            self.logger.error(
                "[EXEC][%s] Failed to place bracket: %s", tag, exc
            )
            return None, None, None

    def place_day_bracket(self, sig: "DaySignal") -> PlacementResult:
        """
        Build and place a Day bracket order for the given signal.

        Returns:
            PlacementResult with order_id on success, rejection info on failure.
        """
        tag = f"DAY_{sig.symbol}_{sig.strategy_id}"
        direction = (sig.direction or "LONG").upper()

        self.logger.info(
            "[EXEC][%s] Placing %s bracket: entry=%.4f stop=%.4f shares=%d",
            tag,
            direction,
            sig.entry_price,
            sig.stop_price,
            sig.shares,
        )

        # Clear any previous error before placement
        self._get_last_error()

        # Build the bracket (handles LONG/SHORT direction automatically)
        bracket = build_day_bracket(sig, sig.trade_date)

        # Link exit legs with OCA group
        bracket = link_bracket(bracket, oca_group=tag)

        # Get the contract (may have been attached by StrategyEngine)
        contract = getattr(sig, "contract", None) or bracket.contract

        # Place the orders
        parent_trade, stop_trade, timed_trade = self._place_bracket_orders(
            tag, bracket, contract
        )

        if parent_trade is None:
            # Capture rejection info
            error_code, error_msg = self._get_last_error()
            self.logger.error(
                "[EXEC][%s] Bracket placement failed. code=%s msg=%s",
                tag, error_code, error_msg
            )
            return PlacementResult(
                order_id=None,
                success=False,
                rejection_code=error_code,
                rejection_message=error_msg,
            )

        parent_order_id = bracket.parent.orderId

        # Register entry with cap manager (placement-based tracking)
        self.cap_manager.register_day_entry(sig)

        # Register with fill tracker for fill-based cap management
        if self.fill_tracker is not None:
            self.fill_tracker.register_pending_order(
                order_id=parent_order_id,
                strategy_id=sig.strategy_id,
                symbol=sig.symbol,
                trade_type="DAY",
                trade_date=sig.trade_date,
                side=direction,
                qty=sig.shares,
                stop_order_id=bracket.stop.orderId,
                timed_order_id=bracket.timed.orderId,
            )

        self.logger.info(
            "[EXEC][%s] Day bracket placed successfully.", tag
        )
        return PlacementResult(
            order_id=parent_order_id,
            success=True,
            rejection_code=None,
            rejection_message=None,
        )

    def place_swing_bracket(self, sig: "SwingSignal") -> bool:
        """
        Build and place a Swing bracket order for the given signal.

        Returns True if order was successfully submitted, False otherwise.
        """
        tag = f"SWING_{sig.symbol}_{sig.strategy_id}"

        # Ensure trade_date is set
        trade_date = getattr(sig, "trade_date", None)
        if trade_date is None:
            trade_date = now_pt().date()
            setattr(sig, "trade_date", trade_date)

        self.logger.info(
            "[EXEC][%s] Placing LONG bracket: entry=%.4f stop=%.4f shares=%d",
            tag,
            sig.entry_price,
            sig.stop_price,
            sig.shares,
        )

        # Build the bracket
        bracket = build_swing_bracket(sig, trade_date)

        # Link exit legs with OCA group
        bracket = link_bracket(bracket, oca_group=tag)

        # Get the contract (may have been attached by StrategyEngine)
        contract = getattr(sig, "contract", None) or bracket.contract

        # Place the orders
        parent_trade, stop_trade, timed_trade = self._place_bracket_orders(
            tag, bracket, contract
        )

        if parent_trade is None:
            self.logger.error("[EXEC][%s] Bracket placement failed.", tag)
            return False

        # Register entry with cap manager (placement-based tracking)
        self.cap_manager.register_swing_entry(sig)

        # Register with fill tracker for fill-based cap management
        if self.fill_tracker is not None:
            self.fill_tracker.register_pending_order(
                order_id=bracket.parent.orderId,
                strategy_id=sig.strategy_id,
                symbol=sig.symbol,
                trade_type="SWING",
                trade_date=trade_date,
                side="LONG",  # Swing trades are LONG-only per spec
                qty=sig.shares,
                stop_order_id=bracket.stop.orderId,
                timed_order_id=bracket.timed.orderId,
            )

        self.logger.info(
            "[EXEC][%s] Swing bracket placed successfully.", tag
        )
        return True

    def flatten_position(self, instruction: "FlattenInstruction") -> bool:
        """
        Flatten (close) a position based on ConflictResolver instruction.

        Steps:
        1. Cancel any active child orders (stop, timed)
        2. Place a market order to close the position
        3. Remove the position from FillTracker

        Returns True if successfully initiated, False otherwise.
        """
        tag = f"FLATTEN_{instruction.symbol}_{instruction.kind}_{instruction.side}"

        self.logger.info(
            "[EXEC][%s] Flattening position: qty=%d strategy=%s",
            tag,
            instruction.qty,
            instruction.strategy_id,
        )

        # Step 1: Cancel child orders if they exist
        orders_to_cancel = []
        if instruction.stop_order_id:
            orders_to_cancel.append(instruction.stop_order_id)
        if instruction.timed_order_id:
            orders_to_cancel.append(instruction.timed_order_id)

        for order_id in orders_to_cancel:
            self._cancel_order_by_id(order_id, tag)

        # Step 2: Place a market order to close the position
        # If we're LONG, we SELL to close. If SHORT, we BUY to close.
        close_action = "SELL" if instruction.side == "LONG" else "BUY"

        from orders import make_stock_contract
        contract = make_stock_contract(instruction.symbol)

        close_order = Order()
        close_order.action = close_action
        close_order.totalQuantity = instruction.qty
        close_order.orderType = "MKT"
        close_order.tif = "DAY"
        close_order.orderId = self._get_next_order_id()

        try:
            trade = self.ib.placeOrder(contract, close_order)
            self.logger.info(
                "[EXEC][%s] Close order placed: orderId=%s action=%s qty=%d",
                tag,
                close_order.orderId,
                close_action,
                instruction.qty,
            )
        except Exception as exc:
            self.logger.error(
                "[EXEC][%s] Failed to place close order: %s", tag, exc
            )
            return False

        # Step 3: Remove position from FillTracker
        if self.fill_tracker is not None:
            removed = self.fill_tracker.remove_filled_position(instruction.parent_order_id)
            if removed:
                self.logger.info(
                    "[EXEC][%s] Position removed from FillTracker.", tag
                )
            else:
                self.logger.warning(
                    "[EXEC][%s] Position not found in FillTracker (parent_order_id=%s).",
                    tag,
                    instruction.parent_order_id,
                )

        return True

    def _cancel_order_by_id(self, order_id: int, tag: str) -> bool:
        """
        Cancel an order by its ID.

        Returns True if cancellation was initiated, False if order not found.
        """
        try:
            for trade in self.ib.openTrades():
                if trade.order.orderId == order_id:
                    self.ib.cancelOrder(trade.order)
                    self.logger.info(
                        "[EXEC][%s] Cancelled order: id=%s", tag, order_id
                    )
                    return True

            self.logger.warning(
                "[EXEC][%s] Order not found for cancellation: id=%s", tag, order_id
            )
            return False
        except Exception as exc:
            self.logger.error(
                "[EXEC][%s] Error cancelling order %s: %s", tag, order_id, exc
            )
            return False

    def _check_position_still_open(self, symbol: str, expected_side: str, expected_qty: int) -> Optional[int]:
        """
        Check if a position is still open for the given symbol.

        Returns the current position quantity if still open, None if closed.
        """
        try:
            positions = self.ib.positions()
            for pos in positions:
                if pos.contract.symbol == symbol:
                    current_qty = int(pos.position)
                    # LONG position has positive qty, SHORT has negative
                    if expected_side == "LONG" and current_qty > 0:
                        return current_qty
                    elif expected_side == "SHORT" and current_qty < 0:
                        return abs(current_qty)
            return None  # Position closed
        except Exception as exc:
            self.logger.warning(
                "[EXEC] Error checking position for %s: %s", symbol, exc
            )
            return None  # Assume closed on error

    def flatten_position_with_retry(self, instruction: "FlattenInstruction") -> tuple:
        """
        Flatten a position with retry on failure.

        If the initial flatten fails or the position remains open,
        keeps retrying with exponential backoff until success.

        If EOD is reached before successful flatten, schedules the position
        for retry at next market open.

        Args:
            instruction: FlattenInstruction from ConflictResolver

        Returns:
            Tuple of (flatten_success: bool, cleanup_success: bool)
            - flatten_success: True if position was successfully flattened
            - cleanup_success: True if position was removed from pending_flattens
            Both False if position was scheduled for next day.
        """
        tag = f"FLATTEN_RETRY_{instruction.symbol}_{instruction.kind}"
        attempt = 0
        delay = FLATTEN_INITIAL_DELAY

        while attempt < FLATTEN_MAX_RETRIES:
            # Check if we're still within RTH
            now = now_pt()
            if not is_rth(now):
                # EOD reached - schedule for next market open
                self.logger.warning(
                    "[EXEC][%s] EOD reached before flatten complete - scheduling for next market open",
                    tag,
                )
                self.state_mgr.pending_flattens.add_pending_flatten(
                    symbol=instruction.symbol,
                    side=instruction.side,
                    qty=instruction.qty,
                    kind=instruction.kind,
                    strategy_id=instruction.strategy_id,
                    reason="eod_reached_during_flatten",
                    parent_order_id=instruction.parent_order_id,
                )
                return (False, False)

            attempt += 1
            self.logger.info(
                "[EXEC][%s] Flatten attempt %d/%d",
                tag, attempt, FLATTEN_MAX_RETRIES,
            )

            # Try to flatten
            success = self.flatten_position(instruction)

            if success:
                # Wait a moment for order to be processed
                self.ib.sleep(1.0)

                # Verify position is actually closed
                remaining = self._check_position_still_open(
                    instruction.symbol,
                    instruction.side,
                    instruction.qty,
                )

                if remaining is None or remaining == 0:
                    self.logger.info(
                        "[EXEC][%s] Position successfully flattened on attempt %d",
                        tag, attempt,
                    )
                    # Remove from pending flattens if it was there
                    cleanup_success = self.state_mgr.pending_flattens.remove_pending_flatten(
                        instruction.symbol, instruction.strategy_id
                    )
                    return (True, cleanup_success)
                else:
                    self.logger.warning(
                        "[EXEC][%s] Position still open after flatten (remaining=%d), will retry",
                        tag, remaining,
                    )
                    # Update instruction qty for partial fill handling
                    instruction.qty = remaining
            else:
                self.logger.warning(
                    "[EXEC][%s] Flatten attempt %d failed, will retry",
                    tag, attempt,
                )

            # Wait before retry with exponential backoff
            if attempt < FLATTEN_MAX_RETRIES:
                self.logger.info(
                    "[EXEC][%s] Waiting %.1f seconds before retry...",
                    tag, delay,
                )
                time.sleep(delay)
                delay = min(delay * FLATTEN_BACKOFF_MULTIPLIER, FLATTEN_MAX_DELAY)

        # All retries exhausted - check if still RTH
        now = now_pt()
        if is_rth(now):
            # Still in RTH but all retries exhausted - schedule for EOD/next day
            self.logger.error(
                "[EXEC][%s] All %d flatten attempts exhausted - scheduling for next market open",
                tag, FLATTEN_MAX_RETRIES,
            )
            self.state_mgr.pending_flattens.add_pending_flatten(
                symbol=instruction.symbol,
                side=instruction.side,
                qty=instruction.qty,
                kind=instruction.kind,
                strategy_id=instruction.strategy_id,
                reason="all_retries_exhausted",
                parent_order_id=instruction.parent_order_id,
            )
        else:
            # Already past RTH
            self.logger.error(
                "[EXEC][%s] All %d flatten attempts exhausted and past RTH - already scheduled",
                tag, FLATTEN_MAX_RETRIES,
            )

        return (False, False)

    def handle_timed_exit_cancel(self, position: "FilledPosition") -> None:
        """
        Handle a cancelled timed exit MKT order.

        Called by FillTracker when a timed exit is cancelled.
        Creates a FlattenInstruction and attempts to flatten with retry.

        Args:
            position: The FilledPosition whose timed exit was cancelled
        """
        from conflict_resolver import FlattenInstruction

        self.logger.warning(
            "[EXEC] Timed exit cancelled for %s %s %s - initiating flatten retry",
            position.symbol, position.kind, position.side,
        )

        instruction = FlattenInstruction(
            symbol=position.symbol,
            side=position.side,
            kind=position.kind,
            strategy_id=position.strategy_id,
            qty=position.qty,
            parent_order_id=position.parent_order_id,
            stop_order_id=position.stop_order_id,
            timed_order_id=position.timed_order_id,
        )

        # Attempt to flatten (will retry with backoff, schedule for next day if EOD)
        flatten_success, cleanup_success = self.flatten_position_with_retry(instruction)
        if flatten_success:
            self.logger.info(
                "[EXEC] Successfully flattened after timed exit cancel: %s %s (cleanup=%s)",
                position.symbol, position.strategy_id, cleanup_success,
            )
            # Remove from fill tracker since position is closed
            if self.fill_tracker is not None:
                self.fill_tracker.remove_filled_position(position.parent_order_id)

    def process_pending_flattens(self) -> None:
        """
        Process all pending flattens at market open.

        Called at 6:30 PT to retry flattening positions that couldn't be
        closed the previous day.
        """
        pending = self.state_mgr.pending_flattens.get_all_pending()
        if not pending:
            self.logger.info("[EXEC] No pending flattens to process at market open.")
            return

        self.logger.info(
            "[EXEC] Processing %d pending flattens at market open.", len(pending)
        )

        for pos in pending:
            symbol = pos["symbol"]
            side = pos["side"]
            qty = pos["qty"]
            kind = pos["kind"]
            strategy_id = pos["strategy_id"]
            parent_order_id = pos.get("parent_order_id", 0)

            self.logger.info(
                "[EXEC] Retrying pending flatten: %s %s %s qty=%d parent_order_id=%d",
                symbol, kind, side, qty, parent_order_id,
            )

            # Check if position is still open
            remaining = self._check_position_still_open(symbol, side, qty)
            if remaining is None or remaining == 0:
                self.logger.info(
                    "[EXEC] Position %s %s already closed - removing from pending",
                    symbol, strategy_id,
                )
                self.state_mgr.pending_flattens.remove_pending_flatten(symbol, strategy_id)
                # Also remove from fill_tracker if we have the parent_order_id
                if parent_order_id != 0 and self.fill_tracker is not None:
                    self.fill_tracker.remove_filled_position(parent_order_id)
                continue

            # Create a FlattenInstruction-like object
            from conflict_resolver import FlattenInstruction
            instruction = FlattenInstruction(
                symbol=symbol,
                side=side,
                kind=kind,
                strategy_id=strategy_id,
                qty=remaining,
                parent_order_id=parent_order_id,
                stop_order_id=None,
                timed_order_id=None,
            )

            # Attempt to flatten (will retry with backoff)
            flatten_success, cleanup_success = self.flatten_position_with_retry(instruction)
            if flatten_success:
                self.logger.info(
                    "[EXEC] Successfully flattened pending position %s %s (cleanup=%s)",
                    symbol, strategy_id, cleanup_success,
                )
                # Remove from fill_tracker if we have the parent_order_id
                if parent_order_id != 0 and self.fill_tracker is not None:
                    self.fill_tracker.remove_filled_position(parent_order_id)
