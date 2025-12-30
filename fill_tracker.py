"""
fill_tracker.py

Fill-based cap management for Day and Swing trades.

This module tracks:
- Pending orders by strategy (order ID → strategy mapping)
- Actual fills per strategy (not just placements)
- Cancels unfilled orders when strategy cap is reached
- Filled positions for conflict resolution (symbol/side/kind)

Caps (from spec):
- Day: 5 fills per strategy per day
- Swing: 5 fills per strategy per week
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import TYPE_CHECKING, Dict, List, Optional, Set

from time_utils import now_pt

if TYPE_CHECKING:
    from ib_insync import IB, Trade


# Cap limits
DAY_STRATEGY_FILL_CAP = 5
SWING_STRATEGY_FILL_CAP = 5


@dataclass
class PendingOrder:
    """Tracks a pending (unfilled) order."""
    order_id: int
    strategy_id: str
    symbol: str
    trade_type: str  # "DAY" or "SWING"
    trade_date: date
    side: str  # "LONG" or "SHORT"
    qty: int  # Total ordered quantity
    stop_order_id: Optional[int] = None
    timed_order_id: Optional[int] = None
    filled_qty: int = 0  # Cumulative filled quantity for partial fill tracking
    avg_fill_price: float = 0.0  # Weighted average fill price


@dataclass
class FilledPosition:
    """
    Tracks an actual filled position for conflict resolution.

    This represents a position that has been entered (parent order filled)
    and is still open (stop/timed exit not yet triggered).
    """
    symbol: str
    side: str  # "LONG" or "SHORT"
    kind: str  # "DAY" or "SWING"
    strategy_id: str
    qty: int
    fill_price: float
    fill_time: datetime
    parent_order_id: int
    stop_order_id: Optional[int] = None
    timed_order_id: Optional[int] = None
    trade_date: date = None


@dataclass
class StrategyFillState:
    """Tracks fill state for a single strategy on a given date/week."""
    strategy_id: str
    fill_count: int = 0
    pending_order_ids: Set[int] = field(default_factory=set)
    filled_symbols: List[str] = field(default_factory=list)


class FillTracker:
    """
    Tracks order fills and enforces strategy caps.

    When an order fills:
    1. Increment the fill counter for that strategy
    2. If fills >= cap, cancel all remaining pending orders for that strategy
    3. Track the filled position for conflict resolution

    This ensures we never exceed 5 fills per strategy, even if
    multiple orders were placed before any filled.

    Also provides position tracking for ConflictResolver:
    - get_filled_positions_by_symbol(symbol) -> List[FilledPosition]
    - remove_filled_position(parent_order_id) -> removes when position exits

    Callbacks:
    - on_day_exit_callback: Called when a DAY trade exits (for ReentryManager)
    - on_day_cancel_callback: Called when a DAY trade is cancelled
    - on_timed_exit_cancel_callback: Called when a timed exit MKT order is cancelled
    """

    def __init__(self, ib: "IB", logger: logging.Logger, cap_manager=None, state_mgr=None) -> None:
        self.ib = ib
        self.logger = logger
        self.cap_manager = cap_manager
        self.state_mgr = state_mgr

        # order_id -> PendingOrder
        self.pending_orders: Dict[int, PendingOrder] = {}

        # (trade_type, date_key, strategy_id) -> StrategyFillState
        # date_key is date.isoformat() for DAY, monday.isoformat() for SWING
        self.strategy_states: Dict[tuple, StrategyFillState] = {}

        # Filled positions for conflict resolution
        # parent_order_id -> FilledPosition
        self.filled_positions: Dict[int, FilledPosition] = {}

        # Track exit order IDs to parent order IDs (for position removal on exit fill)
        # exit_order_id -> parent_order_id
        self._exit_to_parent: Dict[int, int] = {}

        # Track which exit orders are timed exits (for cancelled timed exit handling)
        # timed_order_id -> parent_order_id
        self._timed_exit_to_parent: Dict[int, int] = {}

        # Callbacks for ReentryManager integration
        self._on_day_exit_callback = None
        self._on_day_cancel_callback = None
        self._on_timed_exit_cancel_callback = None
        self._on_reentry_fill_callback = None

        # Track re-entry orders that need slot conversion on fill
        # order_id -> trade_date
        self._pending_reentry_conversions: Dict[int, date] = {}

        # Gap order completion callback (set by GapManager)
        self._on_gap_fill_callback = None

        # Lock to protect concurrent access to shared dictionaries from event handlers
        self._lock = threading.Lock()

        # Subscribe to fill events
        self._setup_event_handlers()

    def set_day_exit_callback(self, callback) -> None:
        """
        Register a callback for when Day trades exit.

        Callback signature: callback(parent_order_id: int) -> None
        """
        self._on_day_exit_callback = callback

    def set_day_cancel_callback(self, callback) -> None:
        """
        Register a callback for when Day trades are cancelled.

        Callback signature: callback(parent_order_id: int) -> None
        """
        self._on_day_cancel_callback = callback

    def set_timed_exit_cancel_callback(self, callback) -> None:
        """
        Register a callback for when timed exit MKT orders are cancelled.

        Called when a Day timed exit is cancelled, allowing for flatten retry.
        Callback signature: callback(position: FilledPosition) -> None
        """
        self._on_timed_exit_cancel_callback = callback

    def set_reentry_fill_callback(self, callback) -> None:
        """
        Register a callback for when re-entry orders fill.

        Called when a re-entry bracket order fills, so slot conversion can happen.
        Callback signature: callback(order_id: int, trade_date: date) -> None
        """
        self._on_reentry_fill_callback = callback

    def set_gap_fill_callback(self, callback) -> None:
        """
        Register a callback for when gap MKT orders complete fill.

        Called when a gap MKT order is completely filled, so stop/timed can be placed.
        Callback signature: callback(order_id: int, fill_price: float, fill_qty: int) -> bool
        """
        self._on_gap_fill_callback = callback

    def register_pending_reentry_conversion(self, order_id: int, trade_date: date) -> None:
        """
        Register a re-entry order that needs slot conversion when it fills.

        Called by ReentryManager after placing a re-entry bracket.
        """
        with self._lock:
            self._pending_reentry_conversions[order_id] = trade_date
            self.logger.info(
                "[FILL_TRACKER] Registered pending re-entry conversion for order %d",
                order_id,
            )

    def register_gap_bracket_completion(
        self,
        parent_order_id: int,
        stop_order_id: int,
        timed_order_id: int,
        symbol: str,
        strategy_id: str,
        signal_type: str,
        direction: str,
        shares: int,
    ) -> None:
        """
        Register a completed gap bracket after stop/timed orders are placed.

        Called by GapManager.complete_gap_bracket() after placing stop/timed.
        This updates the filled position with the new exit order IDs.
        """
        with self._lock:
            if parent_order_id in self.filled_positions:
                pos = self.filled_positions[parent_order_id]
                pos.stop_order_id = stop_order_id
                pos.timed_order_id = timed_order_id

                # Track exit orders for position removal on fill
                self._exit_to_parent[stop_order_id] = parent_order_id
                self._exit_to_parent[timed_order_id] = parent_order_id
                self._timed_exit_to_parent[timed_order_id] = parent_order_id

                self.logger.info(
                    "[FILL_TRACKER] Registered gap bracket completion: %s %s "
                    "parent=%d stop=%d timed=%d",
                    symbol, strategy_id, parent_order_id, stop_order_id, timed_order_id,
                )
            else:
                self.logger.warning(
                    "[FILL_TRACKER] Gap bracket completion for unknown parent %d",
                    parent_order_id,
                )

    def _setup_event_handlers(self) -> None:
        """Subscribe to IBKR fill/status events."""
        self.ib.orderStatusEvent += self._on_order_status
        self.ib.execDetailsEvent += self._on_exec_details
        self.logger.info("[FILL_TRACKER] Event handlers registered.")

    def _get_state_key(self, trade_type: str, trade_date: date, strategy_id: str) -> tuple:
        """Generate a unique key for strategy state lookup."""
        if trade_type == "SWING":
            # Use Monday of the week for swing trades
            monday = trade_date - timedelta(days=trade_date.weekday())
            date_key = monday.isoformat()
        else:
            date_key = trade_date.isoformat()
        return (trade_type, date_key, strategy_id)

    def _get_or_create_state(self, trade_type: str, trade_date: date, strategy_id: str) -> StrategyFillState:
        """Get or create strategy fill state."""
        key = self._get_state_key(trade_type, trade_date, strategy_id)
        if key not in self.strategy_states:
            self.strategy_states[key] = StrategyFillState(strategy_id=strategy_id)
        return self.strategy_states[key]

    def register_pending_order(
        self,
        order_id: int,
        strategy_id: str,
        symbol: str,
        trade_type: str,
        trade_date: date,
        side: str = "LONG",
        qty: int = 0,
        stop_order_id: Optional[int] = None,
        timed_order_id: Optional[int] = None,
    ) -> None:
        """
        Register a newly placed order as pending.

        Called by OrderExecutor after placing an order.
        Only register PARENT orders (not stop/timed children).

        Args:
            order_id: Parent order ID
            strategy_id: Strategy identifier
            symbol: Stock symbol
            trade_type: "DAY" or "SWING"
            trade_date: Trade date
            side: "LONG" or "SHORT"
            qty: Number of shares
            stop_order_id: Stop child order ID (for position tracking)
            timed_order_id: Timed exit child order ID (for position tracking)
        """
        with self._lock:
            pending = PendingOrder(
                order_id=order_id,
                strategy_id=strategy_id,
                symbol=symbol.upper(),
                trade_type=trade_type,
                trade_date=trade_date,
                side=side.upper(),
                qty=qty,
                stop_order_id=stop_order_id,
                timed_order_id=timed_order_id,
            )
            self.pending_orders[order_id] = pending

            # Add to strategy state
            state = self._get_or_create_state(trade_type, trade_date, strategy_id)
            state.pending_order_ids.add(order_id)

            self.logger.info(
                "[FILL_TRACKER] Registered pending order: id=%s symbol=%s strategy=%s type=%s side=%s qty=%d",
                order_id, symbol, strategy_id, trade_type, side, qty
            )

    def get_fill_count(self, trade_type: str, trade_date: date, strategy_id: str) -> int:
        """Get current fill count for a strategy."""
        state = self._get_or_create_state(trade_type, trade_date, strategy_id)
        return state.fill_count

    def can_place_order(self, trade_type: str, trade_date: date, strategy_id: str) -> tuple[bool, Optional[str]]:
        """
        Check if we can place another order for this strategy.

        This is a PRE-PLACEMENT check. Even if this returns True,
        the order might get cancelled later if the cap is reached
        before it fills.
        """
        state = self._get_or_create_state(trade_type, trade_date, strategy_id)
        cap = DAY_STRATEGY_FILL_CAP if trade_type == "DAY" else SWING_STRATEGY_FILL_CAP

        if state.fill_count >= cap:
            return False, f"strategy fill cap reached ({state.fill_count}/{cap})"

        return True, None

    def _on_order_status(self, trade: "Trade") -> None:
        """Handle order status updates from IBKR."""
        with self._lock:
            order_id = trade.order.orderId
            status = trade.orderStatus.status

            # Check if this is an exit order fill (stop or timed)
            if order_id in self._exit_to_parent and status == "Filled":
                self._handle_exit_fill(order_id)
                return

            # Check if this is a timed exit cancel (needs flatten retry)
            if order_id in self._timed_exit_to_parent and status in ("Cancelled", "ApiCancelled", "Inactive"):
                self._handle_timed_exit_cancel(order_id)
                return

            # Check if this is a pending gap order that needs bracket completion
            if status == "Filled" and self._is_pending_gap_order(order_id):
                self._handle_gap_order_fill(order_id, trade)
                return

            # Check if this is a parent order we're tracking
            if order_id not in self.pending_orders:
                return

            pending = self.pending_orders[order_id]

            if status == "Filled":
                self._handle_fill(pending, trade)
            elif status in ("Cancelled", "ApiCancelled", "Inactive"):
                self._handle_cancel(pending)

    def _on_exec_details(self, trade: "Trade", fill) -> None:
        """Handle execution details (fills) from IBKR.

        Tracks partial fills by accumulating filled_qty and calculating
        weighted average fill price.
        """
        with self._lock:
            order_id = trade.order.orderId

            if order_id not in self.pending_orders:
                return

            pending = self.pending_orders[order_id]
            fill_shares = int(fill.execution.shares)
            fill_price = float(fill.execution.price)

            # Calculate weighted average price for partial fills
            prev_total = pending.filled_qty * pending.avg_fill_price
            new_total = prev_total + (fill_shares * fill_price)
            pending.filled_qty += fill_shares
            if pending.filled_qty > 0:
                pending.avg_fill_price = new_total / pending.filled_qty

            self.logger.info(
                "[FILL_TRACKER] Execution: order=%s symbol=%s fill_qty=%d fill_price=%.4f "
                "cumulative=%d/%d avg_price=%.4f",
                order_id, pending.symbol, fill_shares, fill_price,
                pending.filled_qty, pending.qty, pending.avg_fill_price
            )

    def _handle_fill(self, pending: PendingOrder, trade: "Trade") -> None:
        """
        Handle a filled order.

        1. Increment fill counter
        2. Remove from pending
        3. Create FilledPosition for conflict tracking
        4. If cap reached, cancel remaining pending orders for this strategy
        """
        state = self._get_or_create_state(
            pending.trade_type, pending.trade_date, pending.strategy_id
        )

        # Increment fill count
        state.fill_count += 1
        state.filled_symbols.append(pending.symbol)
        state.pending_order_ids.discard(pending.order_id)

        # Use accumulated partial fill data if available, otherwise use trade data
        if pending.filled_qty > 0:
            # Use tracked partial fill data
            actual_qty = pending.filled_qty
            fill_price = pending.avg_fill_price
        else:
            # Fallback to trade status for complete fills
            actual_qty = pending.qty
            fill_price = trade.orderStatus.avgFillPrice or 0.0

        # Log if there was a partial fill (filled less than ordered)
        if actual_qty < pending.qty:
            self.logger.warning(
                "[FILL_TRACKER] Partial fill completed: order=%d symbol=%s filled=%d/%d",
                pending.order_id, pending.symbol, actual_qty, pending.qty
            )

        # Create FilledPosition for conflict resolution tracking
        filled_pos = FilledPosition(
            symbol=pending.symbol,
            side=pending.side,
            kind=pending.trade_type,
            strategy_id=pending.strategy_id,
            qty=actual_qty,  # Use actual filled quantity
            fill_price=fill_price,
            fill_time=now_pt(),
            parent_order_id=pending.order_id,
            stop_order_id=pending.stop_order_id,
            timed_order_id=pending.timed_order_id,
            trade_date=pending.trade_date,
        )
        self.filled_positions[pending.order_id] = filled_pos

        # Track exit orders so we can remove position when they fill
        if pending.stop_order_id:
            self._exit_to_parent[pending.stop_order_id] = pending.order_id
        if pending.timed_order_id:
            self._exit_to_parent[pending.timed_order_id] = pending.order_id
            # Also track timed exits separately for cancel handling
            self._timed_exit_to_parent[pending.timed_order_id] = pending.order_id

        # Remove from pending tracking
        del self.pending_orders[pending.order_id]

        cap = DAY_STRATEGY_FILL_CAP if pending.trade_type == "DAY" else SWING_STRATEGY_FILL_CAP

        self.logger.info(
            "[FILL_TRACKER] FILLED: symbol=%s side=%s strategy=%s fills=%d/%d",
            pending.symbol, pending.side, pending.strategy_id, state.fill_count, cap
        )

        # Check if cap reached
        if state.fill_count >= cap:
            self._cancel_remaining_for_strategy(pending.trade_type, pending.trade_date, pending.strategy_id)

        # Check if this is a re-entry order that needs slot conversion
        if pending.order_id in self._pending_reentry_conversions:
            trade_date = self._pending_reentry_conversions.pop(pending.order_id)
            self.logger.info(
                "[FILL_TRACKER] Re-entry order %d filled, converting reserved slot to open",
                pending.order_id,
            )
            # Convert reserved slot to open directly (more reliable than callback)
            if self.state_mgr is not None:
                try:
                    self.state_mgr.convert_reserved_to_open(trade_date)
                    self.logger.info(
                        "[FILL_TRACKER] Successfully converted reserved slot to open for order %d",
                        pending.order_id,
                    )
                except Exception as exc:
                    self.logger.critical(
                        "[FILL_TRACKER] CRITICAL: Failed to convert reserved slot for order %d: %s. "
                        "Manual state.json fix may be required.",
                        pending.order_id, exc
                    )
            else:
                self.logger.error(
                    "[FILL_TRACKER] No state_mgr available for slot conversion - slot may be leaked"
                )

    def _handle_cancel(self, pending: PendingOrder) -> None:
        """
        Handle a cancelled order - clean up tracking and notify callbacks.

        For Day trades, notifies ReentryManager that the Day trade was cancelled
        (possibly due to stock halt), so it can schedule EOD evaluation.

        Also decrements the cap counter since the order never filled.
        """
        state = self._get_or_create_state(
            pending.trade_type, pending.trade_date, pending.strategy_id
        )
        state.pending_order_ids.discard(pending.order_id)

        if pending.order_id in self.pending_orders:
            del self.pending_orders[pending.order_id]

        self.logger.info(
            "[FILL_TRACKER] Order cancelled/removed: id=%s symbol=%s strategy=%s",
            pending.order_id, pending.symbol, pending.strategy_id
        )

        # Decrement cap counter since order never filled
        # Cap was incremented at placement time, so we need to undo it
        if self.cap_manager is not None:
            if pending.trade_date is None:
                self.logger.warning(
                    "[FILL_TRACKER] Cannot decrement cap counter for cancelled order %s - trade_date is None",
                    pending.symbol,
                )
            else:
                try:
                    if pending.trade_type == "DAY":
                        self.cap_manager.register_day_exit(pending.trade_date)
                        self.logger.info(
                            "[FILL_TRACKER] Decremented DayOpen counter for cancelled order %s",
                            pending.symbol,
                        )
                    elif pending.trade_type == "SWING":
                        self.cap_manager.register_swing_exit(pending.trade_date)
                        self.logger.info(
                            "[FILL_TRACKER] Decremented SwingOpen counter for cancelled order %s",
                            pending.symbol,
                        )
                except Exception as exc:
                    self.logger.error(
                        "[FILL_TRACKER] Error decrementing cap counter for cancelled order: %s", exc
                    )

        # Remove from pending_gap_orders if this was a gap order
        if self.state_mgr is not None:
            removed = self.state_mgr.pending_gap_orders.remove_pending(pending.order_id)
            if removed:
                self.logger.info(
                    "[FILL_TRACKER] Removed cancelled gap order %d from pending_gap_orders",
                    pending.order_id,
                )

        # Notify ReentryManager for any cancelled DAY order (filled or unfilled)
        # This handles:
        # 1. Filled position cancelled (halt) -> schedules EOD evaluation
        # 2. Unfilled order cancelled (limit not reached) -> schedules EOD evaluation
        #    for any linked candidates (SWING was already flattened)
        if pending.trade_type == "DAY" and self._on_day_cancel_callback is not None:
            try:
                self._on_day_cancel_callback(pending.order_id)
            except Exception as exc:
                self.logger.error(
                    "[FILL_TRACKER] Error in day cancel callback: %s", exc
                )

    def _is_pending_gap_order(self, order_id: int) -> bool:
        """Check if an order is a pending gap order awaiting bracket completion."""
        if self.state_mgr is None:
            return False
        return self.state_mgr.pending_gap_orders.is_pending_gap_order(order_id)

    def _handle_gap_order_fill(self, order_id: int, trade: "Trade") -> None:
        """
        Handle a gap MKT order fill.

        When a gap MKT order fills completely, call the gap completion callback
        to place stop and timed exit orders.
        """
        if self.state_mgr is None:
            self.logger.error(
                "[FILL_TRACKER] Cannot handle gap fill - no state_mgr"
            )
            return

        pending_gap = self.state_mgr.pending_gap_orders.get_pending(order_id)
        if pending_gap is None:
            self.logger.warning(
                "[FILL_TRACKER] Gap order %d not found in pending_gap_orders",
                order_id,
            )
            return

        # Get fill details
        filled_qty = int(trade.orderStatus.filled)
        total_qty = pending_gap["shares"]
        avg_price = float(trade.orderStatus.avgFillPrice or 0.0)

        # Check if completely filled
        if filled_qty < total_qty:
            self.logger.info(
                "[FILL_TRACKER] Gap order %d partial fill: %d/%d - waiting for complete fill",
                order_id, filled_qty, total_qty,
            )
            return

        self.logger.info(
            "[FILL_TRACKER] Gap order %d complete fill: %d shares @ %.2f",
            order_id, filled_qty, avg_price,
        )

        # Create FilledPosition for conflict tracking (before stop/timed are placed)
        filled_pos = FilledPosition(
            symbol=pending_gap["symbol"],
            side=pending_gap["direction"],
            kind=pending_gap["signal_type"],
            strategy_id=pending_gap["strategy_id"],
            qty=filled_qty,
            fill_price=avg_price,
            fill_time=now_pt(),
            parent_order_id=order_id,
            stop_order_id=None,  # Will be updated by register_gap_bracket_completion
            timed_order_id=None,
            trade_date=now_pt().date(),
        )
        self.filled_positions[order_id] = filled_pos

        # Call gap completion callback (GapManager.complete_gap_bracket)
        if self._on_gap_fill_callback is not None:
            try:
                success = self._on_gap_fill_callback(order_id, avg_price, filled_qty)
                if success:
                    self.logger.info(
                        "[FILL_TRACKER] Gap bracket completed for order %d",
                        order_id,
                    )
                else:
                    self.logger.error(
                        "[FILL_TRACKER] Gap bracket completion failed for order %d",
                        order_id,
                    )
            except Exception as exc:
                self.logger.error(
                    "[FILL_TRACKER] Error in gap fill callback: %s", exc
                )
        else:
            self.logger.warning(
                "[FILL_TRACKER] No gap fill callback registered for order %d",
                order_id,
            )

    def _cancel_remaining_for_strategy(
        self, trade_type: str, trade_date: date, strategy_id: str
    ) -> None:
        """
        Cancel all remaining unfilled orders for a strategy that has hit its cap.
        """
        state = self._get_or_create_state(trade_type, trade_date, strategy_id)

        if not state.pending_order_ids:
            self.logger.info(
                "[FILL_TRACKER] Strategy %s hit cap - no pending orders to cancel.",
                strategy_id
            )
            return

        self.logger.info(
            "[FILL_TRACKER] Strategy %s hit cap (%d fills) - cancelling %d pending orders.",
            strategy_id, state.fill_count, len(state.pending_order_ids)
        )

        # Get order IDs to cancel (copy to avoid modification during iteration)
        order_ids_to_cancel = list(state.pending_order_ids)

        for order_id in order_ids_to_cancel:
            if order_id in self.pending_orders:
                pending = self.pending_orders[order_id]
                self._cancel_order(order_id, pending.symbol)

    def _cancel_order(self, order_id: int, symbol: str) -> None:
        """Cancel a specific order by ID."""
        try:
            # Find the trade object for this order
            for trade in self.ib.openTrades():
                if trade.order.orderId == order_id:
                    self.ib.cancelOrder(trade.order)
                    self.logger.info(
                        "[FILL_TRACKER] Cancelled order: id=%s symbol=%s (strategy cap reached)",
                        order_id, symbol
                    )
                    return

            self.logger.warning(
                "[FILL_TRACKER] Could not find order to cancel: id=%s symbol=%s",
                order_id, symbol
            )
        except Exception as exc:
            self.logger.error(
                "[FILL_TRACKER] Error cancelling order %s: %s", order_id, exc
            )

    def get_strategy_summary(self, trade_type: str, trade_date: date, strategy_id: str) -> dict:
        """Get a summary of the strategy's fill state."""
        state = self._get_or_create_state(trade_type, trade_date, strategy_id)
        cap = DAY_STRATEGY_FILL_CAP if trade_type == "DAY" else SWING_STRATEGY_FILL_CAP

        return {
            "strategy_id": strategy_id,
            "trade_type": trade_type,
            "fill_count": state.fill_count,
            "fill_cap": cap,
            "pending_count": len(state.pending_order_ids),
            "filled_symbols": state.filled_symbols,
            "cap_reached": state.fill_count >= cap,
        }

    # ---------- Position tracking for ConflictResolver ---------- #

    def _handle_timed_exit_cancel(self, timed_order_id: int) -> None:
        """
        Handle a cancelled timed exit MKT order (Day or Swing).

        When a timed exit is cancelled (stock halt, broker issue, etc.),
        the position is still open and needs to be flattened.

        Notifies via callback to trigger flatten retry.
        """
        parent_order_id = self._timed_exit_to_parent.get(timed_order_id)
        if parent_order_id is None:
            return

        pos = self.filled_positions.get(parent_order_id)
        if pos is None:
            self.logger.warning(
                "[FILL_TRACKER] Timed exit cancelled but no position found: timed_order_id=%d",
                timed_order_id,
            )
            return

        self.logger.warning(
            "[FILL_TRACKER] TIMED EXIT CANCELLED: symbol=%s side=%s kind=%s - needs flatten retry",
            pos.symbol, pos.side, pos.kind,
        )

        # Notify callback to trigger flatten retry (for both Day and Swing)
        callback_success = False
        if self._on_timed_exit_cancel_callback is not None:
            try:
                self._on_timed_exit_cancel_callback(pos)
                callback_success = True
            except Exception as exc:
                self.logger.error(
                    "[FILL_TRACKER] Error in timed exit cancel callback: %s", exc
                )
                self.logger.critical(
                    "[FILL_TRACKER] CRITICAL: Position %s %s %s needs manual intervention! "
                    "Timed exit cancelled and callback failed. Position may still be open.",
                    pos.symbol, pos.kind, pos.side,
                )
        else:
            self.logger.warning(
                "[FILL_TRACKER] No timed exit cancel callback registered! "
                "Position %s %s %s may remain open.",
                pos.symbol, pos.kind, pos.side,
            )

        # Clean up the timed exit tracking (position still open, may get new exit order)
        if timed_order_id in self._timed_exit_to_parent:
            del self._timed_exit_to_parent[timed_order_id]
        # Also clean up _exit_to_parent to prevent stale mappings
        if timed_order_id in self._exit_to_parent:
            del self._exit_to_parent[timed_order_id]

    def _handle_exit_fill(self, exit_order_id: int) -> None:
        """
        Handle an exit order fill (stop or timed).

        Removes the position from tracking since it's now closed.
        Decrements the cap counter and notifies ReentryManager for DAY trades.
        """
        parent_order_id = self._exit_to_parent.get(exit_order_id)
        if parent_order_id is None:
            return

        if parent_order_id in self.filled_positions:
            pos = self.filled_positions[parent_order_id]
            self.logger.info(
                "[FILL_TRACKER] EXIT FILLED: symbol=%s side=%s kind=%s (position closed)",
                pos.symbol, pos.side, pos.kind
            )

            # Decrement cap counter for the exited position
            if self.cap_manager is not None:
                if pos.trade_date is None:
                    self.logger.warning(
                        "[FILL_TRACKER] Cannot decrement cap counter for %s - trade_date is None",
                        pos.symbol,
                    )
                else:
                    try:
                        if pos.kind == "DAY":
                            self.cap_manager.register_day_exit(pos.trade_date)
                            self.logger.info(
                                "[FILL_TRACKER] Decremented DayOpen counter for %s",
                                pos.symbol,
                            )
                        elif pos.kind == "SWING":
                            self.cap_manager.register_swing_exit(pos.trade_date)
                            self.logger.info(
                                "[FILL_TRACKER] Decremented SwingOpen counter for %s",
                                pos.symbol,
                            )
                    except Exception as exc:
                        self.logger.error(
                            "[FILL_TRACKER] Error decrementing cap counter: %s", exc
                        )

            # Notify ReentryManager if this is a DAY trade
            if pos.kind == "DAY" and self._on_day_exit_callback is not None:
                try:
                    self._on_day_exit_callback(parent_order_id)
                except Exception as exc:
                    self.logger.error(
                        "[FILL_TRACKER] Error in day exit callback: %s", exc
                    )

            self.remove_filled_position(parent_order_id)

    def get_filled_positions_by_symbol(self, symbol: str) -> List[FilledPosition]:
        """
        Get all filled (open) positions for a symbol.

        Used by ConflictResolver to check for conflicts.
        """
        symbol = symbol.upper()
        return [p for p in self.filled_positions.values() if p.symbol == symbol]

    def get_all_filled_positions(self) -> List[FilledPosition]:
        """Get all currently filled (open) positions."""
        return list(self.filled_positions.values())

    def remove_filled_position(self, parent_order_id: int) -> Optional[FilledPosition]:
        """
        Remove a filled position from tracking.

        Called when:
        - Exit order fills (stop/timed)
        - Position is manually flattened via ConflictResolver

        Returns the removed position or None if not found.
        """
        with self._lock:
            if parent_order_id not in self.filled_positions:
                return None

            pos = self.filled_positions.pop(parent_order_id)

            # Clean up exit order mappings
            if pos.stop_order_id and pos.stop_order_id in self._exit_to_parent:
                del self._exit_to_parent[pos.stop_order_id]
            if pos.timed_order_id:
                if pos.timed_order_id in self._exit_to_parent:
                    del self._exit_to_parent[pos.timed_order_id]
                # Also clean up _timed_exit_to_parent to prevent stale mappings
                if pos.timed_order_id in self._timed_exit_to_parent:
                    del self._timed_exit_to_parent[pos.timed_order_id]

            self.logger.info(
                "[FILL_TRACKER] Position removed: symbol=%s side=%s kind=%s strategy=%s",
                pos.symbol, pos.side, pos.kind, pos.strategy_id
            )
            return pos

    def has_position(self, symbol: str, kind: Optional[str] = None, side: Optional[str] = None) -> bool:
        """
        Check if any filled position exists for a symbol.

        Args:
            symbol: Stock symbol
            kind: Optional filter by "DAY" or "SWING"
            side: Optional filter by "LONG" or "SHORT"
        """
        positions = self.get_filled_positions_by_symbol(symbol)
        if kind:
            positions = [p for p in positions if p.kind == kind.upper()]
        if side:
            positions = [p for p in positions if p.side == side.upper()]
        return len(positions) > 0

    def get_position_summary(self) -> dict:
        """Get a summary of all filled positions."""
        positions = self.get_all_filled_positions()
        by_symbol: Dict[str, List[str]] = {}

        for p in positions:
            if p.symbol not in by_symbol:
                by_symbol[p.symbol] = []
            by_symbol[p.symbol].append(f"{p.kind} {p.side} ({p.strategy_id})")

        return {
            "total_positions": len(positions),
            "by_symbol": by_symbol,
        }
