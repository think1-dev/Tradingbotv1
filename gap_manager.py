"""
gap_manager.py

Gap-at-Open Trade Manager for handling gap triggers at market open.

Gap Logic:
- Day LONG / Swing Pullback / Swing Mean Reversion: Gap DOWN (open <= entry) triggers MKT buy
- Day SHORT / Swing MOMO: Gap UP (open >= entry) triggers MKT sell/buy

Gap Criteria:
- Single check at 6:30 PT exactly (first tick after open)
- Check if open price is at or beyond entry price
- If gap condition NOT met, signal remains armed for normal LMT fill

Execution Flow:
1. Place single-leg MKT order
2. Wait for complete fill (monitored by FillTracker)
3. After fill, place stop + timed exit orders to complete bracket
4. Stop price = fill_price ± stop_distance (preserves original risk)
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from datetime import date
from typing import TYPE_CHECKING, Dict, List, Optional, Any, Tuple
from zoneinfo import ZoneInfo

from time_utils import now_pt

if TYPE_CHECKING:
    from ib_insync import IB
    from signals import DaySignal, SwingSignal
    from fill_tracker import FillTracker
    from execution import OrderExecutor
    from state_manager import StateManager
    from cap_manager import CapManager
    from conflict_resolver import ConflictResolver
    from reentry_manager import ReentryManager

PT = ZoneInfo("America/Los_Angeles")


@dataclass
class GapCandidate:
    """Represents a signal that could trigger on a gap."""
    symbol: str
    strategy_id: str
    signal_type: str  # "DAY" or "SWING"
    direction: str  # "LONG" or "SHORT"
    entry_price: float
    original_stop: float
    shares: int
    gap_direction_needed: str  # "UP" or "DOWN"
    stop_distance: float  # Pre-calculated from signal


class GapManager:
    """
    Manages gap-at-open trade detection and execution.

    Lifecycle:
    1. Market Open (6:30 PT): Run gap detection
    2. For each signal, check if open price gaps through entry
    3. Place single-leg MKT order (position unprotected)
    4. Wait for complete fill
    5. Complete bracket with stop + timed orders

    Integration:
    - Uses ConflictResolver before gap entries
    - Uses CapManager for cap checks
    - Uses FillTracker for fill monitoring and bracket completion
    """

    def __init__(
        self,
        ib: "IB",
        logger: logging.Logger,
        fill_tracker: "FillTracker",
        executor: "OrderExecutor",
        state_mgr: "StateManager",
        cap_manager: "CapManager",
        conflict_resolver: Optional["ConflictResolver"] = None,
        reentry_manager: Optional["ReentryManager"] = None,
        swing_signals_by_key: Optional[Dict[str, "SwingSignal"]] = None,
    ) -> None:
        self.ib = ib
        self.logger = logger
        self.fill_tracker = fill_tracker
        self.executor = executor
        self.state_mgr = state_mgr
        self.cap_manager = cap_manager
        self.conflict_resolver = conflict_resolver
        self.reentry_manager = reentry_manager
        self._swing_signals_by_key = swing_signals_by_key or {}

        # Track if gap check has run today
        self._gap_check_date: Optional[date] = None

        # Lock to prevent concurrent gap check execution
        self._gap_check_lock = threading.Lock()

        self.logger.info("[GAP] Initialized (simplified - no prev_close tracking).")

    # === Public Interface ===

    def run_gap_check(
        self,
        day_signals: List["DaySignal"],
        swing_signals: List["SwingSignal"],
        day_runtimes: Dict[str, List],
        swing_runtimes: Dict[str, List],
    ) -> Dict[str, List[int]]:
        """
        Run gap detection at market open.

        Should be called once at 6:30 PT exactly.

        Args:
            day_signals: List of DaySignal objects
            swing_signals: List of SwingSignal objects
            day_runtimes: Dict of symbol -> List[DayRuntime] for marking triggered
            swing_runtimes: Dict of symbol -> List[SwingRuntime] for marking triggered

        Returns:
            Dict with "day_orders" and "swing_orders" lists of placed order IDs
        """
        with self._gap_check_lock:
            today = now_pt().date()

            # Only run once per day
            if self._gap_check_date == today:
                self.logger.info("[GAP] Gap check already ran today, skipping.")
                return {"day_orders": [], "swing_orders": []}

            self._gap_check_date = today

            self.logger.info("[GAP] Running gap check for %s", today)

            results = {"day_orders": [], "swing_orders": []}

            # Build gap candidates from signals
            day_candidates = self._build_day_candidates(day_signals)
            swing_candidates = self._build_swing_candidates(swing_signals)

            self.logger.info(
                "[GAP] Candidates: %d day, %d swing",
                len(day_candidates),
                len(swing_candidates),
            )

            # Process day gap candidates
            for candidate in day_candidates:
                order_id = self._process_gap_candidate(
                    candidate, day_runtimes, is_day=True
                )
                if order_id is not None:
                    results["day_orders"].append(order_id)

            # Process swing gap candidates
            for candidate in swing_candidates:
                order_id = self._process_gap_candidate(
                    candidate, swing_runtimes, is_day=False
                )
                if order_id is not None:
                    results["swing_orders"].append(order_id)

            self.logger.info(
                "[GAP] Gap check complete: %d day orders, %d swing orders placed",
                len(results["day_orders"]),
                len(results["swing_orders"]),
            )

            return results

    def complete_gap_bracket(
        self,
        order_id: int,
        fill_price: float,
        fill_qty: int,
    ) -> bool:
        """
        Complete a gap bracket after MKT order fills.

        Called by FillTracker when a pending gap order is completely filled.

        Args:
            order_id: The filled MKT order ID
            fill_price: Average fill price
            fill_qty: Filled quantity

        Returns:
            True if bracket completed successfully, False otherwise
        """
        # Get pending gap order data
        pending = self.state_mgr.pending_gap_orders.get_pending(order_id)
        if pending is None:
            self.logger.error(
                "[GAP] complete_gap_bracket called for unknown order %d", order_id
            )
            return False

        symbol = pending["symbol"]
        strategy_id = pending["strategy_id"]
        signal_type = pending["signal_type"]
        direction = pending["direction"]
        stop_distance = pending["stop_distance"]
        shares = pending["shares"]

        # Calculate new stop based on fill price
        if direction == "LONG":
            new_stop = fill_price - stop_distance
        else:
            new_stop = fill_price + stop_distance

        self.logger.info(
            "[GAP] Completing bracket for %s %s: fill=%.2f, stop=%.2f (distance=%.2f)",
            symbol, strategy_id, fill_price, new_stop, stop_distance,
        )

        try:
            # Place stop and timed exit orders
            success = self._place_stop_and_timed(
                symbol=symbol,
                strategy_id=strategy_id,
                signal_type=signal_type,
                direction=direction,
                shares=shares,
                stop_price=new_stop,
                parent_order_id=order_id,
            )

            if success:
                # Remove from pending
                self.state_mgr.pending_gap_orders.remove_pending(order_id)
                self.logger.info(
                    "[GAP] Bracket complete for %s %s (order %d)",
                    symbol, strategy_id, order_id,
                )
                return True
            else:
                self.logger.error(
                    "[GAP] Failed to place stop/timed for %s %s",
                    symbol, strategy_id,
                )
                return False

        except Exception as exc:
            self.logger.error(
                "[GAP] Error completing bracket for %s: %s", symbol, exc
            )
            return False

    # === Internal Methods ===

    def _build_day_candidates(
        self, signals: List["DaySignal"]
    ) -> List[GapCandidate]:
        """Build gap candidates from day signals."""
        candidates = []

        for sig in signals:
            direction = (getattr(sig, "direction", None) or "LONG").upper()

            # Day LONG: needs gap DOWN (open <= entry)
            # Day SHORT: needs gap UP (open >= entry)
            gap_direction = "DOWN" if direction == "LONG" else "UP"

            candidates.append(GapCandidate(
                symbol=sig.symbol.upper(),
                strategy_id=sig.strategy_id,
                signal_type="DAY",
                direction=direction,
                entry_price=float(sig.entry_price),
                original_stop=float(sig.stop_price),
                shares=sig.shares,
                gap_direction_needed=gap_direction,
                stop_distance=sig.stop_distance,
            ))

        return candidates

    def _build_swing_candidates(
        self, signals: List["SwingSignal"]
    ) -> List[GapCandidate]:
        """Build gap candidates from swing signals."""
        candidates = []

        for sig in signals:
            strategy_lower = (sig.strategy_id or "").lower()

            # Swing MOMO/Breakout: needs gap UP (open >= entry)
            # Swing Pullback/MeanRev: needs gap DOWN (open <= entry)
            gap_direction = "UP" if ("momo" in strategy_lower or "breakout" in strategy_lower) else "DOWN"

            candidates.append(GapCandidate(
                symbol=sig.symbol.upper(),
                strategy_id=sig.strategy_id,
                signal_type="SWING",
                direction="LONG",  # Swings are LONG-only
                entry_price=float(sig.entry_price),
                original_stop=float(sig.stop_price),
                shares=sig.shares,
                gap_direction_needed=gap_direction,
                stop_distance=sig.stop_distance,
            ))

        return candidates

    def _process_gap_candidate(
        self,
        candidate: GapCandidate,
        runtimes: Dict[str, List],
        is_day: bool,
    ) -> Optional[int]:
        """
        Process a single gap candidate.

        Returns order ID if gap MKT order was placed, None otherwise.
        """
        symbol = candidate.symbol

        # Get open price
        open_price = self._get_open_price(symbol)
        if open_price is None:
            self.logger.warning(
                "[GAP] No open price available for %s, skipping gap check.",
                symbol,
            )
            return None

        # Check gap condition (simplified - just open vs entry)
        gap_met = self._check_gap_condition(candidate, open_price)

        if not gap_met:
            self.logger.info(
                "[GAP] No gap trigger for %s %s %s (open=%.2f, entry=%.2f, need=%s)",
                candidate.signal_type,
                candidate.direction,
                symbol,
                open_price,
                candidate.entry_price,
                candidate.gap_direction_needed,
            )
            return None

        # Gap condition met - check if symbol+strategy is blocked
        today = now_pt().date()
        if is_day:
            is_blocked, block_reason = self.state_mgr.blocked.is_blocked(
                symbol, candidate.strategy_id, today
            )
        else:
            is_blocked, block_reason = self.state_mgr.blocked.is_blocked(
                symbol, candidate.strategy_id, today, check_week=True, check_day=False
            )

        if is_blocked:
            self.logger.info(
                "[GAP][SKIP] %s %s on %s - %s",
                candidate.signal_type,
                candidate.direction,
                symbol,
                block_reason,
            )
            self._mark_signal_failed(symbol, candidate.strategy_id, runtimes)
            return None

        # Check caps
        if is_day:
            can_trade, reason = self._check_day_caps(candidate)
        else:
            can_trade, reason = self._check_swing_caps(candidate)

        if not can_trade:
            self.logger.info(
                "[GAP][SKIP] %s %s on %s - cap blocked: %s",
                candidate.signal_type,
                candidate.direction,
                symbol,
                reason,
            )
            self._mark_signal_failed(symbol, candidate.strategy_id, runtimes)
            return None

        # Check conflict resolution
        allowed, reentry_candidate_ids = self._check_conflicts(candidate, is_day)
        if not allowed:
            self.logger.info(
                "[GAP][SKIP] %s %s on %s - conflict resolution blocked",
                candidate.signal_type,
                candidate.direction,
                symbol,
            )
            self._mark_signal_failed(symbol, candidate.strategy_id, runtimes)
            return None

        # Place single-leg MKT order (stop/timed added after fill)
        order_id = self._place_gap_market_order(candidate, is_day)

        if order_id is not None:
            self.logger.warning(
                "[GAP][TRIGGER] %s %s on %s - MKT order %d placed (UNPROTECTED until fill)",
                candidate.signal_type,
                candidate.direction,
                symbol,
                order_id,
            )
            # Mark signal as triggered
            self._mark_signal_triggered(symbol, candidate.strategy_id, runtimes)

            # Link re-entry candidates to this gap Day order
            if is_day and reentry_candidate_ids and self.reentry_manager is not None:
                for cid in reentry_candidate_ids:
                    self.reentry_manager.link_day_trade(cid, order_id)
        else:
            self.logger.warning(
                "[GAP][FAIL] %s %s on %s - MKT order placement failed",
                candidate.signal_type,
                candidate.direction,
                symbol,
            )
            # Clean up re-entry candidates
            if reentry_candidate_ids and self.reentry_manager is not None:
                for cid in reentry_candidate_ids:
                    try:
                        self.reentry_manager.drop_candidate(cid, "gap_trade_failed")
                    except Exception as exc:
                        self.logger.error(
                            "[GAP] Failed to drop candidate %s: %s", cid, exc
                        )

            # Block symbol+strategy
            self.state_mgr.blocked.block_for_week(symbol, candidate.strategy_id, today)
            self.state_mgr.blocked.block_for_day(symbol, candidate.strategy_id, today)

            self._mark_signal_failed(symbol, candidate.strategy_id, runtimes)

        return order_id

    def _check_gap_condition(
        self,
        candidate: GapCandidate,
        open_price: float,
    ) -> bool:
        """
        Check if gap condition is met.

        Simple check: does open price meet or exceed entry in the required direction?
        """
        entry = candidate.entry_price
        gap_needed = candidate.gap_direction_needed

        if gap_needed == "DOWN":
            # Gap down: open must be AT or BELOW entry
            return open_price <= entry
        else:  # UP
            # Gap up: open must be AT or ABOVE entry
            return open_price >= entry

    def _get_open_price(self, symbol: str) -> Optional[float]:
        """Get the open price (current market snapshot) for a symbol."""
        try:
            from ib_insync import Stock

            contract = Stock(symbol, "SMART", "USD")
            qualified = self.ib.qualifyContracts(contract)
            if not qualified:
                return None

            # Request snapshot
            ticker = self.ib.reqMktData(contract, "", True, False)
            self.ib.sleep(0.5)

            # Use last, or bid/ask midpoint
            if ticker.last is not None and ticker.last > 0:
                return ticker.last
            if ticker.bid is not None and ticker.ask is not None:
                if ticker.bid > 0 and ticker.ask > 0:
                    return (ticker.bid + ticker.ask) / 2.0

            return None
        except Exception as exc:
            self.logger.warning(
                "[GAP] Error getting open price for %s: %s", symbol, exc
            )
            return None

    def _check_day_caps(self, candidate: GapCandidate) -> Tuple[bool, Optional[str]]:
        """Check day trade caps."""
        class MinimalSignal:
            def __init__(self, symbol, strategy_id, trade_date):
                self.symbol = symbol
                self.strategy_id = strategy_id
                self.trade_date = trade_date

        sig = MinimalSignal(
            candidate.symbol,
            candidate.strategy_id,
            now_pt().date(),
        )
        return self.cap_manager.can_open_day(sig)

    def _check_swing_caps(self, candidate: GapCandidate) -> Tuple[bool, Optional[str]]:
        """Check swing trade caps."""
        class MinimalSignal:
            def __init__(self, symbol, strategy_id, trade_date):
                self.symbol = symbol
                self.strategy_id = strategy_id
                self.trade_date = trade_date

        sig = MinimalSignal(
            candidate.symbol,
            candidate.strategy_id,
            now_pt().date(),
        )
        return self.cap_manager.can_open_swing(sig)

    def _check_conflicts(self, candidate: GapCandidate, is_day: bool) -> Tuple[bool, List[str]]:
        """
        Check conflict resolution and handle flattening.

        Returns (allowed: bool, reentry_candidate_ids: List[str])
        """
        reentry_candidate_ids = []

        if self.conflict_resolver is None:
            return True, reentry_candidate_ids

        class MinimalSignal:
            def __init__(self, symbol, direction, trade_type):
                self.symbol = symbol
                self.direction = direction
                self.trade_type = trade_type

        sig = MinimalSignal(
            candidate.symbol,
            candidate.direction,
            "DAY" if is_day else "SWING",
        )

        decision = self.conflict_resolver.decide(sig)

        if not decision.allow_entry:
            return False, reentry_candidate_ids

        if decision.requires_flatten:
            for instr in decision.positions_to_flatten:
                # For day gap trades, store re-entry candidate if flattening swing
                if is_day and instr.kind == "SWING" and self.reentry_manager is not None:
                    signal_key = f"{instr.symbol}_{instr.strategy_id}"
                    original_signal = self._swing_signals_by_key.get(signal_key)

                    if original_signal is not None:
                        from fill_tracker import FilledPosition
                        flattened_pos = FilledPosition(
                            symbol=instr.symbol,
                            side=instr.side,
                            kind=instr.kind,
                            strategy_id=instr.strategy_id,
                            qty=instr.qty,
                            fill_price=0.0,
                            fill_time=now_pt(),
                            parent_order_id=instr.parent_order_id,
                            stop_order_id=instr.stop_order_id,
                            timed_order_id=instr.timed_order_id,
                        )
                        try:
                            cid = self.reentry_manager.store_candidate(flattened_pos, original_signal)
                            reentry_candidate_ids.append(cid)
                            self.logger.info(
                                "[GAP] Created re-entry candidate %s for flattened SWING %s %s",
                                cid, instr.symbol, instr.strategy_id,
                            )
                        except Exception as exc:
                            self.logger.error(
                                "[GAP] Failed to create re-entry candidate for %s %s: %s",
                                instr.symbol, instr.strategy_id, exc,
                            )

                success = self.executor.flatten_position_with_retry(instr)
                if not success:
                    for cid in reentry_candidate_ids:
                        try:
                            self.reentry_manager.drop_candidate(cid, "flatten_failed")
                        except Exception as exc:
                            self.logger.error(
                                "[GAP] Failed to drop candidate %s: %s", cid, exc
                            )
                    return False, []

        return True, reentry_candidate_ids

    def _place_gap_market_order(
        self,
        candidate: GapCandidate,
        is_day: bool,
    ) -> Optional[int]:
        """
        Place a single-leg MKT order for a gap trade.

        Stop and timed orders will be added after fill via complete_gap_bracket().
        """
        from ib_insync import Order
        from orders import make_stock_contract

        try:
            # Get contract
            contract = make_stock_contract(candidate.symbol)
            qualified = self.ib.qualifyContracts(contract)
            if not qualified:
                self.logger.error("[GAP] Failed to qualify contract for %s", candidate.symbol)
                return None

            # Create MKT order
            action = "BUY" if candidate.direction == "LONG" else "SELL"
            order = Order(
                action=action,
                orderType="MKT",
                totalQuantity=candidate.shares,
                tif="DAY",
                transmit=True,
            )

            # Get order ID
            order_id = self.executor._get_next_order_id()
            order.orderId = order_id

            # Place order
            self.ib.placeOrder(contract, order)

            self.logger.info(
                "[GAP] MKT order placed: %s %s %d shares (order_id=%d)",
                action, candidate.symbol, candidate.shares, order_id,
            )

            # Store in pending gap orders for completion after fill
            self.state_mgr.pending_gap_orders.add_pending(
                order_id=order_id,
                symbol=candidate.symbol,
                strategy_id=candidate.strategy_id,
                signal_type=candidate.signal_type,
                direction=candidate.direction,
                stop_distance=candidate.stop_distance,
                shares=candidate.shares,
            )

            # Register with cap manager (optimistic - on placement)
            trade_date = now_pt().date()

            class GapSignal:
                def __init__(self, symbol, strategy_id, trade_date):
                    self.symbol = symbol
                    self.strategy_id = strategy_id
                    self.trade_date = trade_date

            gap_sig = GapSignal(candidate.symbol, candidate.strategy_id, trade_date)

            if is_day:
                self.cap_manager.register_day_entry(gap_sig)
            else:
                self.cap_manager.register_swing_entry(gap_sig)

            return order_id

        except Exception as exc:
            self.logger.error(
                "[GAP] Error placing MKT order for %s: %s",
                candidate.symbol, exc,
            )
            return None

    def _place_stop_and_timed(
        self,
        symbol: str,
        strategy_id: str,
        signal_type: str,
        direction: str,
        shares: int,
        stop_price: float,
        parent_order_id: int,
    ) -> bool:
        """
        Place stop and timed exit orders to complete a gap bracket.

        Called after the MKT order fills.
        """
        from ib_insync import Order
        from orders import make_stock_contract
        from time_utils import get_day_exit_time_pt

        try:
            contract = make_stock_contract(symbol)
            qualified = self.ib.qualifyContracts(contract)
            if not qualified:
                self.logger.error("[GAP] Failed to qualify contract for %s", symbol)
                return False

            # Determine exit action (opposite of entry)
            exit_action = "SELL" if direction == "LONG" else "BUY"

            # Create stop order
            stop_order = Order(
                action=exit_action,
                orderType="STP",
                totalQuantity=shares,
                auxPrice=stop_price,
                tif="GTC",
                transmit=False,
            )

            # Create timed exit order
            trade_date = now_pt().date()
            exit_time = get_day_exit_time_pt()

            # Format GAT time for IBKR (YYYYMMDD HH:MM:SS timezone)
            gat_str = f"{trade_date.strftime('%Y%m%d')} {exit_time.strftime('%H:%M:%S')} US/Pacific"

            timed_order = Order(
                action=exit_action,
                orderType="MKT",
                totalQuantity=shares,
                tif="DAY",
                goodAfterTime=gat_str,
                transmit=True,  # Transmit all
            )

            # Get order IDs
            stop_id = self.executor._get_next_order_id()
            timed_id = self.executor._get_next_order_id()

            stop_order.orderId = stop_id
            timed_order.orderId = timed_id

            # Link OCA so stop and timed cancel each other
            oca_group = f"GAP_{signal_type}_{symbol}_{strategy_id}_{parent_order_id}"
            stop_order.ocaGroup = oca_group
            stop_order.ocaType = 1  # Cancel remaining
            timed_order.ocaGroup = oca_group
            timed_order.ocaType = 1

            # Place orders
            self.ib.placeOrder(contract, stop_order)
            self.ib.placeOrder(contract, timed_order)

            self.logger.info(
                "[GAP] Stop/timed orders placed for %s: stop_id=%d (%.2f), timed_id=%d",
                symbol, stop_id, stop_price, timed_id,
            )

            # Register with fill tracker
            if self.fill_tracker is not None:
                self.fill_tracker.register_gap_bracket_completion(
                    parent_order_id=parent_order_id,
                    stop_order_id=stop_id,
                    timed_order_id=timed_id,
                    symbol=symbol,
                    strategy_id=strategy_id,
                    signal_type=signal_type,
                    direction=direction,
                    shares=shares,
                )

            return True

        except Exception as exc:
            self.logger.error(
                "[GAP] Error placing stop/timed for %s: %s", symbol, exc
            )
            return False

    def _mark_signal_triggered(
        self, symbol: str, strategy_id: str, runtimes: Dict[str, List]
    ) -> None:
        """Mark a signal as triggered in its runtime."""
        if symbol not in runtimes:
            return

        for rt in runtimes[symbol]:
            if rt.signal.strategy_id == strategy_id:
                rt.triggered = True
                break

    def _mark_signal_failed(
        self, symbol: str, strategy_id: str, runtimes: Dict[str, List]
    ) -> None:
        """Mark a signal as failed/skipped."""
        self._mark_signal_triggered(symbol, strategy_id, runtimes)

    # === Debug/Status ===

    def get_status(self) -> Dict[str, Any]:
        """Return current gap manager status."""
        pending = self.state_mgr.pending_gap_orders.get_all_pending()
        return {
            "gap_check_date": self._gap_check_date.isoformat() if self._gap_check_date else None,
            "pending_gap_orders": len(pending),
            "pending_symbols": [p["symbol"] for p in pending],
        }
