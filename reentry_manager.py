"""
reentry_manager.py

Re-entry Manager Module for handling Weekly position re-entry after Day conflict exits.

When a Weekly LONG position is flattened due to a Day SHORT conflict, this module:
1. Stores the flattened position as a re-entry candidate
2. Monitors when the conflicting Day trade exits
3. Evaluates re-entry eligibility (current price >= original stop)
4. Executes re-entry with MKT entry if eligible

Re-entry candidates persist across bot restarts via state.json.

Key Rules:
- Re-entry only checks global SwingOpen capacity (not strategy 5/week cap)
- Re-entry uses MKT entry (not original limit)
- Re-entry preserves original stop price and timed exit date
- No re-entries past 12:55 PT on week-ending day
- Price below original stop → candidate dropped permanently
- Other failures (capacity, rejection) → re-check next day at market open
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
from dataclasses import dataclass, field, asdict
from datetime import date, datetime, time
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Any
from zoneinfo import ZoneInfo

import market_calendar as mc
from time_utils import now_pt, is_trading_day, get_week_ending_day, is_week_ending_day

if TYPE_CHECKING:
    from ib_insync import IB, Ticker
    from signals import SwingSignal
    from fill_tracker import FillTracker, FilledPosition
    from execution import OrderExecutor
    from state_manager import StateManager

PT = ZoneInfo("America/Los_Angeles")

# State file path (same as StateManager)
STATE_PATH = Path("logs") / "state.json"


@dataclass
class ReentryCandidate:
    """
    Represents a Weekly position that was flattened due to Day conflict
    and is eligible for re-entry when ALL blocking Day trades exit.

    Multiple Day trades can block the same re-entry candidate:
    - The original Day trade that caused the flatten
    - Any subsequent opposite-side Day trades on the same symbol
    """
    candidate_id: str  # Unique ID for this candidate
    symbol: str
    strategy_id: str
    original_stop: float
    original_entry: float
    original_qty: int
    original_timed_exit_date: str  # ISO format date
    original_signal_data: Dict[str, Any]  # Serialized SwingSignal
    blocking_day_orders: List[int] = field(default_factory=list)  # All Day orders blocking this re-entry
    created_at: str = ""  # ISO format datetime
    flatten_reason: str = "day_conflict"
    status: str = "pending"  # pending, linked, dropped, filled
    drop_reason: Optional[str] = None

    def __post_init__(self):
        if not self.created_at:
            self.created_at = now_pt().isoformat()
        # Ensure blocking_day_orders is a list (for deserialization compatibility)
        if self.blocking_day_orders is None:
            self.blocking_day_orders = []


class ReentryManager:
    """
    Manages re-entry of Weekly positions after Day conflict exits.

    Lifecycle:
    1. store_candidate() - Called when Weekly is flattened for Day conflict
    2. link_day_trade() - Called after Day bracket is placed, links order ID
    3. on_day_trade_exit() - Called when Day trade exits, triggers evaluation
    4. _evaluate_and_execute() - Checks eligibility and executes re-entry

    Persistence:
    - Candidates are stored in state.json under "reentry_candidates" key
    - Loaded on initialization, saved after every change
    """

    def __init__(
        self,
        ib: "IB",
        logger: logging.Logger,
        fill_tracker: "FillTracker",
        executor: "OrderExecutor",
        state_mgr: "StateManager",
    ) -> None:
        self.ib = ib
        self.logger = logger
        self.fill_tracker = fill_tracker
        self.executor = executor
        self.state_mgr = state_mgr

        # In-memory candidate storage: candidate_id -> ReentryCandidate
        self.candidates: Dict[str, ReentryCandidate] = {}

        # Mapping: day_order_id -> list of candidate_ids that this Day order blocks
        # One Day order can block multiple candidates (multiple SWINGs on same symbol)
        self.day_order_to_candidates: Dict[int, List[str]] = {}

        # Candidates pending EOD evaluation (for cancelled Day trades)
        self.pending_eod_evaluation: List[str] = []

        # Lock to prevent concurrent evaluation of the same candidate
        self._eval_lock = threading.Lock()

        # Load persisted candidates
        self._load_candidates()

        self.logger.info(
            "[REENTRY] Initialized with %d persisted candidates.",
            len(self.candidates),
        )

    # === Public Interface ===

    def store_candidate(
        self,
        flattened_position: "FilledPosition",
        original_signal: "SwingSignal",
    ) -> str:
        """
        Store a re-entry candidate when a Weekly is flattened due to Day conflict.

        Args:
            flattened_position: The FilledPosition that was flattened
            original_signal: The original SwingSignal for the position

        Returns:
            candidate_id: Unique identifier for this re-entry candidate
        """
        candidate_id = f"reentry_{flattened_position.symbol}_{flattened_position.strategy_id}_{now_pt().strftime('%Y%m%d%H%M%S')}"

        # Serialize signal data for persistence
        signal_data = {
            "symbol": original_signal.symbol,
            "strategy_id": original_signal.strategy_id,
            "entry_price": float(original_signal.entry_price),
            "stop_price": float(original_signal.stop_price),
            "shares": int(original_signal.shares),
            "trade_date": str(getattr(original_signal, "trade_date", now_pt().date())),
        }

        # Get the original timed exit date from the signal
        original_timed_exit = getattr(original_signal, "trade_date", now_pt().date())
        if isinstance(original_timed_exit, str):
            original_timed_exit = datetime.strptime(original_timed_exit, "%Y-%m-%d").date()
        week_end = get_week_ending_day(original_timed_exit)

        candidate = ReentryCandidate(
            candidate_id=candidate_id,
            symbol=flattened_position.symbol,
            strategy_id=flattened_position.strategy_id,
            original_stop=float(original_signal.stop_price),
            original_entry=float(original_signal.entry_price),
            original_qty=int(original_signal.shares),
            original_timed_exit_date=week_end.isoformat(),
            original_signal_data=signal_data,
            flatten_reason="day_conflict",
            status="pending",
        )

        # Reserve a swing slot FIRST (before saving candidate)
        # This ensures if there's a crash, we don't have a candidate without a slot
        today = now_pt().date()
        self.state_mgr.reserve_swing_slot(today)

        # Now save the candidate (slot is already reserved)
        self.candidates[candidate_id] = candidate
        self._save_candidates()

        self.logger.info(
            "[REENTRY] Stored candidate %s: %s %s qty=%d stop=%.2f exit_date=%s",
            candidate_id,
            candidate.symbol,
            candidate.strategy_id,
            candidate.original_qty,
            candidate.original_stop,
            candidate.original_timed_exit_date,
        )

        return candidate_id

    def link_day_trade(self, candidate_id: str, day_parent_order_id: int) -> None:
        """
        Link a Day trade order ID as a blocker for a re-entry candidate.

        Called after the Day bracket is placed that caused the flatten.
        A candidate can be blocked by multiple Day trades.
        """
        if candidate_id not in self.candidates:
            self.logger.warning(
                "[REENTRY] Cannot link - candidate %s not found.", candidate_id
            )
            return

        candidate = self.candidates[candidate_id]

        # Add to blocking list if not already present
        if day_parent_order_id not in candidate.blocking_day_orders:
            candidate.blocking_day_orders.append(day_parent_order_id)

        candidate.status = "linked"

        # Update reverse mapping (day order -> list of candidates it blocks)
        if day_parent_order_id not in self.day_order_to_candidates:
            self.day_order_to_candidates[day_parent_order_id] = []
        if candidate_id not in self.day_order_to_candidates[day_parent_order_id]:
            self.day_order_to_candidates[day_parent_order_id].append(candidate_id)

        self._save_candidates()

        self.logger.info(
            "[REENTRY] Linked candidate %s to Day order %d. Total blockers: %d",
            candidate_id,
            day_parent_order_id,
            len(candidate.blocking_day_orders),
        )

    def drop_candidate(self, candidate_id: str, reason: str) -> None:
        """
        Drop a candidate and release its reserved slot.

        Called when flatten fails or other error before Day bracket is placed.
        """
        if candidate_id not in self.candidates:
            self.logger.warning(
                "[REENTRY] Cannot drop - candidate %s not found.", candidate_id
            )
            return

        candidate = self.candidates[candidate_id]
        candidate.status = "dropped"
        candidate.drop_reason = reason

        # Release the reserved slot
        today = now_pt().date()
        self.state_mgr.release_swing_slot(today)

        # Clean up day_order_to_candidates mapping to prevent memory leak
        for day_order_id in candidate.blocking_day_orders:
            if day_order_id in self.day_order_to_candidates:
                if candidate_id in self.day_order_to_candidates[day_order_id]:
                    self.day_order_to_candidates[day_order_id].remove(candidate_id)
                # Clean up empty lists
                if not self.day_order_to_candidates[day_order_id]:
                    del self.day_order_to_candidates[day_order_id]

        # Remove from pending EOD evaluation if present
        if candidate_id in self.pending_eod_evaluation:
            self.pending_eod_evaluation.remove(candidate_id)

        self._save_candidates()

        self.logger.info(
            "[REENTRY] Dropped candidate %s: %s (reason=%s)",
            candidate_id,
            candidate.symbol,
            reason,
        )

    def add_blocker_for_symbol(self, symbol: str, day_parent_order_id: int, day_side: str) -> None:
        """
        Add a new Day trade as a blocker for any pending re-entry candidates on the same symbol.

        Called when a NEW Day trade opens on a symbol that has pending re-entry candidates.
        This ensures the re-entry waits for ALL opposite-side Day trades to exit.

        Args:
            symbol: The stock symbol
            day_parent_order_id: The new Day trade's parent order ID
            day_side: The Day trade's side ("LONG" or "SHORT")
        """
        symbol = symbol.upper()

        # Find all pending/linked candidates for this symbol that would conflict
        for candidate in self.candidates.values():
            if candidate.symbol != symbol:
                continue
            if candidate.status not in ("pending", "linked"):
                continue

            # Re-entry is always SWING LONG, so only opposite-side (SHORT) Day trades block it
            if day_side.upper() != "SHORT":
                continue

            # Add this Day trade as a blocker
            if day_parent_order_id not in candidate.blocking_day_orders:
                candidate.blocking_day_orders.append(day_parent_order_id)
                self.logger.info(
                    "[REENTRY] Added Day order %d as blocker for candidate %s (%s). Total blockers: %d",
                    day_parent_order_id,
                    candidate.candidate_id,
                    symbol,
                    len(candidate.blocking_day_orders),
                )

        # Update reverse mapping
        if day_parent_order_id not in self.day_order_to_candidates:
            self.day_order_to_candidates[day_parent_order_id] = []

        # Add all matching candidates to the reverse mapping
        for candidate in self.candidates.values():
            if candidate.symbol == symbol and candidate.status in ("pending", "linked"):
                if day_parent_order_id in candidate.blocking_day_orders:
                    if candidate.candidate_id not in self.day_order_to_candidates[day_parent_order_id]:
                        self.day_order_to_candidates[day_parent_order_id].append(candidate.candidate_id)

        self._save_candidates()

    def on_day_trade_exit(self, day_parent_order_id: int) -> None:
        """
        Called when a Day trade exits (stop, timed, or manual close).

        Removes this Day order from all candidate blocking lists.
        Evaluates candidates whose blocking list becomes empty.
        """
        # Get all candidates blocked by this Day order
        candidate_ids = self.day_order_to_candidates.get(day_parent_order_id, [])

        if not candidate_ids:
            # No re-entry candidates linked to this Day trade
            return

        self.logger.info(
            "[REENTRY] Day trade %d exited - checking %d linked candidates.",
            day_parent_order_id,
            len(candidate_ids),
        )

        # Process each candidate
        candidates_to_evaluate = []
        for candidate_id in list(candidate_ids):  # Copy list to avoid modification during iteration
            candidate = self.candidates.get(candidate_id)
            if candidate is None:
                continue

            # Remove this Day order from the blocking list
            if day_parent_order_id in candidate.blocking_day_orders:
                candidate.blocking_day_orders.remove(day_parent_order_id)
                self.logger.info(
                    "[REENTRY] Removed Day order %d from candidate %s blockers. Remaining: %d",
                    day_parent_order_id,
                    candidate_id,
                    len(candidate.blocking_day_orders),
                )

            # If no more blockers, candidate is ready for evaluation
            if len(candidate.blocking_day_orders) == 0 and candidate.status == "linked":
                candidates_to_evaluate.append(candidate)

        # Clean up reverse mapping
        if day_parent_order_id in self.day_order_to_candidates:
            del self.day_order_to_candidates[day_parent_order_id]

        self._save_candidates()

        # Evaluate candidates with no remaining blockers
        for candidate in candidates_to_evaluate:
            self.logger.info(
                "[REENTRY] All blockers cleared for %s (%s) - evaluating re-entry.",
                candidate.candidate_id,
                candidate.symbol,
            )
            self._evaluate_and_execute(candidate)

    def on_day_trade_cancelled(self, day_parent_order_id: int) -> None:
        """
        Called when a Day trade order is cancelled.

        Handles two scenarios:
        1. Filled position cancelled (stock halt) - SWING was flattened, Day opened then cancelled
        2. Unfilled order cancelled (limit not reached) - SWING was flattened but Day never opened

        In both cases, schedules re-entry evaluation for 12:59 PT EOD.
        """
        candidate_ids = self.day_order_to_candidates.get(day_parent_order_id, [])
        if not candidate_ids:
            return

        for candidate_id in candidate_ids:
            candidate = self.candidates.get(candidate_id)
            if candidate is None:
                continue

            self.logger.info(
                "[REENTRY] Day trade %d cancelled - scheduling EOD evaluation for %s (%s).",
                day_parent_order_id,
                candidate_id,
                candidate.symbol,
            )

            if candidate_id not in self.pending_eod_evaluation:
                self.pending_eod_evaluation.append(candidate_id)

    def on_market_open(self) -> None:
        """
        Called at market open (6:30 PT) for crash recovery of orphaned candidates.

        Handles candidates that are still "linked" at market open, which means
        the Day trade exited but the exit callback never fired (crash/bug).
        This is the ONE-SHOT evaluation for these orphaned candidates.

        Also drops expired candidates past their timed exit date.
        """
        today = now_pt().date()

        # Find candidates that need re-evaluation
        for candidate in list(self.candidates.values()):
            if candidate.status not in ("pending", "linked"):
                continue

            # Check if original timed exit date has passed
            exit_date = date.fromisoformat(candidate.original_timed_exit_date)
            if exit_date < today:
                self.logger.info(
                    "[REENTRY] Candidate %s expired (exit_date=%s < today=%s).",
                    candidate.candidate_id,
                    exit_date,
                    today,
                )
                candidate.status = "dropped"
                candidate.drop_reason = "expired_past_exit_date"
                # Release the reserved slot since candidate is dropped
                self.state_mgr.release_swing_slot(today)
                self._cleanup_candidate(candidate)
                self._save_candidates()
                continue

            # If linked to a Day trade that never filled/exited, evaluate now
            if candidate.status == "linked":
                self.logger.info(
                    "[REENTRY] Market open re-evaluation for %s (%s).",
                    candidate.candidate_id,
                    candidate.symbol,
                )
                self._evaluate_and_execute(candidate)

    def on_reentry_fill(self, order_id: int, trade_date: date) -> None:
        """
        Called by FillTracker when a re-entry order fills.

        Converts the reserved slot to an open position.
        """
        self.logger.info(
            "[REENTRY] Re-entry order %d filled, converting reserved slot to open",
            order_id,
        )
        self.state_mgr.convert_reserved_to_open(trade_date)

    def evaluate_eod_candidates(self) -> None:
        """
        Called at 12:59 PT to evaluate candidates whose Day trades were cancelled.
        """
        for candidate_id in list(self.pending_eod_evaluation):
            candidate = self.candidates.get(candidate_id)
            if candidate is None:
                self.pending_eod_evaluation.remove(candidate_id)
                continue

            self.logger.info(
                "[REENTRY] EOD evaluation for cancelled Day trade: %s (%s).",
                candidate_id,
                candidate.symbol,
            )

            try:
                self._evaluate_and_execute(candidate)
            except Exception as exc:
                self.logger.error(
                    "[REENTRY] Exception during EOD evaluation for %s: %s",
                    candidate_id, exc,
                )
            finally:
                # Always remove from pending to avoid infinite retry loop
                self.pending_eod_evaluation.remove(candidate_id)

    # === Internal Methods ===

    def _evaluate_and_execute(self, candidate: ReentryCandidate) -> bool:
        """
        Evaluate re-entry eligibility and execute if eligible.

        Returns True if re-entry was executed successfully.

        Uses a lock to prevent concurrent evaluation of the same candidate
        from multiple call sites (on_day_trade_exit, evaluate_eod_candidates).
        """
        with self._eval_lock:
            # Guard against double evaluation
            if candidate.status in ("filled", "dropped"):
                self.logger.warning(
                    "[REENTRY] Candidate %s already %s, skipping evaluation.",
                    candidate.candidate_id,
                    candidate.status,
                )
                return False

            now = now_pt()
            today = now.date()

            # Check if symbol+strategy is blocked for the week (bracket/gap failure)
            is_blocked, block_reason = self.state_mgr.blocked.is_blocked(
                candidate.symbol, candidate.strategy_id, today, check_week=True, check_day=False
            )
            if is_blocked:
                self.logger.info(
                    "[REENTRY][SKIP] %s %s - %s",
                    candidate.symbol,
                    candidate.strategy_id,
                    block_reason,
                )
                candidate.status = "dropped"
                candidate.drop_reason = block_reason
                # Release the reserved slot since candidate is dropped
                self.state_mgr.release_swing_slot(today)
                self._cleanup_candidate(candidate)
                self._save_candidates()
                return False

            # Check if we're past the weekly cutoff (12:55 PT on week-ending day)
            if is_week_ending_day(today):
                cutoff_time = mc.get_day_stop_time_pt(today)
                if now.time() >= cutoff_time:
                    self.logger.info(
                        "[REENTRY][SKIP] %s - past weekly cutoff (%s) on week-ending day.",
                        candidate.symbol,
                        cutoff_time,
                    )
                    candidate.status = "dropped"
                    candidate.drop_reason = "past_weekly_cutoff"
                    # Release the reserved slot since candidate is dropped
                    self.state_mgr.release_swing_slot(today)
                    self._cleanup_candidate(candidate)
                    self._save_candidates()
                    return False

            # Check if original timed exit date has passed
            exit_date = date.fromisoformat(candidate.original_timed_exit_date)
            if exit_date < today:
                self.logger.info(
                    "[REENTRY][SKIP] %s - past original exit date (%s).",
                    candidate.symbol,
                    exit_date,
                )
                candidate.status = "dropped"
                candidate.drop_reason = "past_exit_date"
                # Release the reserved slot since candidate is dropped
                self.state_mgr.release_swing_slot(today)
                self._cleanup_candidate(candidate)
                self._save_candidates()
                return False

            # Get current mid-price
            mid_price = self._get_mid_price(candidate.symbol)
            if mid_price is None:
                self.logger.warning(
                    "[REENTRY][FAIL] %s - unable to get mid-price, dropping candidate.",
                    candidate.symbol,
                )
                candidate.status = "dropped"
                candidate.drop_reason = "mid_price_unavailable"
                # Release the reserved slot since candidate is dropped
                self.state_mgr.release_swing_slot(today)
                self._cleanup_candidate(candidate)
                self._save_candidates()
                return False

            # Check price vs original stop
            if mid_price < candidate.original_stop:
                self.logger.info(
                    "[REENTRY][SKIP] Weekly LONG on %s not re-entered - price below original stop (mid=%.2f, orig_stop=%.2f)",
                    candidate.symbol,
                    mid_price,
                    candidate.original_stop,
                )
                candidate.status = "dropped"
                candidate.drop_reason = f"price_below_stop (mid={mid_price:.2f} < stop={candidate.original_stop:.2f})"
                # Release the reserved slot since candidate is dropped
                self.state_mgr.release_swing_slot(today)
                self._cleanup_candidate(candidate)
                self._save_candidates()
                return False

            # Check global SwingOpen capacity only (should not fail with reserved slots)
            can_open, reason = self._can_reentry_swing()
            if not can_open:
                self.logger.warning(
                    "[REENTRY][FAIL] %s - capacity blocked: %s (reserved slot should have prevented this), dropping candidate.",
                    candidate.symbol,
                    reason,
                )
                candidate.status = "dropped"
                candidate.drop_reason = f"capacity_blocked ({reason})"
                # Release the reserved slot since candidate is dropped
                self.state_mgr.release_swing_slot(today)
                self._cleanup_candidate(candidate)
                self._save_candidates()
                return False

            # Execute re-entry
            self.logger.info(
                "[REENTRY] Re-entering Weekly LONG on %s after Day SHORT exit (mid=%.2f, orig_stop=%.2f)",
                candidate.symbol,
                mid_price,
                candidate.original_stop,
            )

            success = self._execute_reentry(candidate)

            if success:
                candidate.status = "filled"
                self._cleanup_candidate(candidate)
            else:
                self.logger.warning(
                    "[REENTRY][FAIL] Failed to execute re-entry for %s, dropping candidate.",
                    candidate.symbol,
                )
                candidate.status = "dropped"
                candidate.drop_reason = "execution_failed"
                # Release the reserved slot since candidate is dropped
                self.state_mgr.release_swing_slot(now_pt().date())
                self._cleanup_candidate(candidate)

            self._save_candidates()
            return success

    def _execute_reentry(self, candidate: ReentryCandidate) -> bool:
        """
        Execute the re-entry bracket placement.

        Uses MKT entry with original stop price and timed exit date.
        """
        from orders import build_swing_bracket, link_bracket
        from signals import SwingSignal

        # Reconstruct signal from stored data
        signal_data = candidate.original_signal_data
        signal = SwingSignal(
            symbol=signal_data["symbol"],
            strategy_id=signal_data["strategy_id"],
            entry_price=signal_data["entry_price"],
            stop_price=candidate.original_stop,  # Use original stop
            shares=candidate.original_qty,
        )
        signal.trade_date = date.fromisoformat(candidate.original_timed_exit_date)

        # Build bracket with market entry
        trade_date = now_pt().date()
        bracket = build_swing_bracket(signal, trade_date, market_entry=True)

        # Override the stop GTD and timed GAT to use original exit date
        exit_date = date.fromisoformat(candidate.original_timed_exit_date)
        stop_time = mc.get_day_stop_time_pt(exit_date)
        stop_gtd = datetime(
            exit_date.year, exit_date.month, exit_date.day,
            stop_time.hour, stop_time.minute, 0
        ).strftime("%Y%m%d %H:%M:%S")
        bracket.stop.goodTillDate = stop_gtd

        timed_gat = datetime(
            exit_date.year, exit_date.month, exit_date.day,
            6, 30, 0
        ).strftime("%Y%m%d %H:%M:%S")
        bracket.timed.goodAfterTime = timed_gat

        # Link OCA
        tag = f"REENTRY_{candidate.symbol}_{candidate.strategy_id}"
        link_bracket(bracket, oca_group=tag)

        # Place orders using executor's internal method
        try:
            # Get order IDs
            parent_id = self.executor._get_next_order_id()
            stop_id = self.executor._get_next_order_id()
            timed_id = self.executor._get_next_order_id()

            bracket.parent.orderId = parent_id
            bracket.stop.orderId = stop_id
            bracket.stop.parentId = parent_id
            bracket.timed.orderId = timed_id
            bracket.timed.parentId = parent_id

            # Qualify contract
            qualified = self.ib.qualifyContracts(bracket.contract)
            if not qualified:
                self.logger.error(
                    "[REENTRY] Failed to qualify contract for %s.", candidate.symbol
                )
                return False

            # Place orders
            self.ib.placeOrder(bracket.contract, bracket.parent)
            self.ib.placeOrder(bracket.contract, bracket.stop)
            self.ib.placeOrder(bracket.contract, bracket.timed)

            self.logger.info(
                "[REENTRY] Re-entry bracket placed for %s: parent=%d, stop=%d, timed=%d",
                candidate.symbol,
                parent_id,
                stop_id,
                timed_id,
            )

            # Register with fill tracker
            if self.fill_tracker is not None:
                self.fill_tracker.register_pending_order(
                    order_id=parent_id,
                    strategy_id=candidate.strategy_id,
                    symbol=candidate.symbol,
                    trade_type="SWING",
                    trade_date=trade_date,
                    side="LONG",
                    qty=candidate.original_qty,
                    stop_order_id=stop_id,
                    timed_order_id=timed_id,
                )
                # Register for slot conversion when order fills
                # (This ensures convert_reserved_to_open is only called on fill, not placement)
                self.fill_tracker.register_pending_reentry_conversion(parent_id, trade_date)

            return True

        except Exception as exc:
            self.logger.error(
                "[REENTRY] Error placing re-entry bracket for %s: %s",
                candidate.symbol,
                exc,
            )
            return False

    def _can_reentry_swing(self) -> tuple:
        """
        Check if re-entry is allowed using its reserved slot.

        Re-entry uses RESERVED slots (not available slots).
        The slot was already reserved when the candidate was created.

        Returns (True, None) if allowed, (False, reason) if blocked.
        """
        today = now_pt().date()
        return self.state_mgr.can_reentry_swing(today)

    def _get_mid_price(self, symbol: str) -> Optional[float]:
        """
        Get the current mid-price (bid/ask midpoint) for a symbol.

        Returns None if unable to get a valid price.
        """
        try:
            # Try to get ticker from strategy engine's cached tickers
            # If not available, request snapshot
            from ib_insync import Stock

            contract = Stock(symbol, "SMART", "USD")
            qualified = self.ib.qualifyContracts(contract)
            if not qualified:
                return None

            ticker = self.ib.reqMktData(contract, "", True, False)
            self.ib.sleep(0.5)  # Wait for data

            bid = ticker.bid
            ask = ticker.ask

            # Clean prices
            if bid is None or ask is None:
                return None
            if bid <= 0 or ask <= 0:
                return None

            mid = (bid + ask) / 2.0
            return mid

        except Exception as exc:
            self.logger.warning(
                "[REENTRY] Error getting mid-price for %s: %s", symbol, exc
            )
            return None

    def _cleanup_candidate(self, candidate: ReentryCandidate) -> None:
        """
        Remove a candidate from tracking after it's been filled or dropped.
        """
        # Remove from all blocking day order mappings
        for day_order_id in candidate.blocking_day_orders:
            if day_order_id in self.day_order_to_candidates:
                if candidate.candidate_id in self.day_order_to_candidates[day_order_id]:
                    self.day_order_to_candidates[day_order_id].remove(candidate.candidate_id)
                # Clean up empty lists
                if not self.day_order_to_candidates[day_order_id]:
                    del self.day_order_to_candidates[day_order_id]

        if candidate.candidate_id in self.pending_eod_evaluation:
            self.pending_eod_evaluation.remove(candidate.candidate_id)

        # Remove from in-memory dict to prevent memory leak
        if candidate.candidate_id in self.candidates:
            del self.candidates[candidate.candidate_id]

    # === Persistence ===

    def _load_candidates(self) -> None:
        """
        Load re-entry candidates from state.json.

        Handles migration from old format (conflicting_day_order_id)
        to new format (blocking_day_orders list).
        """
        if not STATE_PATH.exists():
            return

        try:
            with STATE_PATH.open("r", encoding="utf-8") as f:
                data = json.load(f)

            candidates_data = data.get("reentry_candidates", {})
            for cid, cdata in candidates_data.items():
                # Handle migration from old format
                if "conflicting_day_order_id" in cdata and "blocking_day_orders" not in cdata:
                    old_id = cdata.pop("conflicting_day_order_id")
                    cdata["blocking_day_orders"] = [old_id] if old_id is not None else []

                # Remove old field if present (backwards compatibility)
                cdata.pop("conflicting_day_order_id", None)

                candidate = ReentryCandidate(**cdata)
                self.candidates[cid] = candidate

                # Rebuild day_order_to_candidates mapping (List per Day order)
                for day_order_id in candidate.blocking_day_orders:
                    if day_order_id not in self.day_order_to_candidates:
                        self.day_order_to_candidates[day_order_id] = []
                    if cid not in self.day_order_to_candidates[day_order_id]:
                        self.day_order_to_candidates[day_order_id].append(cid)

            self.logger.info(
                "[REENTRY] Loaded %d candidates from state.", len(self.candidates)
            )

            # Verify slot reservation counts match active candidates
            self._sync_slot_reservations()

        except Exception as exc:
            self.logger.warning(
                "[REENTRY] Failed to load candidates from state: %s", exc
            )

    def _sync_slot_reservations(self) -> None:
        """
        Sync SwingReserved slot counts with actual active candidates.

        This handles crash recovery where slots and candidates may be out of sync.
        """
        today = now_pt().date()

        # Count active candidates (pending or linked status)
        active_count = sum(
            1 for c in self.candidates.values()
            if c.status in ("pending", "linked")
        )

        # Get current reserved count from state
        current_reserved = self.state_mgr.get_swing_reserved(today)

        if active_count != current_reserved:
            self.logger.warning(
                "[REENTRY] Slot reservation mismatch detected: active_candidates=%d, reserved_slots=%d. Syncing...",
                active_count,
                current_reserved,
            )

            # Adjust the reserved count to match active candidates
            diff = active_count - current_reserved
            if diff > 0:
                # Need to reserve more slots
                for _ in range(diff):
                    self.state_mgr.reserve_swing_slot(today)
            elif diff < 0:
                # Need to release slots
                for _ in range(-diff):
                    self.state_mgr.release_swing_slot(today)

            self.logger.info(
                "[REENTRY] Slot reservations synced. Now reserved=%d",
                self.state_mgr.get_swing_reserved(today),
            )

    def _save_candidates(self) -> None:
        """
        Save re-entry candidates to state.json using atomic write pattern.
        """
        STATE_PATH.parent.mkdir(exist_ok=True)

        try:
            # Load existing state
            if STATE_PATH.exists():
                with STATE_PATH.open("r", encoding="utf-8") as f:
                    data = json.load(f)
            else:
                data = {}

            # Update reentry_candidates section
            candidates_data = {}
            for cid, candidate in self.candidates.items():
                # Only persist active candidates
                if candidate.status in ("pending", "linked"):
                    candidates_data[cid] = asdict(candidate)

            data["reentry_candidates"] = candidates_data

            # Atomic write: write to temp file, then rename
            fd, tmp_path = tempfile.mkstemp(dir=str(STATE_PATH.parent), suffix=".tmp")
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2, sort_keys=True)
                os.replace(tmp_path, str(STATE_PATH))
            except Exception:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                raise

        except Exception as exc:
            self.logger.warning(
                "[REENTRY] Failed to save candidates to state: %s", exc
            )

    # === Debug/Status ===

    def get_status(self) -> Dict[str, Any]:
        """
        Return current status of re-entry candidates.
        """
        return {
            "total_candidates": len(self.candidates),
            "pending": sum(1 for c in self.candidates.values() if c.status == "pending"),
            "linked": sum(1 for c in self.candidates.values() if c.status == "linked"),
            "filled": sum(1 for c in self.candidates.values() if c.status == "filled"),
            "dropped": sum(1 for c in self.candidates.values() if c.status == "dropped"),
            "pending_eod": len(self.pending_eod_evaluation),
            "candidates": [
                {
                    "id": c.candidate_id,
                    "symbol": c.symbol,
                    "status": c.status,
                    "stop": c.original_stop,
                    "qty": c.original_qty,
                }
                for c in self.candidates.values()
            ],
        }
