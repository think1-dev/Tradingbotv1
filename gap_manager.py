"""
gap_manager.py

Gap-at-Open Trade Manager for handling gap triggers at market open.

Gap Logic:
- Day LONG: Gap DOWN (open < entry) triggers MKT buy
- Day SHORT: Gap UP (open > entry) triggers MKT sell
- Swing MOMO: Gap UP (open > entry) triggers MKT buy
- Swing Pullback: Gap DOWN (open < entry) triggers MKT buy

Gap Criteria:
- Gap must fully cross the signal's entry price overnight
- Single check at 6:30 PT exactly (first tick after open)
- If gap condition NOT met, signal remains armed for normal LMT fill

Stop Recalculation:
- Uses fixed dollar stop distance from open price
- Stop distance preserved from original signal

Persistence:
- Stores prev_close per symbol (last RTH tick from previous day)
- Persisted in state.json under "gap_data" key
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass, asdict
from datetime import date, datetime, time
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Any, Tuple
from zoneinfo import ZoneInfo

from time_utils import now_pt, is_trading_day, get_week_ending_day, is_week_ending_day
import market_calendar as mc

if TYPE_CHECKING:
    from ib_insync import IB, Ticker
    from signals import DaySignal, SwingSignal
    from fill_tracker import FillTracker, FilledPosition
    from execution import OrderExecutor
    from state_manager import StateManager
    from cap_manager import CapManager
    from conflict_resolver import ConflictResolver
    from reentry_manager import ReentryManager

PT = ZoneInfo("America/Los_Angeles")
STATE_PATH = Path("logs") / "state.json"

# Default stop distances (can be overridden per strategy)
DEFAULT_DAY_STOP_DISTANCE = 0.50  # $0.50 fixed stop
DEFAULT_SWING_STOP_DISTANCE = 1.00  # $1.00 fixed stop


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
    stop_distance: float  # Fixed dollar stop distance


@dataclass
class PrevCloseData:
    """Previous close data for a symbol."""
    symbol: str
    prev_close: float
    close_date: str  # ISO format date
    updated_at: str  # ISO format datetime


class GapManager:
    """
    Manages gap-at-open trade detection and execution.

    Lifecycle:
    1. EOD: Store prev_close for each symbol (last RTH tick)
    2. Market Open (6:30 PT): Run gap detection
    3. For each signal, check if gap crosses entry level
    4. Execute gap MKT orders with recalculated stops
    5. Mark signals as triggered to prevent duplicate entries

    Integration:
    - Uses ConflictResolver before gap entries (creates re-entry candidates)
    - Uses CapManager for cap checks
    - Uses FillTracker for order registration
    - Coordinates with ReentryManager (both can execute simultaneously)
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

        # Previous close data: symbol -> PrevCloseData
        self.prev_closes: Dict[str, PrevCloseData] = {}

        # Track if gap check has run today
        self._gap_check_date: Optional[date] = None

        # Lock to prevent concurrent gap check execution
        self._gap_check_lock = threading.Lock()

        # Load persisted prev_close data
        self._load_prev_closes()

        self.logger.info(
            "[GAP] Initialized with %d prev_close entries.",
            len(self.prev_closes),
        )

    # === Public Interface ===

    def check_and_fetch_stale_prev_closes(self, symbols: List[str]) -> None:
        """
        Check if prev_close data is stale and fetch historical data if needed.

        Called at startup to ensure gap check has valid data.
        Data is considered stale if it's more than 1 trading day old.

        Args:
            symbols: List of symbols to check
        """
        today = now_pt().date()

        # Determine the previous trading day
        from market_calendar import get_previous_trading_day
        prev_trading_day = get_previous_trading_day(today)

        if prev_trading_day is None:
            self.logger.warning("[GAP] Could not determine previous trading day.")
            return

        stale_symbols = []
        for symbol in symbols:
            symbol = symbol.upper()
            data = self.prev_closes.get(symbol)

            if data is None:
                stale_symbols.append(symbol)
                continue

            # Check if data is from the previous trading day
            try:
                close_date = date.fromisoformat(data.close_date)
                if close_date < prev_trading_day:
                    stale_symbols.append(symbol)
            except (ValueError, TypeError):
                stale_symbols.append(symbol)

        if not stale_symbols:
            self.logger.info("[GAP] All prev_close data is current.")
            return

        self.logger.info(
            "[GAP] Found %d symbols with stale prev_close data, fetching historical: %s",
            len(stale_symbols),
            ", ".join(stale_symbols[:5]) + ("..." if len(stale_symbols) > 5 else ""),
        )

        # Fetch historical data for stale symbols
        for symbol in stale_symbols:
            self._fetch_historical_close(symbol, prev_trading_day)

        # Save updated data
        self._save_prev_closes()

    def _fetch_historical_close(self, symbol: str, target_date: date) -> None:
        """
        Fetch historical close price for a symbol from IBKR.

        Args:
            symbol: Stock symbol
            target_date: The date to fetch close price for
        """
        try:
            from ib_insync import Stock

            contract = Stock(symbol, "SMART", "USD")
            self.ib.qualifyContracts(contract)

            # Request 1 day of historical data
            bars = self.ib.reqHistoricalData(
                contract,
                endDateTime="",
                durationStr="2 D",
                barSizeSetting="1 day",
                whatToShow="TRADES",
                useRTH=True,
                formatDate=1,
            )

            if bars:
                # Get the most recent bar's close
                last_bar = bars[-1]
                close_price = last_bar.close
                bar_date = last_bar.date

                self.prev_closes[symbol.upper()] = PrevCloseData(
                    symbol=symbol.upper(),
                    prev_close=close_price,
                    close_date=str(bar_date)[:10] if isinstance(bar_date, datetime) else str(bar_date),
                    updated_at=now_pt().isoformat(),
                )

                self.logger.info(
                    "[GAP] Fetched historical close for %s: %.2f (date=%s)",
                    symbol,
                    close_price,
                    bar_date,
                )
            else:
                self.logger.warning(
                    "[GAP] No historical data returned for %s", symbol
                )

        except Exception as exc:
            self.logger.warning(
                "[GAP] Failed to fetch historical close for %s: %s",
                symbol,
                exc,
            )

    def store_prev_close(self, symbol: str, price: float) -> None:
        """
        Store the previous close price for a symbol.

        Called at EOD (end of RTH) with the last tick price.
        """
        today = now_pt().date()
        self.prev_closes[symbol.upper()] = PrevCloseData(
            symbol=symbol.upper(),
            prev_close=price,
            close_date=today.isoformat(),
            updated_at=now_pt().isoformat(),
        )
        self._save_prev_closes()

        self.logger.info(
            "[GAP] Stored prev_close for %s: %.2f (date=%s)",
            symbol,
            price,
            today,
        )

    def get_prev_close(self, symbol: str) -> Optional[float]:
        """Get the previous close price for a symbol."""
        data = self.prev_closes.get(symbol.upper())
        if data is None:
            return None
        return data.prev_close

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

        Uses a lock to prevent concurrent execution from multiple threads.
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

    def update_prev_close_from_ticker(self, symbol: str, ticker: "Ticker") -> None:
        """
        Update prev_close from a ticker at EOD.

        Called during RTH to track the last price. The final call at
        market close becomes the prev_close for the next day.
        """
        # Use last price, or bid/ask midpoint if last is unavailable
        price = None
        if ticker.last is not None and ticker.last > 0:
            price = ticker.last
        elif ticker.bid is not None and ticker.ask is not None:
            if ticker.bid > 0 and ticker.ask > 0:
                price = (ticker.bid + ticker.ask) / 2.0

        if price is not None:
            self.prev_closes[symbol.upper()] = PrevCloseData(
                symbol=symbol.upper(),
                prev_close=price,
                close_date=now_pt().date().isoformat(),
                updated_at=now_pt().isoformat(),
            )
            # Don't save on every tick - save at EOD

    def save_all_prev_closes(self) -> None:
        """Save all prev_close data to state.json. Call at EOD."""
        self._save_prev_closes()
        self.logger.info(
            "[GAP] Saved %d prev_close entries to state.",
            len(self.prev_closes),
        )

    # === Internal Methods ===

    def _build_day_candidates(
        self, signals: List["DaySignal"]
    ) -> List[GapCandidate]:
        """Build gap candidates from day signals using pre-calculated stop_distance."""
        candidates = []

        for sig in signals:
            direction = (getattr(sig, "direction", None) or "LONG").upper()
            entry = float(sig.entry_price)
            stop = float(sig.stop_price)

            # Use pre-calculated stop_distance from signal (persisted from CSV load)
            stop_distance = getattr(sig, "stop_distance", 0.0)
            if stop_distance <= 0:
                # Fallback: calculate if not pre-calculated (legacy signals)
                stop_distance = abs(entry - stop)
                self.logger.debug(
                    "[GAP] Fallback stop_distance calculation for %s: %.2f",
                    sig.symbol, stop_distance,
                )

            # Determine gap direction needed
            # Day LONG: needs gap DOWN (open < entry)
            # Day SHORT: needs gap UP (open > entry)
            if direction == "LONG":
                gap_direction = "DOWN"
            else:
                gap_direction = "UP"

            candidates.append(GapCandidate(
                symbol=sig.symbol.upper(),
                strategy_id=sig.strategy_id,
                signal_type="DAY",
                direction=direction,
                entry_price=entry,
                original_stop=stop,
                shares=sig.shares,
                gap_direction_needed=gap_direction,
                stop_distance=stop_distance if stop_distance > 0 else DEFAULT_DAY_STOP_DISTANCE,
            ))

        return candidates

    def _build_swing_candidates(
        self, signals: List["SwingSignal"]
    ) -> List[GapCandidate]:
        """Build gap candidates from swing signals using pre-calculated stop_distance."""
        candidates = []

        for sig in signals:
            entry = float(sig.entry_price)
            stop = float(sig.stop_price)
            strategy_lower = (sig.strategy_id or "").lower()

            # Use pre-calculated stop_distance from signal (persisted from CSV load)
            stop_distance = getattr(sig, "stop_distance", 0.0)
            if stop_distance <= 0:
                # Fallback: calculate if not pre-calculated (legacy signals)
                stop_distance = abs(entry - stop)
                self.logger.debug(
                    "[GAP] Fallback stop_distance calculation for %s: %.2f",
                    sig.symbol, stop_distance,
                )

            # Determine gap direction based on strategy type
            # Swing MOMO/Breakout: needs gap UP (open > entry)
            # Swing Pullback/MeanRev: needs gap DOWN (open < entry)
            if "momo" in strategy_lower or "breakout" in strategy_lower:
                gap_direction = "UP"
            else:
                gap_direction = "DOWN"

            candidates.append(GapCandidate(
                symbol=sig.symbol.upper(),
                strategy_id=sig.strategy_id,
                signal_type="SWING",
                direction="LONG",  # Swings are LONG-only
                entry_price=entry,
                original_stop=stop,
                shares=sig.shares,
                gap_direction_needed=gap_direction,
                stop_distance=stop_distance if stop_distance > 0 else DEFAULT_SWING_STOP_DISTANCE,
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

        Returns order ID if gap trade was placed, None otherwise.
        """
        symbol = candidate.symbol

        # Get open price (first tick)
        open_price = self._get_open_price(symbol)
        if open_price is None:
            self.logger.warning(
                "[GAP] No open price available for %s, skipping gap check.",
                symbol,
            )
            return None

        # Get prev_close
        prev_close = self.get_prev_close(symbol)
        if prev_close is None:
            self.logger.warning(
                "[GAP] No prev_close available for %s, skipping gap check.",
                symbol,
            )
            return None

        # Check gap condition
        gap_met, gap_percent = self._check_gap_condition(
            candidate, open_price, prev_close
        )

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
            # Day gap trades check both week and day blocks
            is_blocked, block_reason = self.state_mgr.blocked.is_blocked(
                symbol, candidate.strategy_id, today
            )
        else:
            # Swing gap trades only check week blocks
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

        # Calculate new stop based on open price
        new_stop = self._calculate_gap_stop(candidate, open_price)

        # Execute gap trade
        order_id = self._execute_gap_trade(candidate, open_price, new_stop, is_day)

        if order_id is not None:
            self.logger.info(
                "[GAP][TRIGGER] %s %s on %s - gap_%s=%.2f%%, entry=MKT@%.2f, stop=%.2f",
                candidate.signal_type,
                candidate.direction,
                symbol,
                candidate.gap_direction_needed.lower(),
                gap_percent,
                open_price,
                new_stop,
            )
            # Mark signal as triggered
            self._mark_signal_triggered(symbol, candidate.strategy_id, runtimes)

            # Link re-entry candidates to this gap Day order
            if is_day and reentry_candidate_ids and self.reentry_manager is not None:
                for cid in reentry_candidate_ids:
                    self.reentry_manager.link_day_trade(cid, order_id)
        else:
            self.logger.warning(
                "[GAP][FAIL] %s %s on %s - order placement failed",
                candidate.signal_type,
                candidate.direction,
                symbol,
            )
            # Gap trade failed - drop any orphaned re-entry candidates (both Day and Swing)
            if reentry_candidate_ids and self.reentry_manager is not None:
                for cid in reentry_candidate_ids:
                    try:
                        self.reentry_manager.drop_candidate(cid, "gap_trade_failed")
                    except Exception as exc:
                        self.logger.error(
                            "[GAP] Failed to drop candidate %s: %s", cid, exc
                        )

            # Block symbol+strategy for rest of week (swing re-entry) and day (day/gap entry)
            today = now_pt().date()
            self.state_mgr.blocked.block_for_week(symbol, candidate.strategy_id, today)
            self.state_mgr.blocked.block_for_day(symbol, candidate.strategy_id, today)
            self.logger.warning(
                "[GAP][BLOCKED] %s %s - gap trade failed, blocked for week and day",
                symbol,
                candidate.strategy_id,
            )

            self._mark_signal_failed(symbol, candidate.strategy_id, runtimes)

        return order_id

    def _check_gap_condition(
        self,
        candidate: GapCandidate,
        open_price: float,
        prev_close: float,
    ) -> Tuple[bool, float]:
        """
        Check if gap condition is met.

        Gap must fully cross entry level overnight.

        Returns (condition_met, gap_percent).
        """
        entry = candidate.entry_price
        gap_needed = candidate.gap_direction_needed

        # Validate prev_close to prevent division by zero or invalid values
        if prev_close is None or prev_close <= 0:
            self.logger.warning(
                "[GAP] Invalid prev_close (%.4f) for %s - skipping gap check",
                prev_close if prev_close is not None else 0,
                candidate.symbol,
            )
            return False, 0.0

        # Calculate gap percentage from prev_close
        gap_percent = ((open_price - prev_close) / prev_close) * 100.0

        if gap_needed == "DOWN":
            # Gap down: open must be AT or BELOW entry
            # AND prev_close must have been ABOVE entry (crossed overnight)
            condition = open_price <= entry and prev_close > entry
        else:  # UP
            # Gap up: open must be AT or ABOVE entry
            # AND prev_close must have been BELOW entry (crossed overnight)
            condition = open_price >= entry and prev_close < entry

        return condition, gap_percent

    def _get_open_price(self, symbol: str) -> Optional[float]:
        """Get the open price (first tick) for a symbol."""
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
        # Create a minimal signal-like object for cap check
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
        """Check swing trade caps (uses available slots, not reserved)."""
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

        # Create minimal signal for conflict check
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
                    # Look up the original signal
                    signal_key = f"{instr.symbol}_{instr.strategy_id}"
                    original_signal = self._swing_signals_by_key.get(signal_key)

                    if original_signal is not None:
                        # Create FilledPosition-like object for store_candidate
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
                                cid,
                                instr.symbol,
                                instr.strategy_id,
                            )
                        except Exception as exc:
                            self.logger.error(
                                "[GAP] Failed to create re-entry candidate for %s %s: %s",
                                instr.symbol,
                                instr.strategy_id,
                                exc,
                            )
                    else:
                        self.logger.warning(
                            "[GAP] Could not find original signal for SWING %s %s - no re-entry candidate created",
                            instr.symbol,
                            instr.strategy_id,
                        )

                success = self.executor.flatten_position_with_retry(instr)
                if not success:
                    # Clean up any candidates we created since flatten failed
                    for cid in reentry_candidate_ids:
                        try:
                            self.reentry_manager.drop_candidate(cid, "flatten_failed")
                        except Exception as exc:
                            self.logger.error(
                                "[GAP] Failed to drop candidate %s: %s", cid, exc
                            )
                    return False, []

        return True, reentry_candidate_ids

    def _calculate_gap_stop(
        self, candidate: GapCandidate, open_price: float
    ) -> float:
        """Calculate new stop price based on open price."""
        if candidate.direction == "LONG":
            # Stop below entry for LONG
            return open_price - candidate.stop_distance
        else:
            # Stop above entry for SHORT
            return open_price + candidate.stop_distance

    def _execute_gap_trade(
        self,
        candidate: GapCandidate,
        open_price: float,
        new_stop: float,
        is_day: bool,
    ) -> Optional[int]:
        """Execute the gap trade with MKT entry."""
        from orders import build_day_bracket, build_swing_bracket, link_bracket, make_stock_contract

        trade_date = now_pt().date()

        # Create a signal-like object with gap parameters
        class GapSignal:
            def __init__(self, symbol, strategy_id, entry, stop, shares, direction):
                self.symbol = symbol
                self.strategy_id = strategy_id
                self.entry_price = entry
                self.stop_price = stop
                self.shares = shares
                self.direction = direction
                self.trade_date = trade_date

        gap_sig = GapSignal(
            candidate.symbol,
            candidate.strategy_id,
            open_price,
            new_stop,
            candidate.shares,
            candidate.direction,
        )

        try:
            if is_day:
                # Build day bracket with MKT entry
                bracket = build_day_bracket(gap_sig, trade_date)
                # Override to MKT
                bracket.parent.orderType = "MKT"
                if hasattr(bracket.parent, 'lmtPrice'):
                    delattr(bracket.parent, 'lmtPrice')
            else:
                # Build swing bracket with MKT entry
                bracket = build_swing_bracket(gap_sig, trade_date, market_entry=True)

            # Link OCA
            tag = f"GAP_{candidate.signal_type}_{candidate.symbol}_{candidate.strategy_id}"
            link_bracket(bracket, oca_group=tag)

            # Get contract
            contract = make_stock_contract(candidate.symbol)
            qualified = self.ib.qualifyContracts(contract)
            if not qualified:
                self.logger.error("[GAP] Failed to qualify contract for %s", candidate.symbol)
                return None

            # Assign order IDs
            parent_id = self.executor._get_next_order_id()
            stop_id = self.executor._get_next_order_id()
            timed_id = self.executor._get_next_order_id()

            bracket.parent.orderId = parent_id
            bracket.stop.orderId = stop_id
            bracket.stop.parentId = parent_id
            bracket.timed.orderId = timed_id
            bracket.timed.parentId = parent_id

            # Transmit
            bracket.parent.transmit = False
            bracket.stop.transmit = False
            bracket.timed.transmit = True

            # Place orders
            self.ib.placeOrder(contract, bracket.parent)
            self.ib.placeOrder(contract, bracket.stop)
            self.ib.placeOrder(contract, bracket.timed)

            self.logger.info(
                "[GAP] Bracket placed: %s parent=%d stop=%d timed=%d",
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
                    trade_type=candidate.signal_type,
                    trade_date=trade_date,
                    side=candidate.direction,
                    qty=candidate.shares,
                    stop_order_id=stop_id,
                    timed_order_id=timed_id,
                )

            # Register with cap manager
            if is_day:
                self.cap_manager.register_day_entry(gap_sig)
            else:
                self.cap_manager.register_swing_entry(gap_sig)

            return parent_id

        except Exception as exc:
            self.logger.error(
                "[GAP] Error executing gap trade for %s: %s",
                candidate.symbol,
                exc,
            )
            return None

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
        # For failed gap trades, mark as triggered so normal logic doesn't retry
        self._mark_signal_triggered(symbol, strategy_id, runtimes)

    # === Persistence ===

    def _load_prev_closes(self) -> None:
        """Load prev_close data from state.json."""
        if not STATE_PATH.exists():
            return

        try:
            with STATE_PATH.open("r", encoding="utf-8") as f:
                data = json.load(f)

            gap_data = data.get("gap_data", {})
            prev_closes = gap_data.get("prev_closes", {})

            for symbol, pdata in prev_closes.items():
                self.prev_closes[symbol] = PrevCloseData(**pdata)

            self.logger.info(
                "[GAP] Loaded %d prev_close entries from state.",
                len(self.prev_closes),
            )
        except Exception as exc:
            self.logger.warning(
                "[GAP] Failed to load prev_close data: %s", exc
            )

    def _save_prev_closes(self) -> None:
        """Save prev_close data to state.json."""
        STATE_PATH.parent.mkdir(exist_ok=True)

        try:
            # Load existing state
            if STATE_PATH.exists():
                with STATE_PATH.open("r", encoding="utf-8") as f:
                    data = json.load(f)
            else:
                data = {}

            # Update gap_data section
            gap_data = data.get("gap_data", {})
            gap_data["prev_closes"] = {
                symbol: asdict(pdata)
                for symbol, pdata in self.prev_closes.items()
            }
            data["gap_data"] = gap_data

            with STATE_PATH.open("w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, sort_keys=True)

        except Exception as exc:
            self.logger.warning(
                "[GAP] Failed to save prev_close data: %s", exc
            )

    # === Debug/Status ===

    def get_status(self) -> Dict[str, Any]:
        """Return current gap manager status."""
        return {
            "prev_closes_count": len(self.prev_closes),
            "gap_check_date": self._gap_check_date.isoformat() if self._gap_check_date else None,
            "symbols_tracked": list(self.prev_closes.keys()),
        }
