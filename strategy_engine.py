from __future__ import annotations

import threading
from dataclasses import dataclass
from datetime import datetime, date, time
from math import isnan
from typing import Dict, List, Optional

from ib_insync import IB, Ticker, Stock, Contract

from time_utils import now_pt, is_rth, today_pt_date_str
from state_manager import StateManager
from signals import DaySignal, SwingSignal
from execution import OrderExecutor
from conflict_resolver import ConflictResolver
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from reentry_manager import ReentryManager
    from gap_manager import GapManager


@dataclass
class DayRuntime:
    signal: DaySignal
    triggered: bool = False
    triggered_date: Optional[date] = None  # Date when triggered, for auto-reset


@dataclass
class SwingRuntime:
    signal: SwingSignal
    triggered: bool = False
    triggered_date: Optional[date] = None  # Date when triggered, for auto-reset


def _clean_price(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    try:
        if isnan(value):
            return None
    except TypeError:
        return None
    if value <= 0:
        return None
    return value


class StrategyEngine:
    def __init__(
        self,
        ib: IB,
        logger,
        state_mgr: StateManager,
        cap_manager,
        executor: OrderExecutor,
        day_signals: List[DaySignal],
        swing_signals: List[SwingSignal],
        enable_day_trading: bool,
        enable_swing_trading: bool,
        conflict_resolver: Optional[ConflictResolver] = None,
        reentry_manager: Optional["ReentryManager"] = None,
        gap_manager: Optional["GapManager"] = None,
    ) -> None:
        self.ib = ib
        self.logger = logger
        self.state_mgr = state_mgr
        self.cap_manager = cap_manager
        self.executor = executor
        self.enable_day_trading = enable_day_trading
        self.enable_swing_trading = enable_swing_trading
        self.conflict_resolver = conflict_resolver
        self.reentry_manager = reentry_manager
        self.gap_manager = gap_manager

        # Store original signals for gap check
        self._day_signals = day_signals
        self._swing_signals = swing_signals

        # Track signals by strategy for re-entry candidate creation
        self._swing_signals_by_key: Dict[str, SwingSignal] = {}
        for sig in swing_signals:
            key = f"{sig.symbol}_{sig.strategy_id}"
            self._swing_signals_by_key[key] = sig

        self.day_by_symbol: Dict[str, List[DayRuntime]] = {}
        self.swing_by_symbol: Dict[str, List[SwingRuntime]] = {}
        self.contracts: Dict[str, Contract] = {}
        self.tickers: Dict[str, Ticker] = {}

        # Track if gap check has run today (persisted to prevent duplicates on restart)
        self._gap_check_done: bool = False
        self._gap_check_date: Optional[date] = self._load_gap_check_date()
        self._gap_check_lock = threading.Lock()

        # Track if EOD prev_close save has run today
        self._eod_save_done: bool = False
        self._eod_save_date: Optional[date] = None

        # Track if late-day re-check has run today (12:59 PT final re-entry eval)
        self._late_recheck_done: bool = False
        self._late_recheck_date: Optional[date] = None

        for sig in day_signals:
            self.day_by_symbol.setdefault(sig.symbol, []).append(DayRuntime(sig))
        for sig in swing_signals:
            self.swing_by_symbol.setdefault(sig.symbol, []).append(SwingRuntime(sig))

    # ---------- startup / market data ---------- #

    def start(self) -> None:
        all_symbols = set(self.day_by_symbol.keys()) | set(
            self.swing_by_symbol.keys()
        )
        if not all_symbols:
            self.logger.info("[MD] No symbols to stream (no Day or Swing signals).")
            return

        self.logger.info("[MD] Subscribing market data for %d symbols.", len(all_symbols))

        subscribed: List[str] = []
        skipped: List[str] = []

        for sym in sorted(all_symbols):
            base = Stock(sym, "SMART", "USD")
            try:
                qualified_list = self.ib.qualifyContracts(base)
            except Exception as exc:
                self.logger.warning(
                    "[MD] Skipping %s: qualifyContracts() raised %s", sym, exc
                )
                skipped.append(sym)
                continue

            if not qualified_list:
                self.logger.warning(
                    "[MD] Skipping %s: qualifyContracts() returned no contracts "
                    "(unknown symbol or no security definition).",
                    sym,
                )
                skipped.append(sym)
                continue

            contract = qualified_list[0]
            self.contracts[sym] = contract

            try:
                ticker: Ticker = self.ib.reqMktData(contract, "", False, False)
            except Exception as exc:
                self.logger.warning(
                    "[MD] Skipping %s: reqMktData() failed: %s", sym, exc
                )
                skipped.append(sym)
                continue

            self.tickers[sym] = ticker

            def _on_update(t: Ticker, symbol=sym):
                self.on_ticker(symbol, t)

            ticker.updateEvent += _on_update
            subscribed.append(sym)

        for sym, contract in self.contracts.items():
            for rt in self.day_by_symbol.get(sym, []):
                setattr(rt.signal, "contract", contract)
            for rt in self.swing_by_symbol.get(sym, []):
                setattr(rt.signal, "contract", contract)

        if subscribed:
            self.logger.info(
                "[MD] Market data subscriptions active for: %s", ", ".join(subscribed)
            )
        else:
            self.logger.info("[MD] No subscriptions active (all symbols skipped).")

        if skipped:
            self.logger.info(
                "[MD] Symbols skipped due to contract issues: %s", ", ".join(skipped)
            )

        mode = (
            "LIVE"
            if (self.enable_day_trading or self.enable_swing_trading)
            else "log-only"
        )
        self.logger.info(
            "[MD] Engine running in '%s' mode (day_trading=%s, swing_trading=%s).",
            mode,
            self.enable_day_trading,
            self.enable_swing_trading,
        )

    # ---------- signal updates (for rollover) ---------- #

    def update_day_signals(self, new_signals: List[DaySignal]) -> None:
        """
        Update day signals after rollover.

        Clears old day signals and loads new ones. For new symbols that weren't
        previously subscribed, subscribes to market data.
        """
        old_symbols = set(self.day_by_symbol.keys())

        # Clear old day signals
        self._day_signals = new_signals
        self.day_by_symbol.clear()

        # Load new signals
        for sig in new_signals:
            self.day_by_symbol.setdefault(sig.symbol, []).append(DayRuntime(sig))

        new_symbols = set(self.day_by_symbol.keys())

        # Subscribe to new symbols that weren't already subscribed
        symbols_to_subscribe = new_symbols - set(self.contracts.keys())
        if symbols_to_subscribe:
            self._subscribe_new_symbols(symbols_to_subscribe)

        # Attach contracts to new signals
        for sym, contract in self.contracts.items():
            for rt in self.day_by_symbol.get(sym, []):
                setattr(rt.signal, "contract", contract)

        self.logger.info(
            "[ENGINE] Updated day signals: %d signals, %d symbols (was %d symbols)",
            len(new_signals), len(new_symbols), len(old_symbols),
        )

    def update_swing_signals(self, new_signals: List[SwingSignal]) -> None:
        """
        Update swing signals after rollover.

        Clears old swing signals and loads new ones. For new symbols that weren't
        previously subscribed, subscribes to market data. Also updates the
        swing_signals_by_key lookup used by GapManager.
        """
        old_symbols = set(self.swing_by_symbol.keys())

        # Clear old swing signals
        self._swing_signals = new_signals
        self.swing_by_symbol.clear()
        self._swing_signals_by_key.clear()

        # Load new signals
        for sig in new_signals:
            self.swing_by_symbol.setdefault(sig.symbol, []).append(SwingRuntime(sig))
            key = f"{sig.symbol}_{sig.strategy_id}"
            self._swing_signals_by_key[key] = sig

        new_symbols = set(self.swing_by_symbol.keys())

        # Subscribe to new symbols that weren't already subscribed
        symbols_to_subscribe = new_symbols - set(self.contracts.keys())
        if symbols_to_subscribe:
            self._subscribe_new_symbols(symbols_to_subscribe)

        # Attach contracts to new signals
        for sym, contract in self.contracts.items():
            for rt in self.swing_by_symbol.get(sym, []):
                setattr(rt.signal, "contract", contract)

        # Update GapManager's swing signals lookup if available
        if hasattr(self, 'gap_manager') and self.gap_manager is not None:
            self.gap_manager.swing_signals_by_key = self._swing_signals_by_key

        self.logger.info(
            "[ENGINE] Updated swing signals: %d signals, %d symbols (was %d symbols)",
            len(new_signals), len(new_symbols), len(old_symbols),
        )

    def _subscribe_new_symbols(self, symbols: set) -> None:
        """Subscribe to market data for new symbols."""
        for sym in sorted(symbols):
            if sym in self.contracts:
                continue  # Already subscribed

            base = Stock(sym, "SMART", "USD")
            try:
                qualified_list = self.ib.qualifyContracts(base)
            except Exception as exc:
                self.logger.warning(
                    "[MD] Rollover: Skipping %s: qualifyContracts() raised %s", sym, exc
                )
                continue

            if not qualified_list:
                self.logger.warning(
                    "[MD] Rollover: Skipping %s: no contract definition", sym
                )
                continue

            contract = qualified_list[0]
            self.contracts[sym] = contract

            try:
                ticker: Ticker = self.ib.reqMktData(contract, "", False, False)
            except Exception as exc:
                self.logger.warning(
                    "[MD] Rollover: Failed to subscribe %s: %s", sym, exc
                )
                continue

            self.tickers[sym] = ticker

            def _on_update(t: Ticker, symbol=sym):
                self.on_ticker(symbol, t)

            ticker.updateEvent += _on_update
            self.logger.info("[MD] Rollover: Subscribed to new symbol %s", sym)

    def resubscribe_all(self) -> None:
        """
        Re-subscribe to market data for all symbols after reconnection.

        Called by ConnectionManager after successfully reconnecting to IBKR.
        Clears old tickers and creates fresh subscriptions.
        """
        self.logger.info("[MD] Re-subscribing to market data after reconnection...")

        # Get all symbols we should be subscribed to
        all_symbols = set(self.day_by_symbol.keys()) | set(self.swing_by_symbol.keys())

        if not all_symbols:
            self.logger.info("[MD] No symbols to re-subscribe (no signals loaded).")
            return

        # Clear old tickers (they're stale after disconnect)
        self.tickers.clear()

        subscribed = []
        failed = []

        for sym in sorted(all_symbols):
            # Use existing contract if we have it
            contract = self.contracts.get(sym)
            if contract is None:
                # Try to qualify the contract
                base = Stock(sym, "SMART", "USD")
                try:
                    qualified_list = self.ib.qualifyContracts(base)
                    if qualified_list:
                        contract = qualified_list[0]
                        self.contracts[sym] = contract
                except Exception as exc:
                    self.logger.warning(
                        "[MD] Reconnect: Failed to qualify %s: %s", sym, exc
                    )
                    failed.append(sym)
                    continue

            if contract is None:
                failed.append(sym)
                continue

            try:
                ticker: Ticker = self.ib.reqMktData(contract, "", False, False)

                def _on_update(t: Ticker, symbol=sym):
                    self.on_ticker(symbol, t)

                ticker.updateEvent += _on_update
                self.tickers[sym] = ticker
                subscribed.append(sym)

            except Exception as exc:
                self.logger.warning(
                    "[MD] Reconnect: Failed to subscribe %s: %s", sym, exc
                )
                failed.append(sym)

        self.logger.info(
            "[MD] Re-subscribed to %d symbols after reconnection. Failed: %d",
            len(subscribed), len(failed),
        )

        if failed:
            self.logger.warning(
                "[MD] Symbols that failed to re-subscribe: %s", ", ".join(failed)
            )

    # ---------- gap-at-open check ---------- #

    def _load_gap_check_date(self) -> Optional[date]:
        """Load persisted gap check date from state to prevent duplicate gap checks on restart."""
        try:
            gap_data = self.state_mgr.state.get("gap_check", {})
            date_str = gap_data.get("last_check_date")
            if date_str:
                return date.fromisoformat(date_str)
        except Exception as exc:
            self.logger.warning("[ENGINE] Failed to load gap_check_date: %s", exc)
        return None

    def _save_gap_check_date(self, d: date) -> None:
        """Persist gap check date to state to prevent duplicate gap checks on restart."""
        try:
            if "gap_check" not in self.state_mgr.state:
                self.state_mgr.state["gap_check"] = {}
            self.state_mgr.state["gap_check"]["last_check_date"] = d.isoformat()
            self.state_mgr._save()
        except Exception as exc:
            self.logger.warning("[ENGINE] Failed to save gap_check_date: %s", exc)

    def run_market_open_gap_check(self) -> Dict[str, List[int]]:
        """
        Run gap check at market open (6:30 PT).

        Should be called once at the start of RTH.
        Returns dict with "day_orders" and "swing_orders" lists.

        Uses a lock to prevent concurrent execution and persists state
        to prevent duplicate gap checks on bot restart.
        """
        with self._gap_check_lock:
            today = now_pt().date()

            # Only run once per day
            if self._gap_check_date == today:
                self.logger.info("[ENGINE] Gap check already ran today, skipping.")
                return {"day_orders": [], "swing_orders": []}

            if self.gap_manager is None:
                self.logger.info("[ENGINE] No gap manager configured, skipping gap check.")
                return {"day_orders": [], "swing_orders": []}

            if not (self.enable_day_trading or self.enable_swing_trading):
                self.logger.info("[ENGINE] Trading disabled, skipping gap check.")
                return {"day_orders": [], "swing_orders": []}

            self._gap_check_date = today
            self._gap_check_done = True
            self._save_gap_check_date(today)

            self.logger.info("[ENGINE] Running market open gap check for %s", today)

            # Run gap check via gap manager
            results = self.gap_manager.run_gap_check(
                day_signals=self._day_signals,
                swing_signals=self._swing_signals,
                day_runtimes=self.day_by_symbol,
                swing_runtimes=self.swing_by_symbol,
            )

            self.logger.info(
                "[ENGINE] Gap check complete: %d day orders, %d swing orders",
                len(results.get("day_orders", [])),
                len(results.get("swing_orders", [])),
            )

            return results

    def store_eod_prev_closes(self) -> None:
        """
        Store previous close prices for all tracked symbols at EOD.

        Should be called at end of RTH (12:55-1:00 PT).
        """
        if self.gap_manager is None:
            return

        for symbol, ticker in self.tickers.items():
            self.gap_manager.update_prev_close_from_ticker(symbol, ticker)

        self.gap_manager.save_all_prev_closes()
        self.logger.info("[ENGINE] Stored EOD prev_close for %d symbols.", len(self.tickers))

    # ---------- conflict resolution ---------- #

    def _resolve_conflicts_and_flatten(self, signal, is_day_signal: bool = False) -> tuple:
        """
        Check for position conflicts and flatten opposite-side positions if needed.

        Returns (allowed: bool, reentry_candidate_ids: List[str])
        - allowed: True if entry is allowed (after flattening if necessary)
        - reentry_candidate_ids: List of re-entry candidate IDs for flattened SWING
          positions (only populated when a DAY signal flattens a SWING position)
        """
        reentry_candidate_ids = []

        if self.conflict_resolver is None:
            # No conflict resolver configured - allow entry
            return True, reentry_candidate_ids

        decision = self.conflict_resolver.decide(signal)

        if not decision.allow_entry:
            self.logger.info(
                "[CONFLICT] Entry blocked for %s: %s",
                signal.symbol,
                decision.reason,
            )
            return False, reentry_candidate_ids

        if decision.requires_flatten:
            self.logger.info(
                "[CONFLICT] Flattening %d position(s) before %s entry: %s",
                len(decision.positions_to_flatten),
                signal.symbol,
                decision.reason,
            )
            for instr in decision.positions_to_flatten:
                # Store re-entry candidate if DAY signal is flattening a SWING position
                if is_day_signal and instr.kind == "SWING" and self.reentry_manager is not None:
                    # Get the original signal for this position
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
                            fill_price=0.0,  # Not needed for re-entry
                            fill_time=now_pt(),
                            parent_order_id=instr.parent_order_id,
                            stop_order_id=instr.stop_order_id,
                            timed_order_id=instr.timed_order_id,
                        )
                        try:
                            candidate_id = self.reentry_manager.store_candidate(
                                flattened_pos, original_signal
                            )
                            reentry_candidate_ids.append(candidate_id)
                            self.logger.info(
                                "[CONFLICT] Created re-entry candidate %s for flattened SWING %s %s",
                                candidate_id,
                                instr.symbol,
                                instr.strategy_id,
                            )
                        except Exception as exc:
                            self.logger.error(
                                "[CONFLICT] Failed to create re-entry candidate for %s %s: %s",
                                instr.symbol,
                                instr.strategy_id,
                                exc,
                            )

                flatten_success, cleanup_success = self.executor.flatten_position_with_retry(instr)
                if not flatten_success:
                    self.logger.error(
                        "[CONFLICT] Failed to flatten %s %s %s after all retries - blocking entry",
                        instr.symbol,
                        instr.kind,
                        instr.side,
                    )
                    # Clean up any candidates we created since flatten failed
                    for cid in reentry_candidate_ids:
                        try:
                            self.reentry_manager.drop_candidate(cid, "flatten_failed")
                        except Exception as exc:
                            self.logger.error(
                                "[CONFLICT] Failed to drop candidate %s: %s", cid, exc
                            )
                    return False, []

        return True, reentry_candidate_ids

    # ---------- tick handling ---------- #

    def on_ticker(self, symbol: str, ticker: Ticker) -> None:
        bid = _clean_price(ticker.bid)
        ask = _clean_price(ticker.ask)
        last = _clean_price(ticker.last)

        if bid is None and ask is None and last is None:
            return

        now = now_pt()
        today = now.date()

        # Update gap manager's prev_close tracking (during RTH)
        if is_rth(now) and self.gap_manager is not None and last is not None:
            self.gap_manager.update_prev_close_from_ticker(symbol, ticker)

        # Check for EOD prev_close save (12:55-13:00 PT)
        if is_rth(now) and self._eod_save_date != today:
            from market_calendar import get_market_close_time_pt
            from datetime import timedelta
            close_time = get_market_close_time_pt(today)
            # Save 5 minutes before close - use timedelta to handle hour rollover
            close_dt = datetime.combine(today, close_time)
            save_window_start = (close_dt - timedelta(minutes=5)).time()
            if now.time() >= save_window_start:
                self._trigger_eod_save(today)

        # Late-day re-check at 12:59 PT (1 minute before close)
        # This catches any Day trades that filled late and exited just before the save window
        if is_rth(now) and self._late_recheck_date != today:
            from market_calendar import get_market_close_time_pt
            from datetime import timedelta
            close_time = get_market_close_time_pt(today)
            # Trigger at 1 minute before close - use timedelta to handle hour rollover
            close_dt = datetime.combine(today, close_time)
            late_recheck_time = (close_dt - timedelta(minutes=1)).time()
            if now.time() >= late_recheck_time:
                self._trigger_late_recheck(today)

        if not is_rth(now):
            return

        for rt in self.day_by_symbol.get(symbol, []):
            # Auto-reset triggered flag when date changes
            if rt.triggered and rt.triggered_date != today:
                rt.triggered = False
                rt.triggered_date = None
            if not rt.triggered:
                self._evaluate_day(rt, bid, ask)

        for rt in self.swing_by_symbol.get(symbol, []):
            # Auto-reset triggered flag when date changes
            if rt.triggered and rt.triggered_date != today:
                rt.triggered = False
                rt.triggered_date = None
            if not rt.triggered:
                self._evaluate_swing(rt, bid, ask)

    def _trigger_eod_save(self, today: date) -> None:
        """Trigger EOD prev_close save and pending re-entry evaluations (once per day)."""
        if self._eod_save_date == today:
            return

        self._eod_save_date = today
        self._eod_save_done = True
        self.store_eod_prev_closes()
        self.logger.info("[ENGINE] Triggered EOD prev_close save for %s", today)

        # Evaluate any pending EOD re-entry candidates (from cancelled Day trades)
        if self.reentry_manager is not None:
            self.reentry_manager.evaluate_eod_candidates()
            self.logger.info("[ENGINE] Triggered EOD re-entry evaluation for %s", today)

    def _trigger_late_recheck(self, today: date) -> None:
        """
        Trigger late-day re-check at 12:59 PT (once per day).

        This final re-check catches any Day trades that may have filled late
        and exited just before the 12:55 save window, ensuring we don't miss
        any re-entry candidates for SWING positions.
        """
        if self._late_recheck_date == today:
            return

        self._late_recheck_date = today
        self._late_recheck_done = True

        if self.reentry_manager is not None:
            self.reentry_manager.evaluate_eod_candidates()
            self.logger.info("[ENGINE] Triggered late-day re-entry re-check for %s", today)

    # ---------- Day evaluation ---------- #

    def _evaluate_day(
        self,
        rt: DayRuntime,
        bid: Optional[float],
        ask: Optional[float],
    ) -> None:
        sig = rt.signal
        direction = (sig.direction or "").upper()
        lmt = float(sig.entry_price)

        if direction == "SHORT":
            if bid is None or bid < lmt:
                return

            today = now_pt().date()

            # Check if signal already filled (optimistic lock check)
            if self.state_mgr.signal_cache.is_signal_filled(
                today, sig.symbol, sig.strategy_id, "DAY"
            ):
                self.logger.info(
                    "[TRIG][DAY][SHORT][SKIP] %s strategy=%s - already filled today",
                    sig.symbol,
                    sig.strategy_id,
                )
                rt.triggered = True
                rt.triggered_date = today
                return

            # Check if symbol+strategy is blocked (bracket/gap failure)
            is_blocked, block_reason = self.state_mgr.blocked.is_blocked(
                sig.symbol, sig.strategy_id, today
            )
            if is_blocked:
                self.logger.info(
                    "[TRIG][DAY][SHORT][BLOCKED] %s strategy=%s bid=%.4f >= Lmt=%.4f but %s",
                    sig.symbol,
                    sig.strategy_id,
                    bid,
                    lmt,
                    block_reason,
                )
                rt.triggered = True
                rt.triggered_date = now_pt().date()
                return

            allowed, reason = self.cap_manager.can_open_day(sig)
            if not allowed:
                self.logger.info(
                    "[TRIG][DAY][SHORT][BLOCKED] %s strategy=%s bid=%.4f >= Lmt=%.4f but cap blocked: %s",
                    sig.symbol,
                    sig.strategy_id,
                    bid,
                    lmt,
                    reason,
                )
                rt.triggered = True
                rt.triggered_date = now_pt().date()
                return

            if not self.enable_day_trading:
                self.logger.info(
                    "[TRIG][DAY][SHORT][WOULD TRADE] %s strategy=%s bid=%.4f >= Lmt=%.4f (caps OK, trading disabled).",
                    sig.symbol,
                    sig.strategy_id,
                    bid,
                    lmt,
                )
                rt.triggered = True
                rt.triggered_date = now_pt().date()
                return

            # Check for position conflicts before entry
            allowed, reentry_candidate_ids = self._resolve_conflicts_and_flatten(sig, is_day_signal=True)
            if not allowed:
                self.logger.info(
                    "[TRIG][DAY][SHORT][CONFLICT_BLOCKED] %s strategy=%s - conflict resolution failed",
                    sig.symbol,
                    sig.strategy_id,
                )
                rt.triggered = True
                rt.triggered_date = now_pt().date()
                return

            self.logger.info(
                "[TRIG][DAY][SHORT][EXEC] %s strategy=%s bid=%.4f >= Lmt=%.4f (caps OK, sending bracket).",
                sig.symbol,
                sig.strategy_id,
                bid,
                lmt,
            )

            # Mark signal as filled BEFORE placing order (optimistic lock)
            self.state_mgr.signal_cache.mark_signal_filled(
                today, sig.symbol, sig.strategy_id, "DAY", filled=True
            )

            day_order_id = self.executor.place_day_bracket(sig)

            if day_order_id is not None and self.reentry_manager is not None:
                # Link re-entry candidates to the Day trade order ID
                for candidate_id in reentry_candidate_ids:
                    self.reentry_manager.link_day_trade(candidate_id, day_order_id)

                # Add this Day trade as a blocker for any OTHER pending re-entry candidates
                # on the same symbol (handles case where new Day trade opens while waiting)
                self.reentry_manager.add_blocker_for_symbol(
                    sig.symbol, day_order_id, sig.direction
                )
            elif day_order_id is None:
                # Day bracket placement failed - unmark signal
                self.state_mgr.signal_cache.mark_signal_filled(
                    today, sig.symbol, sig.strategy_id, "DAY", filled=False
                )

                # Drop orphaned re-entry candidates
                if reentry_candidate_ids and self.reentry_manager is not None:
                    for candidate_id in reentry_candidate_ids:
                        try:
                            self.reentry_manager.drop_candidate(candidate_id, "day_bracket_failed")
                        except Exception as exc:
                            self.logger.error(
                                "[TRIG][DAY][SHORT] Failed to drop candidate %s: %s",
                                candidate_id, exc
                            )

                # Block symbol+strategy for rest of week (swing re-entry) and day (day entry)
                self.state_mgr.blocked.block_for_week(sig.symbol, sig.strategy_id, today)
                self.state_mgr.blocked.block_for_day(sig.symbol, sig.strategy_id, today)
                self.logger.warning(
                    "[TRIG][DAY][SHORT][BLOCKED] %s %s - bracket failed, blocked for week and day",
                    sig.symbol,
                    sig.strategy_id,
                )

            rt.triggered = True
            rt.triggered_date = today

        else:
            if ask is None or ask > lmt:
                return

            today = now_pt().date()

            # Check if signal already filled (optimistic lock check)
            if self.state_mgr.signal_cache.is_signal_filled(
                today, sig.symbol, sig.strategy_id, "DAY"
            ):
                self.logger.info(
                    "[TRIG][DAY][LONG][SKIP] %s strategy=%s - already filled today",
                    sig.symbol,
                    sig.strategy_id,
                )
                rt.triggered = True
                rt.triggered_date = today
                return

            # Check if symbol+strategy is blocked (bracket/gap failure)
            is_blocked, block_reason = self.state_mgr.blocked.is_blocked(
                sig.symbol, sig.strategy_id, today
            )
            if is_blocked:
                self.logger.info(
                    "[TRIG][DAY][LONG][BLOCKED] %s strategy=%s ask=%.4f <= Lmt=%.4f but %s",
                    sig.symbol,
                    sig.strategy_id,
                    ask,
                    lmt,
                    block_reason,
                )
                rt.triggered = True
                rt.triggered_date = now_pt().date()
                return

            allowed, reason = self.cap_manager.can_open_day(sig)
            if not allowed:
                self.logger.info(
                    "[TRIG][DAY][LONG][BLOCKED] %s strategy=%s ask=%.4f <= Lmt=%.4f but cap blocked: %s",
                    sig.symbol,
                    sig.strategy_id,
                    ask,
                    lmt,
                    reason,
                )
                rt.triggered = True
                rt.triggered_date = now_pt().date()
                return

            if not self.enable_day_trading:
                self.logger.info(
                    "[TRIG][DAY][LONG][WOULD TRADE] %s strategy=%s ask=%.4f <= Lmt=%.4f (caps OK, trading disabled).",
                    sig.symbol,
                    sig.strategy_id,
                    ask,
                    lmt,
                )
                rt.triggered = True
                rt.triggered_date = now_pt().date()
                return

            # Check for position conflicts before entry
            allowed, reentry_candidate_ids = self._resolve_conflicts_and_flatten(sig, is_day_signal=True)
            if not allowed:
                self.logger.info(
                    "[TRIG][DAY][LONG][CONFLICT_BLOCKED] %s strategy=%s - conflict resolution failed",
                    sig.symbol,
                    sig.strategy_id,
                )
                rt.triggered = True
                rt.triggered_date = now_pt().date()
                return

            self.logger.info(
                "[TRIG][DAY][LONG][EXEC] %s strategy=%s ask=%.4f <= Lmt=%.4f (caps OK, sending bracket).",
                sig.symbol,
                sig.strategy_id,
                ask,
                lmt,
            )

            # Mark signal as filled BEFORE placing order (optimistic lock)
            self.state_mgr.signal_cache.mark_signal_filled(
                today, sig.symbol, sig.strategy_id, "DAY", filled=True
            )

            day_order_id = self.executor.place_day_bracket(sig)

            if day_order_id is not None and self.reentry_manager is not None:
                # Link re-entry candidates to the Day trade order ID
                for candidate_id in reentry_candidate_ids:
                    self.reentry_manager.link_day_trade(candidate_id, day_order_id)

                # Add this Day trade as a blocker for any OTHER pending re-entry candidates
                # on the same symbol (handles case where new Day trade opens while waiting)
                self.reentry_manager.add_blocker_for_symbol(
                    sig.symbol, day_order_id, sig.direction
                )
            elif day_order_id is None:
                # Day bracket placement failed - unmark signal
                self.state_mgr.signal_cache.mark_signal_filled(
                    today, sig.symbol, sig.strategy_id, "DAY", filled=False
                )

                # Drop orphaned re-entry candidates
                if reentry_candidate_ids and self.reentry_manager is not None:
                    for candidate_id in reentry_candidate_ids:
                        try:
                            self.reentry_manager.drop_candidate(candidate_id, "day_bracket_failed")
                        except Exception as exc:
                            self.logger.error(
                                "[TRIG][DAY][LONG] Failed to drop candidate %s: %s",
                                candidate_id, exc
                            )

                # Block symbol+strategy for rest of week (swing re-entry) and day (day entry)
                self.state_mgr.blocked.block_for_week(sig.symbol, sig.strategy_id, today)
                self.state_mgr.blocked.block_for_day(sig.symbol, sig.strategy_id, today)
                self.logger.warning(
                    "[TRIG][DAY][LONG][BLOCKED] %s %s - bracket failed, blocked for week and day",
                    sig.symbol,
                    sig.strategy_id,
                )

            rt.triggered = True
            rt.triggered_date = today

    # ---------- Swing evaluation ---------- #

    def _evaluate_swing(
        self,
        rt: SwingRuntime,
        bid: Optional[float],
        ask: Optional[float],
    ) -> None:
        sig = rt.signal
        entry = float(sig.entry_price)
        stype = (sig.strategy_id or "").lower()

        if ask is None:
            return

        if "momo" in stype or "breakout" in stype:
            cond = ask >= entry
            label = "MOMO"
            cmp_desc = ">="
            side_desc = "breakout"
        else:
            cond = ask <= entry
            label = "PULLBACK"
            cmp_desc = "<="
            side_desc = "pullback"

        if not cond:
            return

        # Ensure SwingSignal has a datetime.date trade_date
        trade_date = getattr(sig, "trade_date", None)
        if trade_date is None:
            trade_date = now_pt().date()
        elif isinstance(trade_date, str):
            try:
                trade_date = datetime.strptime(trade_date, "%Y-%m-%d").date()
            except ValueError as exc:
                self.logger.warning(
                    "[TRIG][SWING] Invalid trade_date format '%s' for %s, using today: %s",
                    trade_date, sig.symbol, exc,
                )
                trade_date = now_pt().date()
        setattr(sig, "trade_date", trade_date)

        # Check if signal already filled (optimistic lock check)
        if self.state_mgr.signal_cache.is_signal_filled(
            trade_date, sig.symbol, sig.strategy_id, "SWING"
        ):
            self.logger.info(
                "[TRIG][SWING][%s][SKIP] %s strategy=%s - already filled this week",
                label,
                sig.symbol,
                sig.strategy_id,
            )
            rt.triggered = True
            rt.triggered_date = trade_date
            return

        # Check if symbol+strategy is blocked for the week (bracket/gap failure)
        # Swing entries only check week-level blocks, not day-level
        is_blocked, block_reason = self.state_mgr.blocked.is_blocked(
            sig.symbol, sig.strategy_id, trade_date, check_week=True, check_day=False
        )
        if is_blocked:
            self.logger.info(
                "[TRIG][SWING][%s][BLOCKED] %s strategy=%s ask=%.4f %s Entry=%.4f but %s",
                label,
                sig.symbol,
                sig.strategy_id,
                ask,
                cmp_desc,
                entry,
                block_reason,
            )
            rt.triggered = True
            rt.triggered_date = now_pt().date()
            return

        allowed, reason = self.cap_manager.can_open_swing(sig)
        if not allowed:
            self.logger.info(
                "[TRIG][SWING][%s][BLOCKED] %s strategy=%s ask=%.4f %s Entry=%.4f but cap blocked: %s",
                label,
                sig.symbol,
                sig.strategy_id,
                ask,
                cmp_desc,
                entry,
                reason,
            )
            rt.triggered = True
            rt.triggered_date = now_pt().date()
            return

        if not self.enable_swing_trading:
            self.logger.info(
                "[TRIG][SWING][%s][WOULD TRADE] %s strategy=%s ask=%.4f %s Entry=%.4f (%s, caps OK, trading disabled).",
                label,
                sig.symbol,
                sig.strategy_id,
                ask,
                cmp_desc,
                entry,
                side_desc,
            )
            rt.triggered = True
            rt.triggered_date = now_pt().date()
            return

        # Check for position conflicts before entry (swing trades don't trigger re-entry)
        allowed, _ = self._resolve_conflicts_and_flatten(sig, is_day_signal=False)
        if not allowed:
            self.logger.info(
                "[TRIG][SWING][%s][CONFLICT_BLOCKED] %s strategy=%s - conflict resolution failed",
                label,
                sig.symbol,
                sig.strategy_id,
            )
            rt.triggered = True
            rt.triggered_date = now_pt().date()
            return

        self.logger.info(
            "[TRIG][SWING][%s][EXEC] %s strategy=%s ask=%.4f %s Entry=%.4f (%s, caps OK, sending bracket).",
            label,
            sig.symbol,
            sig.strategy_id,
            ask,
            cmp_desc,
            entry,
            side_desc,
        )

        # Mark signal as filled BEFORE placing order (optimistic lock)
        self.state_mgr.signal_cache.mark_signal_filled(
            trade_date, sig.symbol, sig.strategy_id, "SWING", filled=True
        )

        success = self.executor.place_swing_bracket(sig)
        if not success:
            # Swing bracket placement failed - unmark signal
            self.state_mgr.signal_cache.mark_signal_filled(
                trade_date, sig.symbol, sig.strategy_id, "SWING", filled=False
            )
            # Block for rest of week
            self.state_mgr.blocked.block_for_week(sig.symbol, sig.strategy_id, trade_date)
            self.logger.warning(
                "[TRIG][SWING][%s][FAILED] %s %s - bracket placement failed, blocked for week",
                label,
                sig.symbol,
                sig.strategy_id,
            )
        rt.triggered = True
        rt.triggered_date = trade_date
