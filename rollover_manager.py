"""
rollover_manager.py

Manages daily and weekly rollover tasks for 24/7 bot operation.

Daily at 6 AM PST:
- Clear FillTracker strategy_states (runtime memory)
- Clear day signal fill flags
- Reload day signals from CSV
- Update StrategyEngine with new signals
- Cleanup old state.json buckets (30d day, 4w swing)

Monday at 6 AM PST (in addition to daily):
- Clear swing signal fill flags
- Reload swing signals from CSV
- Update StrategyEngine with new swing signals
"""

from __future__ import annotations

import logging
import threading
from datetime import date, time, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Callable

if TYPE_CHECKING:
    from ib_insync import IB
    from state_manager import StateManager
    from fill_tracker import FillTracker
    from strategy_engine import StrategyEngine
    from signals import CsvLoader, DaySignal, SwingSignal


# Rollover time: 6:00 AM PST
ROLLOVER_HOUR = 6
ROLLOVER_MINUTE = 0

# Cleanup thresholds
DAY_BUCKET_KEEP_DAYS = 30
SWING_BUCKET_KEEP_WEEKS = 4


class RolloverManager:
    """
    Manages daily and weekly rollover for 24/7 operation.

    Schedules a check every minute during the 6 AM hour to detect
    when rollover should occur, then performs the rollover tasks.
    """

    def __init__(
        self,
        ib: "IB",
        logger: logging.Logger,
        state_mgr: "StateManager",
        fill_tracker: "FillTracker",
        loader: "CsvLoader",
        basket_dir: Path,
    ) -> None:
        self.ib = ib
        self.logger = logger
        self.state_mgr = state_mgr
        self.fill_tracker = fill_tracker
        self.loader = loader
        self.basket_dir = basket_dir

        # Track last rollover date to prevent duplicate rollovers
        self._last_rollover_date: Optional[date] = None
        self._lock = threading.Lock()

        # Callbacks to update StrategyEngine (set after engine is created)
        self._on_day_signals_updated: Optional[Callable[[List["DaySignal"]], None]] = None
        self._on_swing_signals_updated: Optional[Callable[[List["SwingSignal"]], None]] = None

        # Timer handle for cleanup
        self._timer: Optional[threading.Timer] = None
        self._running = False

    def set_day_signals_callback(
        self, callback: Callable[[List["DaySignal"]], None]
    ) -> None:
        """Set callback to notify when day signals are reloaded."""
        self._on_day_signals_updated = callback

    def set_swing_signals_callback(
        self, callback: Callable[[List["SwingSignal"]], None]
    ) -> None:
        """Set callback to notify when swing signals are reloaded."""
        self._on_swing_signals_updated = callback

    def start(self) -> None:
        """Start the rollover check scheduler."""
        self._running = True
        self._schedule_next_check()
        self.logger.info("[ROLLOVER] RolloverManager started. Checking daily at 6 AM PST.")

    def stop(self) -> None:
        """Stop the rollover check scheduler."""
        self._running = False
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None
        self.logger.info("[ROLLOVER] RolloverManager stopped.")

    def _schedule_next_check(self) -> None:
        """Schedule the next rollover check in 60 seconds."""
        if not self._running:
            return
        self._timer = threading.Timer(60.0, self._check_and_rollover)
        self._timer.daemon = True
        self._timer.start()

    def _check_and_rollover(self) -> None:
        """Check if rollover is needed and perform it."""
        try:
            from time_utils import now_pt

            now = now_pt()
            today = now.date()

            with self._lock:
                # Check if we're in the rollover window (6:00-6:10 AM PST)
                if now.hour == ROLLOVER_HOUR and now.minute < 10:
                    # Check if we already did rollover today
                    if self._last_rollover_date != today:
                        self._perform_rollover(today)
                        self._last_rollover_date = today
        except Exception as e:
            self.logger.exception("[ROLLOVER] Error during rollover check: %s", e)
        finally:
            # Schedule next check
            self._schedule_next_check()

    def _perform_rollover(self, today: date) -> None:
        """
        Perform the full rollover sequence.

        Daily tasks:
        - Clear FillTracker strategy_states
        - Clear day signal fill flags
        - Reload day signals
        - Cleanup old state buckets

        Monday tasks (additional):
        - Clear swing signal fill flags
        - Reload swing signals
        """
        from time_utils import week_monday_for
        from signals import restore_day_signal_from_cache, restore_swing_signal_from_cache

        is_monday = today.weekday() == 0  # Monday = 0

        self.logger.info(
            "[ROLLOVER] === Starting %s rollover for %s ===",
            "WEEKLY" if is_monday else "DAILY",
            today.isoformat(),
        )

        # === Step 1: Clear FillTracker strategy_states ===
        self._clear_strategy_states()

        # === Step 2: Clear signal fill flags ===
        self._clear_signal_fill_flags(today, is_monday)

        # === Step 3: Reload day signals ===
        day_signals = self._reload_day_signals(today)

        # === Step 4: Monday - reload swing signals ===
        swing_signals = None
        if is_monday:
            swing_signals = self._reload_swing_signals(today)

        # === Step 5: Cleanup old state buckets ===
        self._cleanup_old_buckets(today)

        # === Step 6: Notify StrategyEngine of new signals ===
        if day_signals is not None and self._on_day_signals_updated:
            self._on_day_signals_updated(day_signals)
            self.logger.info("[ROLLOVER] Notified StrategyEngine of %d new day signals.", len(day_signals))

        if swing_signals is not None and self._on_swing_signals_updated:
            self._on_swing_signals_updated(swing_signals)
            self.logger.info("[ROLLOVER] Notified StrategyEngine of %d new swing signals.", len(swing_signals))

        self.logger.info("[ROLLOVER] === Rollover complete for %s ===", today.isoformat())

    def _clear_strategy_states(self) -> None:
        """Clear FillTracker's in-memory strategy states."""
        count = len(self.fill_tracker.strategy_states)
        self.fill_tracker.strategy_states.clear()
        self.logger.info("[ROLLOVER] Cleared %d strategy_states entries from FillTracker.", count)

    def _clear_signal_fill_flags(self, today: date, is_monday: bool) -> None:
        """Clear signal fill flags for the new day/week."""
        # Clear day fills for today (new day = fresh slate)
        cache = self.state_mgr.signal_cache
        day_cleared = cache.clear_day_fills(today)
        self.logger.info("[ROLLOVER] Cleared %d day signal fill flags.", day_cleared)

        if is_monday:
            # Clear swing fills for this week (new week = fresh slate)
            swing_cleared = cache.clear_swing_fills(today)
            self.logger.info("[ROLLOVER] Cleared %d swing signal fill flags.", swing_cleared)

    def _reload_day_signals(self, today: date) -> Optional[List["DaySignal"]]:
        """Reload day signals from CSV."""
        from signals import restore_day_signal_from_cache

        day_csv_path = self.basket_dir / f"daytrades_{today.strftime('%Y%m%d')}.csv"

        # Try cache first
        cached = self.state_mgr.signal_cache.get_cached_day_signals(today, day_csv_path)
        if cached is not None:
            day_signals = [restore_day_signal_from_cache(s) for s in cached]
            self.logger.info("[ROLLOVER] Restored %d day signals from cache.", len(day_signals))
            return day_signals

        # Load fresh from CSV
        day_signals = self.loader.load_day_signals(today)
        if day_signals and day_csv_path.exists():
            self.state_mgr.signal_cache.cache_day_signals(today, day_csv_path, day_signals)

        self.logger.info("[ROLLOVER] Loaded %d day signals from CSV.", len(day_signals))
        return day_signals

    def _reload_swing_signals(self, today: date) -> Optional[List["SwingSignal"]]:
        """Reload swing signals from CSV."""
        from signals import restore_swing_signal_from_cache

        swing_csv_path = self.basket_dir / f"swingtrades_{today.strftime('%Y%m%d')}.csv"

        # Try cache first
        cached = self.state_mgr.signal_cache.get_cached_swing_signals(today, swing_csv_path)
        if cached is not None:
            swing_signals = [restore_swing_signal_from_cache(s) for s in cached]
            self.logger.info("[ROLLOVER] Restored %d swing signals from cache.", len(swing_signals))
            return swing_signals

        # Load fresh from CSV
        swing_signals = self.loader.load_swing_signals(today)
        if swing_signals and swing_csv_path.exists():
            self.state_mgr.signal_cache.cache_swing_signals(today, swing_csv_path, swing_signals)

        self.logger.info("[ROLLOVER] Loaded %d swing signals from CSV.", len(swing_signals))
        return swing_signals

    def _cleanup_old_buckets(self, today: date) -> None:
        """Remove old day and swing buckets from state.json."""
        from time_utils import week_monday_for

        # Cleanup day buckets older than 30 days
        day_cutoff = today - timedelta(days=DAY_BUCKET_KEEP_DAYS)
        day_removed = self._cleanup_day_buckets(day_cutoff)

        # Cleanup swing buckets older than 4 weeks
        swing_cutoff = week_monday_for(today) - timedelta(weeks=SWING_BUCKET_KEEP_WEEKS)
        swing_removed = self._cleanup_swing_buckets(swing_cutoff)

        if day_removed or swing_removed:
            self.logger.info(
                "[ROLLOVER] Cleaned up %d old day buckets, %d old swing buckets.",
                day_removed, swing_removed,
            )

    def _cleanup_day_buckets(self, cutoff: date) -> int:
        """Remove day buckets older than cutoff date."""
        day_state = self.state_mgr.state.get("day", {})
        keys_to_remove = []

        for key in day_state.keys():
            try:
                bucket_date = date.fromisoformat(key)
                if bucket_date < cutoff:
                    keys_to_remove.append(key)
            except ValueError:
                # Skip malformed keys
                continue

        for key in keys_to_remove:
            del day_state[key]

        if keys_to_remove:
            self.state_mgr._save()

        return len(keys_to_remove)

    def _cleanup_swing_buckets(self, cutoff: date) -> int:
        """Remove swing buckets older than cutoff date (by Monday key)."""
        swing_state = self.state_mgr.state.get("swing", {})
        keys_to_remove = []

        for key in swing_state.keys():
            try:
                bucket_monday = date.fromisoformat(key)
                if bucket_monday < cutoff:
                    keys_to_remove.append(key)
            except ValueError:
                # Skip malformed keys
                continue

        for key in keys_to_remove:
            del swing_state[key]

        if keys_to_remove:
            self.state_mgr._save()

        return len(keys_to_remove)

    def force_rollover(self) -> None:
        """
        Force an immediate rollover (for testing or manual trigger).

        Does NOT update _last_rollover_date, so the scheduled rollover
        will still run at 6 AM if not already done today.
        """
        from time_utils import now_pt
        today = now_pt().date()

        self.logger.info("[ROLLOVER] Forcing immediate rollover...")
        with self._lock:
            self._perform_rollover(today)
