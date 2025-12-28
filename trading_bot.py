from __future__ import annotations

"""
Phase 1–6:

- Phase 1: Logging + IBKR connection skeleton.
- Phase 2: Time utilities sanity logs (PT).
- Phase 3: CSV loader integration (Day/Swing signals).
- Phase 4: StateManager initialization (caps + JSON state).
- Phase 5: Bracket order factory dry-run (no placement).
- Phase 6: StrategyEngine + OrderExecutor:
    * Subscribes to NBBO for list symbols.
    * Evaluates triggers and caps.
    * When enabled, places real brackets via OrderExecutor.
    * Keeps fills/exits in sync with StateManager.

By default, ENABLE_DAY_TRADING and ENABLE_SWING_TRADING are False,
so the bot runs in log-only mode (no orders are sent).

Phase 8.1:
- Introduce CapManager to centralize all cap logic.

Phase 8.2:
- Introduce FillTracker for fill-based cap management.
- Tracks pending orders by strategy.
- When an order fills, increments fill counter.
- When strategy cap (5) is reached, cancels all remaining unfilled orders.
"""

import asyncio
import sys
import logging
from pathlib import Path
from datetime import datetime

# --- Python 3.11+ event loop fix for ib_insync / eventkit ---
# On newer Python versions there is no default event loop until we create one.
# ib_insync expects an event loop to exist when it is imported.
if sys.version_info >= (3, 11):
    try:
        asyncio.get_event_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())

from ib_insync import IB

from time_utils import (
    now_pt,
    today_pt_date_str,
    is_rth,
    get_day_stop_time_pt,
    get_day_exit_time_pt,
)
from signals import (
    CsvLoader,
    BASKET_DIR,
    restore_day_signal_from_cache,
    restore_swing_signal_from_cache,
)
from state_manager import StateManager
from orders import build_day_bracket, build_swing_bracket
from strategy_engine import StrategyEngine
from execution import OrderExecutor
from cap_manager import CapManager
from fill_tracker import FillTracker
from conflict_resolver import ConflictResolver
from reentry_manager import ReentryManager
from gap_manager import GapManager


# === IB CONNECTION CONFIG ===
IB_HOST = "127.0.0.1"
IB_PORT = 7497
IB_CLIENT_ID = 1

# === TRADING ENABLE FLAGS ===
# Flip these to True *only* when you are ready to let the bot place
# real (paper) orders.
ENABLE_DAY_TRADING = True
ENABLE_SWING_TRADING = True


def setup_logging() -> logging.Logger:
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    # Use PT timezone for consistent log file naming
    today_str = now_pt().strftime("%Y%m%d")
    log_file = log_dir / f"ibkr_{today_str}.log"

    logger = logging.getLogger("trading_bot")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    logger.info("[SYNC] Logging initialized. Log file: %s", log_file)
    return logger


def main() -> None:
    logger = setup_logging()

    # === Phase 2: time-utils sanity logs ===
    pt_now = now_pt()
    logger.info("[STATUS] now_pt() = %s", pt_now)
    logger.info("[STATUS] today_pt_date_str() = %s", today_pt_date_str())
    logger.info("[STATUS] is_rth(now_pt()) = %s", is_rth(pt_now))
    logger.info("[STATUS] Day stop time (PT) today = %s", get_day_stop_time_pt())
    logger.info("[STATUS] Day exit time (PT) today = %s", get_day_exit_time_pt())

    # === Phase 4: StateManager init ===
    state_mgr = StateManager(logger)
    logger.info("[STATUS] StateManager initialized with state file at logs/state.json.")

    # === Phase 8.1: CapManager init ===
    cap_manager = CapManager(logger, state_mgr)
    logger.info("[STATUS] CapManager initialized and wired to StateManager.")

    # === Phase 3: CSV loader with caching ===
    loader = CsvLoader(logger)
    today_pt = now_pt().date()

    # Try to restore from cache first (preserves pre-calculated stop_distance)
    day_csv_path = BASKET_DIR / f"daytrades_{today_pt.strftime('%Y%m%d')}.csv"
    swing_csv_path = BASKET_DIR / f"swingtrades_{today_pt.strftime('%Y%m%d')}.csv"

    # Check for cached day signals
    cached_day = state_mgr.signal_cache.get_cached_day_signals(today_pt, day_csv_path)
    if cached_day is not None:
        day_signals = [restore_day_signal_from_cache(s) for s in cached_day]
        logger.info("[CACHE] Restored %d day signals from cache.", len(day_signals))
    else:
        day_signals = loader.load_day_signals(today_pt)
        if day_signals and day_csv_path.exists():
            state_mgr.signal_cache.cache_day_signals(today_pt, day_csv_path, day_signals)

    # Check for cached swing signals
    cached_swing = state_mgr.signal_cache.get_cached_swing_signals(today_pt, swing_csv_path)
    if cached_swing is not None:
        swing_signals = [restore_swing_signal_from_cache(s) for s in cached_swing]
        logger.info("[CACHE] Restored %d swing signals from cache.", len(swing_signals))
    else:
        swing_signals = loader.load_swing_signals(today_pt)
        if swing_signals and swing_csv_path.exists():
            state_mgr.signal_cache.cache_swing_signals(today_pt, swing_csv_path, swing_signals)

    # Clean up old cache entries (older than 7 days)
    state_mgr.signal_cache.cleanup_old_cache(today_pt)

    logger.info(
        "[CSV] Phase 3 summary: %d Day signals, %d Swing signals loaded for %s.",
        len(day_signals),
        len(swing_signals),
        today_pt.isoformat(),
    )

    # === Phase 5: Order factory dry-run (no placement) ===
    if day_signals:
        sample_day = day_signals[0]
        bracket = build_day_bracket(sample_day, sample_day.trade_date)
        logger.info("[TEST] Day bracket sample for %s:", sample_day.symbol)
        logger.info(
            "[TEST]   Parent: action=%s type=%s qty=%s lmtPrice=%s tif=%s",
            bracket.parent.action,
            bracket.parent.orderType,
            bracket.parent.totalQuantity,
            getattr(bracket.parent, "lmtPrice", None),
            bracket.parent.tif,
        )
        logger.info(
            "[TEST]   Stop:   action=%s type=%s qty=%s stop=%s tif=%s GTD=%s",
            bracket.stop.action,
            bracket.stop.orderType,
            bracket.stop.totalQuantity,
            bracket.stop.auxPrice,
            bracket.stop.tif,
            getattr(bracket.stop, "goodTillDate", None),
        )
        logger.info(
            "[TEST]   Timed:  action=%s type=%s qty=%s tif=%s GAT=%s",
            bracket.timed.action,
            bracket.timed.orderType,
            bracket.timed.totalQuantity,
            bracket.timed.tif,
            getattr(bracket.timed, "goodAfterTime", None),
        )
    else:
        logger.info("[TEST] No Day signals available for bracket sample.")

    if swing_signals:
        sample_swing = swing_signals[0]
        bracket = build_swing_bracket(sample_swing, today_pt)
        logger.info("[TEST] Swing bracket sample for %s:", sample_swing.symbol)
        logger.info(
            "[TEST]   Parent: action=%s type=%s qty=%s lmtPrice=%s tif=%s",
            bracket.parent.action,
            bracket.parent.orderType,
            bracket.parent.totalQuantity,
            getattr(bracket.parent, "lmtPrice", None),
            bracket.parent.tif,
        )
        logger.info(
            "[TEST]   Stop:   action=%s type=%s qty=%s stop=%s tif=%s GTD=%s",
            bracket.stop.action,
            bracket.stop.orderType,
            bracket.stop.totalQuantity,
            bracket.stop.auxPrice,
            bracket.stop.tif,
            getattr(bracket.stop, "goodTillDate", None),
        )
        logger.info(
            "[TEST]   Timed:  action=%s type=%s qty=%s tif=%s GAT=%s",
            bracket.timed.action,
            bracket.timed.orderType,
            bracket.timed.totalQuantity,
            bracket.timed.tif,
            getattr(bracket.timed, "goodAfterTime", None),
        )
    else:
        logger.info("[TEST] No Swing signals available for bracket sample.")

    # === Phase 1: IB connection skeleton ===
    ib = IB()

    def on_disconnected():
        logger.warning("[WARN] Disconnected from IBKR.")

    ib.disconnectedEvent += on_disconnected

    logger.info(
        "[STATUS] Connecting to IBKR at %s:%s with clientId=%s ...",
        IB_HOST,
        IB_PORT,
        IB_CLIENT_ID,
    )

    try:
        ib.connect(IB_HOST, IB_PORT, clientId=IB_CLIENT_ID)

        if not ib.isConnected():
            logger.warning("[WARN] ib.isConnected() returned False; exiting.")
            return

        logger.info(
            "[SYNC] Connected to IBKR (host=%s, port=%s, clientId=%s).",
            IB_HOST,
            IB_PORT,
            IB_CLIENT_ID,
        )

        # === Phase 8.2: FillTracker for fill-based cap management ===
        fill_tracker = FillTracker(ib, logger, cap_manager, state_mgr)
        logger.info("[STATUS] FillTracker initialized for fill-based cap enforcement.")

        # === Phase 8.3: ConflictResolver for position conflict management ===
        conflict_resolver = ConflictResolver(logger, fill_tracker)
        logger.info("[STATUS] ConflictResolver initialized for position conflict management.")

        # === Phase 6 + 8.1: OrderExecutor + StrategyEngine with CapManager ===
        executor = OrderExecutor(ib, logger, state_mgr, cap_manager, fill_tracker)

        # === Phase 8.4: ReentryManager for Weekly re-entry after Day conflict ===
        reentry_manager = ReentryManager(ib, logger, fill_tracker, executor, state_mgr)

        # Register callbacks so ReentryManager gets notified when Day trades exit
        fill_tracker.set_day_exit_callback(reentry_manager.on_day_trade_exit)
        fill_tracker.set_day_cancel_callback(reentry_manager.on_day_trade_cancelled)

        # Register callback for timed exit cancellation (Day/Swing MKT exit failed)
        fill_tracker.set_timed_exit_cancel_callback(executor.handle_timed_exit_cancel)
        logger.info("[STATUS] ReentryManager initialized for Weekly re-entry management.")

        # === Phase 8.5: GapManager for gap-at-open trades ===
        # Build swing signals lookup for re-entry candidate creation
        swing_signals_by_key = {}
        for sig in swing_signals:
            key = f"{sig.symbol}_{sig.strategy_id}"
            swing_signals_by_key[key] = sig

        gap_manager = GapManager(
            ib=ib,
            logger=logger,
            fill_tracker=fill_tracker,
            executor=executor,
            state_mgr=state_mgr,
            cap_manager=cap_manager,
            conflict_resolver=conflict_resolver,
            reentry_manager=reentry_manager,
            swing_signals_by_key=swing_signals_by_key,
        )
        logger.info("[STATUS] GapManager initialized for gap-at-open trade detection.")

        # Check for stale prev_close data and fetch historical if needed
        all_symbols = list(set(
            [s.symbol for s in day_signals] + [s.symbol for s in swing_signals]
        ))
        gap_manager.check_and_fetch_stale_prev_closes(all_symbols)

        # Check for pending re-entry candidates at startup (market open re-check)
        if is_rth(now_pt()):
            # Process any pending flattens from previous day(s)
            executor.process_pending_flattens()
            # Clean up expired blocked entries
            state_mgr.blocked.cleanup_expired(now_pt().date())
            # Check pending re-entry candidates
            reentry_manager.on_market_open()

        engine = StrategyEngine(
            ib=ib,
            logger=logger,
            state_mgr=state_mgr,
            cap_manager=cap_manager,
            executor=executor,
            day_signals=day_signals,
            swing_signals=swing_signals,
            enable_day_trading=ENABLE_DAY_TRADING,
            enable_swing_trading=ENABLE_SWING_TRADING,
            conflict_resolver=conflict_resolver,
            reentry_manager=reentry_manager,
            gap_manager=gap_manager,
        )
        engine.start()

        # Run market open gap check if starting during RTH
        if is_rth(now_pt()):
            engine.run_market_open_gap_check()

        mode = "LIVE" if (ENABLE_DAY_TRADING or ENABLE_SWING_TRADING) else "log-only"
        logger.info(
            "[STATUS] Starting ib.run() event loop (mode=%s). Press Ctrl+C to stop the bot.",
            mode,
        )

        ib.run()

    except KeyboardInterrupt:
        logger.info("[STATUS] KeyboardInterrupt received; shutting down.")
    except Exception as exc:
        logger.exception("[WARN] Unhandled exception in main(): %s", exc)
    finally:
        if ib.isConnected():
            ib.disconnect()
            logger.info("[SYNC] Disconnected from IBKR.")
        logger.info("[STATUS] Phase 1–6 skeleton terminated.")


if __name__ == "__main__":
    main()
