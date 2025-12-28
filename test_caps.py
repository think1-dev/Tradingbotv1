from __future__ import annotations

"""
Phase 8.3 – CapManager cap logic test harness.

This script:
- Connects to IBKR paper (sanity check only; no orders placed).
- Uses CapManager + StateManager to simulate entries/exits.
- Writes to real logs/state.json.
- Tests:
  1) Day per-strategy cap (5 -> 6th blocked).
  2) Day global cap (20 -> 21st blocked).
  3) Swing per-strategy weekly cap (5 -> 6th blocked).
  4) Swing global cap (15 -> 16th blocked).
  5) Exits decrement open counts only and never go negative.

Run from project root:

    python test_caps.py

You can delete logs/state.json before running if you want a clean test.
"""

import logging
from pathlib import Path
from datetime import datetime, date, timedelta

from ib_insync import IB

from time_utils import now_pt
from state_manager import StateManager
from cap_manager import CapManager


# ---------- Simple dummy signal ---------- #


class DummySignal:
    """
    Minimal signal-like object for cap testing.

    CapManager only needs:
      - strategy_id
      - symbol
      - trade_date
    """

    def __init__(self, strategy_id: str, symbol: str, trade_date: date) -> None:
        self.strategy_id = strategy_id
        self.symbol = symbol
        self.trade_date = trade_date


# ---------- Logging setup ---------- #


def setup_logger() -> logging.Logger:
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    today_str = datetime.now().strftime("%Y%m%d")
    log_file = log_dir / f"test_caps_{today_str}.log"

    logger = logging.getLogger("test_caps")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter(
        "%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    ch.setLevel(logging.INFO)
    logger.addHandler(ch)

    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(fmt)
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)

    logger.info("[TEST] Logging initialized. File: %s", log_file)
    return logger


# ---------- Utility: state inspection ---------- #


def _get_day_bucket(state_mgr: StateManager, d: date) -> dict:
    """Return raw day bucket for date d from state."""
    state = getattr(state_mgr, "state", {})
    day_root = state.get("day", {})
    return day_root.get(d.isoformat(), {}) or {}


def _get_swing_bucket(state_mgr: StateManager, week_monday: date) -> dict:
    """Return raw swing bucket for the Monday-week containing week_monday."""
    state = getattr(state_mgr, "state", {})
    swing_root = state.get("swing", {})
    return swing_root.get(week_monday.isoformat(), {}) or {}


# ---------- Scenarios ---------- #


def scenario_day_strategy_cap(
    cap_mgr: CapManager,
    state_mgr: StateManager,
    test_date: date,
    logger: logging.Logger,
) -> None:
    logger.info("========== Scenario 1: Day per-strategy cap (5 -> 6th blocked) ==========")

    strat_id = "TEST_DAY_STRAT_1"
    symbol = "DAYCAP1"

    for i in range(1, 7):
        sig = DummySignal(strat_id, symbol, test_date)
        allowed, reason = cap_mgr.can_open_day(sig)
        day_bucket = _get_day_bucket(state_mgr, test_date)
        fills = day_bucket.get("strategy_fills", {}).get(strat_id, 0)
        day_open = day_bucket.get("DayOpen", 0)

        logger.info(
            "  Attempt %d: can_open_day -> allowed=%s, reason=%s (before entry: fills=%d, DayOpen=%d)",
            i,
            allowed,
            reason,
            fills,
            day_open,
        )

        if allowed:
            cap_mgr.register_day_entry(sig)
            day_bucket = _get_day_bucket(state_mgr, test_date)
            fills = day_bucket.get("strategy_fills", {}).get(strat_id, 0)
            day_open = day_bucket.get("DayOpen", 0)
            logger.info(
                "    Registered Day entry %d for %s. Now: fills=%d, DayOpen=%d",
                i,
                strat_id,
                fills,
                day_open,
            )
        else:
            logger.info(
                "    Entry %d BLOCKED for %s as expected (caps).", i, strat_id
            )

    # Final check
    day_bucket = _get_day_bucket(state_mgr, test_date)
    final_fills = day_bucket.get("strategy_fills", {}).get(strat_id, 0)
    final_open = day_bucket.get("DayOpen", 0)
    logger.info(
        "[SUMMARY] Scenario 1: strategy=%s -> final fills=%d (expected 5), DayOpen=%d (expected 5).",
        strat_id,
        final_fills,
        final_open,
    )


def scenario_day_global_cap(
    cap_mgr: CapManager,
    state_mgr: StateManager,
    test_date: date,
    logger: logging.Logger,
) -> None:
    logger.info("========== Scenario 2: Day global cap (20 -> 21st blocked) ==========")

    # We will open additional entries across multiple strategies until DayOpen hits 20.
    strategies = ["TEST_DAY_S2", "TEST_DAY_S3", "TEST_DAY_S4", "TEST_DAY_S5"]
    symbol_template = "DAYCAP_GL_%s"

    # Determine starting DayOpen (in case Scenario 1 already added some).
    day_bucket = _get_day_bucket(state_mgr, test_date)
    starting_open = day_bucket.get("DayOpen", 0)
    logger.info(
        "  Starting DayOpen for %s: %d", test_date.isoformat(), starting_open
    )

    max_attempts = 40
    attempt = 0

    # Open until DayOpen >= 20 or attempts exhausted
    while attempt < max_attempts:
        day_bucket = _get_day_bucket(state_mgr, test_date)
        day_open = day_bucket.get("DayOpen", 0)
        if day_open >= 20:
            break

        strat_id = strategies[attempt % len(strategies)]
        symbol = symbol_template % strat_id[-1]  # simple variation
        sig = DummySignal(strat_id, symbol, test_date)

        allowed, reason = cap_mgr.can_open_day(sig)
        logger.info(
            "  Global attempt %d for %s: allowed=%s, reason=%s (DayOpen=%d)",
            attempt + 1,
            strat_id,
            allowed,
            reason,
            day_open,
        )

        if not allowed:
            logger.info(
                "    Unexpected BLOCK while DayOpen < 20 (strategy cap?). Breaking."
            )
            break

        cap_mgr.register_day_entry(sig)
        attempt += 1

    # After loop, DayOpen should be >= 20 (ideally == 20).
    day_bucket = _get_day_bucket(state_mgr, test_date)
    day_open = day_bucket.get("DayOpen", 0)
    logger.info(
        "  After opening attempts, DayOpen for %s is %d (expected at least 20).",
        test_date.isoformat(),
        day_open,
    )

    # Now attempt one more entry with a fresh strategy to trigger global cap.
    extra_strat = "TEST_DAY_GLOBAL_EXTRA"
    extra_sig = DummySignal(extra_strat, "DAYCAP_EXTRA", test_date)
    allowed, reason = cap_mgr.can_open_day(extra_sig)
    logger.info(
        "  Extra (global) attempt with %s: allowed=%s, reason=%s (DayOpen=%d)",
        extra_strat,
        allowed,
        reason,
        day_open,
    )

    if allowed:
        logger.warning(
            "[SUMMARY] Scenario 2: extra attempt was ALLOWED; expected global cap to block."
        )
    else:
        logger.info(
            "[SUMMARY] Scenario 2: extra attempt correctly BLOCKED with reason='%s'. DayOpen=%d (expected 20).",
            reason,
            day_open,
        )


def scenario_swing_strategy_cap(
    cap_mgr: CapManager,
    state_mgr: StateManager,
    week_monday: date,
    logger: logging.Logger,
) -> None:
    logger.info(
        "========== Scenario 3: Swing per-strategy weekly cap (5 -> 6th blocked) =========="
    )

    strat_id = "TEST_SWING_STRAT_1"
    symbol = "SWINGCAP1"

    for i in range(1, 7):
        sig = DummySignal(strat_id, symbol, week_monday)
        allowed, reason = cap_mgr.can_open_swing(sig)

        swing_bucket = _get_swing_bucket(state_mgr, week_monday)
        fills = swing_bucket.get("strategy_fills", {}).get(strat_id, 0)
        swing_open = swing_bucket.get("SwingOpen", 0)

        logger.info(
            "  Attempt %d: can_open_swing -> allowed=%s, reason=%s (before entry: fills=%d, SwingOpen=%d)",
            i,
            allowed,
            reason,
            fills,
            swing_open,
        )

        if allowed:
            cap_mgr.register_swing_entry(sig)
            swing_bucket = _get_swing_bucket(state_mgr, week_monday)
            fills = swing_bucket.get("strategy_fills", {}).get(strat_id, 0)
            swing_open = swing_bucket.get("SwingOpen", 0)
            logger.info(
                "    Registered Swing entry %d for %s. Now: fills=%d, SwingOpen=%d",
                i,
                strat_id,
                fills,
                swing_open,
            )
        else:
            logger.info(
                "    Entry %d BLOCKED for %s as expected (caps).", i, strat_id
            )

    # Final check
    swing_bucket = _get_swing_bucket(state_mgr, week_monday)
    final_fills = swing_bucket.get("strategy_fills", {}).get(strat_id, 0)
    final_open = swing_bucket.get("SwingOpen", 0)
    logger.info(
        "[SUMMARY] Scenario 3: strategy=%s -> final fills=%d (expected 5), SwingOpen=%d (expected 5).",
        strat_id,
        final_fills,
        final_open,
    )


def scenario_swing_global_cap(
    cap_mgr: CapManager,
    state_mgr: StateManager,
    week_monday: date,
    logger: logging.Logger,
) -> None:
    logger.info("========== Scenario 4: Swing global cap (15 -> 16th blocked) ==========")

    strategies = ["TEST_SWING_S2", "TEST_SWING_S3", "TEST_SWING_S4", "TEST_SWING_S5"]
    symbol_template = "SWINGCAP_GL_%s"

    swing_bucket = _get_swing_bucket(state_mgr, week_monday)
    starting_open = swing_bucket.get("SwingOpen", 0)
    logger.info(
        "  Starting SwingOpen for week %s: %d", week_monday.isoformat(), starting_open
    )

    max_attempts = 40
    attempt = 0

    while attempt < max_attempts:
        swing_bucket = _get_swing_bucket(state_mgr, week_monday)
        swing_open = swing_bucket.get("SwingOpen", 0)
        if swing_open >= 15:
            break

        strat_id = strategies[attempt % len(strategies)]
        symbol = symbol_template % strat_id[-1]
        sig = DummySignal(strat_id, symbol, week_monday)

        allowed, reason = cap_mgr.can_open_swing(sig)
        logger.info(
            "  Global swing attempt %d for %s: allowed=%s, reason=%s (SwingOpen=%d)",
            attempt + 1,
            strat_id,
            allowed,
            reason,
            swing_open,
        )

        if not allowed:
            logger.info(
                "    Unexpected BLOCK while SwingOpen < 15 (strategy cap?). Breaking."
            )
            break

        cap_mgr.register_swing_entry(sig)
        attempt += 1

    swing_bucket = _get_swing_bucket(state_mgr, week_monday)
    swing_open = swing_bucket.get("SwingOpen", 0)
    logger.info(
        "  After opening attempts, SwingOpen for week %s is %d (expected at least 15).",
        week_monday.isoformat(),
        swing_open,
    )

    extra_strat = "TEST_SWING_GLOBAL_EXTRA"
    extra_sig = DummySignal(extra_strat, "SWINGCAP_EXTRA", week_monday)
    allowed, reason = cap_mgr.can_open_swing(extra_sig)
    logger.info(
        "  Extra (global) swing attempt with %s: allowed=%s, reason=%s (SwingOpen=%d)",
        extra_strat,
        allowed,
        reason,
        swing_open,
    )

    if allowed:
        logger.warning(
            "[SUMMARY] Scenario 4: extra attempt was ALLOWED; expected global cap to block."
        )
    else:
        logger.info(
            "[SUMMARY] Scenario 4: extra attempt correctly BLOCKED with reason='%s'. SwingOpen=%d (expected 15).",
            reason,
            swing_open,
        )


def scenario_exits(
    cap_mgr: CapManager,
    state_mgr: StateManager,
    test_date: date,
    week_monday: date,
    logger: logging.Logger,
) -> None:
    logger.info(
        "========== Scenario 5: Exit behavior (open counts decrement, no negatives) =========="
    )

    # --- Day exits ---
    day_bucket = _get_day_bucket(state_mgr, test_date)
    day_open = day_bucket.get("DayOpen", 0)
    logger.info(
        "  DayOpen BEFORE exits for %s: %d", test_date.isoformat(), day_open
    )

    dummy_day_sig = DummySignal("EXIT_DAY", "EXITDAY", test_date)
    for i in range(day_open + 2):  # a couple extra to test floor at 0
        cap_mgr.register_day_exit(dummy_day_sig)
        day_bucket = _get_day_bucket(state_mgr, test_date)
        current_open = day_bucket.get("DayOpen", 0)
        logger.info(
            "    Day exit call %d -> DayOpen now %d", i + 1, current_open
        )

    # --- Swing exits ---
    swing_bucket = _get_swing_bucket(state_mgr, week_monday)
    swing_open = swing_bucket.get("SwingOpen", 0)
    logger.info(
        "  SwingOpen BEFORE exits for week %s: %d",
        week_monday.isoformat(),
        swing_open,
    )

    dummy_swing_sig = DummySignal("EXIT_SWING", "EXITSWING", week_monday)
    for i in range(swing_open + 2):
        cap_mgr.register_swing_exit(dummy_swing_sig)
        swing_bucket = _get_swing_bucket(state_mgr, week_monday)
        current_open = swing_bucket.get("SwingOpen", 0)
        logger.info(
            "    Swing exit call %d -> SwingOpen now %d", i + 1, current_open
        )

    # Final check
    day_bucket = _get_day_bucket(state_mgr, test_date)
    final_day = day_bucket.get("DayOpen", 0)

    swing_bucket = _get_swing_bucket(state_mgr, week_monday)
    final_swing = swing_bucket.get("SwingOpen", 0)

    logger.info(
        "[SUMMARY] Scenario 5: final DayOpen=%d (expected >=0, typically 0), final SwingOpen=%d (expected >=0, typically 0).",
        final_day,
        final_swing,
    )


# ---------- Main ---------- #


def main() -> None:
    logger = setup_logger()

    # Connect to IBKR paper just to ensure environment is sane (no orders used here).
    ib = IB()
    try:
        logger.info("[TEST] Connecting to IBKR paper (127.0.0.1:7497, clientId=999)...")
        ib.connect("127.0.0.1", 7497, clientId=999)
        if ib.isConnected():
            logger.info("[TEST] Connected to IBKR paper. (No orders will be sent.)")
        else:
            logger.warning(
                "[TEST] ib.isConnected() is False; cap tests will still run offline."
            )
    except Exception as exc:
        logger.warning(
            "[TEST] Could not connect to IBKR paper: %s. Proceeding with offline cap tests.",
            exc,
        )

    # Initialize StateManager + CapManager
    state_mgr = StateManager(logger)
    cap_mgr = CapManager(logger, state_mgr)

    today_pt = now_pt().date()
    weekday = today_pt.weekday()  # Monday=0
    week_monday = today_pt - timedelta(days=weekday)

    logger.info(
        "[TEST] Using PT date %s and week Monday %s for cap tests.",
        today_pt.isoformat(),
        week_monday.isoformat(),
    )

    # Run scenarios
    scenario_day_strategy_cap(cap_mgr, state_mgr, today_pt, logger)
    scenario_day_global_cap(cap_mgr, state_mgr, today_pt, logger)
    scenario_swing_strategy_cap(cap_mgr, state_mgr, week_monday, logger)
    scenario_swing_global_cap(cap_mgr, state_mgr, week_monday, logger)
    scenario_exits(cap_mgr, state_mgr, today_pt, week_monday, logger)

    # Done
    logger.info("[TEST] Phase 8.3 cap tests complete. Inspect logs/state.json for details.")

    try:
        if ib.isConnected():
            ib.disconnect()
            logger.info("[TEST] Disconnected from IBKR.")
    except Exception:
        pass


if __name__ == "__main__":
    main()
