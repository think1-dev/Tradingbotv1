"""
test_place_day_bracket.py

Phase 6: Minimal paper-trade test for a Day bracket.

- Loads today's Day CSV with CsvLoader.
- Takes the first DaySignal.
- Builds a Day bracket with build_day_bracket().
- Uses link_bracket(ib, ...) to assign orderIds/parentId/OCA.
- Qualifies the stock contract.
- Places the three orders in paper TWS.

This script WILL place a live paper bracket when run.
trading_bot.py still does NO trading.
"""

from __future__ import annotations

import logging
from pathlib import Path
from datetime import datetime, timedelta

from ib_insync import IB

from time_utils import now_pt, PT
from signals import CsvLoader
from orders import build_day_bracket, link_bracket


IB_HOST = "127.0.0.1"
IB_PORT = 7497
IB_CLIENT_ID = 1


def setup_logging() -> logging.Logger:
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    today_str = datetime.now().strftime("%Y%m%d")
    log_file = log_dir / f"test_bracket_{today_str}.log"

    logger = logging.getLogger("test_place_day_bracket")
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

    logger.info("[SYNC] Test logging initialized. Log file: %s", log_file)
    return logger


def choose_future_trade_date(logger: logging.Logger) -> datetime.date:
    """
    Pick a trade_date so that 12:55 PT and 12:58 PT are in the future
    relative to now. This avoids 'Order already expired' / 'Invalid
    effective time' during testing.

    - If it's before ~12:55 PT now -> use today.
    - Otherwise -> use tomorrow.
    """
    now = now_pt()
    today = now.date()

    stop_today = datetime(today.year, today.month, today.day, 12, 55, tzinfo=PT)

    if now < stop_today:
        logger.info(
            "[TEST] Using today's date %s for GTD/GAT (times still in the future).",
            today.isoformat(),
        )
        return today
    else:
        tomorrow = today + timedelta(days=1)
        logger.info(
            "[TEST] Using tomorrow's date %s for GTD/GAT (today's times already past).",
            tomorrow.isoformat(),
        )
        return tomorrow


def main() -> None:
    logger = setup_logging()

    # Load today's Day signals (file named by today's PT date)
    loader = CsvLoader(logger)
    today_pt = now_pt().date()
    day_signals = loader.load_day_signals(today_pt)

    if not day_signals:
        logger.info(
            "[TEST] No Day signals found for %s; aborting bracket placement.",
            today_pt.isoformat(),
        )
        return

    sample = day_signals[0]
    logger.info(
        "[TEST] Using first Day signal for bracket test: symbol=%s strategy=%s shares=%s entry=%s stop=%s",
        sample.symbol,
        sample.strategy_id,
        sample.shares,
        sample.entry_price,
        sample.stop_price,
    )

    # For testing, choose a trade_date with GTD/GAT in the future
    trade_date = choose_future_trade_date(logger)

    # Build bracket (no orderIds yet)
    bracket = build_day_bracket(sample, trade_date)

    ib = IB()
    logger.info(
        "[STATUS] Connecting to IBKR (paper) at %s:%s with clientId=%s ...",
        IB_HOST,
        IB_PORT,
        IB_CLIENT_ID,
    )
    ib.connect(IB_HOST, IB_PORT, clientId=IB_CLIENT_ID)

    if not ib.isConnected():
        logger.warning("[WARN] Could not connect to IBKR; aborting.")
        return

    logger.info("[SYNC] Connected to IBKR (test bracket).")

    try:
        # Qualify the contract (gets conId, exchange details, etc.)
        [qualified] = ib.qualifyContracts(bracket.contract)
        bracket.contract = qualified

        # Assign OCA group to exit legs
        oca_group = f"TEST_DAY_{sample.symbol}"
        bracket = link_bracket(bracket, oca_group=oca_group)

        # Assign order IDs and parent-child relationships
        parent_id = ib.client.getReqId()
        stop_id = ib.client.getReqId()
        timed_id = ib.client.getReqId()

        bracket.parent.orderId = parent_id
        bracket.stop.orderId = stop_id
        bracket.stop.parentId = parent_id
        bracket.timed.orderId = timed_id
        bracket.timed.parentId = parent_id

        # Set transmit flags (last child transmits the group)
        bracket.parent.transmit = False
        bracket.stop.transmit = False
        bracket.timed.transmit = True

        logger.info(
            "[TEST] Placing Day bracket for %s: parentId=%s, ocaGroup=%s",
            sample.symbol,
            bracket.parent.orderId,
            bracket.stop.ocaGroup,
        )

        # Place orders: parent, then children
        parent_trade = ib.placeOrder(bracket.contract, bracket.parent)
        stop_trade = ib.placeOrder(bracket.contract, bracket.stop)
        timed_trade = ib.placeOrder(bracket.contract, bracket.timed)

        logger.info("[TEST] Orders submitted. Waiting briefly for TWS to acknowledge...")
        ib.sleep(3.0)

        logger.info(
            "[TEST] Parent order status: %s",
            parent_trade.orderStatus.status if parent_trade.orderStatus else "UNKNOWN",
        )
        logger.info(
            "[TEST] Stop order status:   %s",
            stop_trade.orderStatus.status if stop_trade.orderStatus else "UNKNOWN",
        )
        logger.info(
            "[TEST] Timed order status:  %s",
            timed_trade.orderStatus.status if timed_trade.orderStatus else "UNKNOWN",
        )

        logger.info(
            "[TEST] Check TWS paper: you should see a parent + 2 child orders for %s in OCA group %s.",
            sample.symbol,
            bracket.stop.ocaGroup,
        )

    finally:
        if ib.isConnected():
            ib.disconnect()
            logger.info("[SYNC] Disconnected from IBKR (test bracket).")

    logger.info("[STATUS] Test bracket script complete.")


if __name__ == "__main__":
    main()
