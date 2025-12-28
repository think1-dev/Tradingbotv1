"""
test_bracket_anytime.py

Test bracket order placement at ANY time (ignores RTH).
Uses a hardcoded test signal - no CSV required.

WARNING: This WILL place real paper orders in TWS.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from pathlib import Path

from ib_insync import IB, Stock

from orders import build_day_bracket, link_bracket
from time_utils import now_pt, PT


IB_HOST = "127.0.0.1"
IB_PORT = 7497
IB_CLIENT_ID = 2  # Use different clientId to avoid conflicts


class TestSignal:
    """Minimal signal for testing."""
    def __init__(self, symbol: str, entry_price: float, stop_price: float, shares: int):
        self.symbol = symbol
        self.entry_price = entry_price
        self.stop_price = stop_price
        self.shares = shares
        self.strategy_id = "TEST_ANYTIME"
        self.direction = "LONG"


def setup_logging() -> logging.Logger:
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    today_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"test_anytime_{today_str}.log"

    logger = logging.getLogger("test_anytime")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger


def get_future_trade_date() -> datetime.date:
    """
    Pick a trade date so GTD/GAT times are in the future.
    Uses tomorrow if today's times have passed.
    """
    now = now_pt()
    today = now.date()

    # Check if 12:55 PT today is still in the future
    stop_time_today = datetime(today.year, today.month, today.day, 12, 55, tzinfo=PT)

    if now < stop_time_today and today.weekday() < 5:
        return today
    else:
        # Use next trading day (skip weekends)
        next_day = today + timedelta(days=1)
        while next_day.weekday() >= 5:  # Saturday=5, Sunday=6
            next_day += timedelta(days=1)
        return next_day


# Set to True to use GTC (Good Till Cancel) instead of GTD for testing
USE_GTC_FOR_TESTING = True


def main():
    logger = setup_logging()

    # === CONFIGURE YOUR TEST HERE ===
    # Use a liquid stock you don't mind having a paper order for
    TEST_SYMBOL = "AAPL"

    # Get current price to set realistic entry/stop
    # Entry slightly below market for a limit buy that won't fill immediately
    # This lets you see the bracket in TWS without it executing

    logger.info("=" * 60)
    logger.info("BRACKET TEST - ANYTIME (ignores RTH)")
    logger.info("=" * 60)

    ib = IB()
    logger.info(f"Connecting to IBKR at {IB_HOST}:{IB_PORT}...")

    try:
        ib.connect(IB_HOST, IB_PORT, clientId=IB_CLIENT_ID)

        if not ib.isConnected():
            logger.error("Failed to connect to IBKR")
            return

        logger.info("Connected to IBKR")

        # Get current market price for the test symbol
        contract = Stock(TEST_SYMBOL, "SMART", "USD")
        [qualified] = ib.qualifyContracts(contract)

        ticker = ib.reqMktData(qualified, "", False, False)
        ib.sleep(2)  # Wait for data

        # Use last price, or estimate if market closed
        current_price = ticker.last or ticker.close or 150.0
        logger.info(f"Current/Last price for {TEST_SYMBOL}: {current_price}")

        ib.cancelMktData(qualified)

        # Create test signal with entry BELOW current price (won't fill immediately)
        entry_price = round(current_price * 0.98, 2)  # 2% below market
        stop_price = round(entry_price * 0.95, 2)     # 5% below entry
        shares = 1  # Minimal size for testing

        test_signal = TestSignal(
            symbol=TEST_SYMBOL,
            entry_price=entry_price,
            stop_price=stop_price,
            shares=shares,
        )

        logger.info(f"Test Signal: {TEST_SYMBOL}")
        logger.info(f"  Entry (LMT): ${entry_price}")
        logger.info(f"  Stop (STP):  ${stop_price}")
        logger.info(f"  Shares:      {shares}")

        # Get future trade date for GTD/GAT
        trade_date = get_future_trade_date()
        logger.info(f"  Trade Date:  {trade_date} (for GTD/GAT times)")

        # Build bracket
        bracket = build_day_bracket(test_signal, trade_date)
        bracket.contract = qualified

        # For testing outside RTH: use GTC instead of GTD to avoid time conflicts
        if USE_GTC_FOR_TESTING:
            logger.info("Using GTC (Good Till Cancel) for stop order instead of GTD")
            bracket.stop.tif = "GTC"
            bracket.stop.goodTillDate = ""  # Clear GTD time

            # Remove GAT from timed exit - make it a simple GTC market order
            # that won't auto-trigger (you can manually test or cancel)
            bracket.timed.tif = "GTC"
            bracket.timed.goodAfterTime = ""
            bracket.timed.orderType = "LMT"  # Change to LMT so it doesn't execute immediately
            bracket.timed.lmtPrice = round(entry_price * 1.05, 2)  # 5% above entry (profit target)
            logger.info(f"Timed exit changed to LMT @ ${bracket.timed.lmtPrice} (profit target for testing)")

        # Link with OCA group
        oca_group = f"TEST_{TEST_SYMBOL}_{datetime.now().strftime('%H%M%S')}"
        bracket = link_bracket(bracket, oca_group=oca_group)

        # Assign order IDs
        parent_id = ib.client.getReqId()
        stop_id = ib.client.getReqId()
        timed_id = ib.client.getReqId()

        bracket.parent.orderId = parent_id
        bracket.stop.orderId = stop_id
        bracket.stop.parentId = parent_id
        bracket.timed.orderId = timed_id
        bracket.timed.parentId = parent_id

        # Transmit flags
        bracket.parent.transmit = False
        bracket.stop.transmit = False
        bracket.timed.transmit = True

        logger.info(f"Placing bracket with OCA group: {oca_group}")
        logger.info(f"  Parent ID: {parent_id}")
        logger.info(f"  Stop ID:   {stop_id}")
        logger.info(f"  Timed ID:  {timed_id}")

        # Place orders
        parent_trade = ib.placeOrder(qualified, bracket.parent)
        stop_trade = ib.placeOrder(qualified, bracket.stop)
        timed_trade = ib.placeOrder(qualified, bracket.timed)

        logger.info("Orders submitted! Waiting for acknowledgment...")
        ib.sleep(3)

        # Check status
        logger.info(f"Parent status: {parent_trade.orderStatus.status}")
        logger.info(f"Stop status:   {stop_trade.orderStatus.status}")
        logger.info(f"Timed status:  {timed_trade.orderStatus.status}")

        logger.info("=" * 60)
        logger.info("CHECK TWS: You should see a bracket order group")
        logger.info(f"  - Parent: BUY {shares} {TEST_SYMBOL} @ ${entry_price} LMT")
        logger.info(f"  - Stop:   SELL {shares} {TEST_SYMBOL} @ ${stop_price} STP")
        logger.info(f"  - Timed:  SELL {shares} {TEST_SYMBOL} MKT (GAT ~12:58 PT)")
        logger.info("=" * 60)

    except Exception as e:
        logger.exception(f"Error: {e}")
    finally:
        if ib.isConnected():
            ib.disconnect()
            logger.info("Disconnected from IBKR")


if __name__ == "__main__":
    main()
