"""
ib_env_check.py

Phase 0: Verify that Python + ib_insync can connect to IBKR paper
and fetch basic data (server time, account summary, positions, and
NBBO for one test symbol).
"""

from ib_insync import IB, Stock
from typing import Optional


# === CONFIG: adjust these if needed ===
IB_HOST = "127.0.0.1"        # TWS / Gateway host (usually localhost)
IB_PORT = 7497               # Change if your paper TWS/Gateway uses a different port
IB_CLIENT_ID = 1             # Fixed clientId for the bot
TEST_SYMBOL = "AAPL"         # Any liquid US stock is fine for testing


def connect_ib() -> Optional[IB]:
    """
    Connect to IBKR and return an IB instance, or None on failure.
    """
    ib = IB()
    try:
        print(f"Connecting to IB at {IB_HOST}:{IB_PORT} with clientId={IB_CLIENT_ID} ...")
        ib.connect(IB_HOST, IB_PORT, clientId=IB_CLIENT_ID)
        if not ib.isConnected():
            print("ERROR: ib.isConnected() returned False.")
            return None
        print("Connected to IBKR.")
        return ib
    except Exception as exc:
        print(f"ERROR: Could not connect to IBKR: {exc}")
        return None


def print_server_time(ib: IB) -> None:
    """
    Print IBKR server time.
    """
    try:
        server_time = ib.reqCurrentTime()
        print(f"Server time: {server_time}")
    except Exception as exc:
        print(f"ERROR requesting server time: {exc}")


def print_account_summary(ib: IB) -> None:
    """
    Print a simple account summary (first few rows).
    Note: ib.accountSummary() in this version does not accept 'tags=' kwarg.
    """
    try:
        print("\n=== Account Summary (first 10 rows) ===")
        rows = ib.accountSummary()
        for row in rows[:10]:
            # row has fields: account, tag, value, currency
            print(f"{row.tag} ({row.currency}): {row.value}")
    except Exception as exc:
        print(f"ERROR requesting account summary: {exc}")


def print_positions(ib: IB) -> None:
    """
    Print current positions (if any).
    """
    try:
        positions = ib.positions()
        print("\n=== Positions ===")
        if not positions:
            print("No open positions.")
        else:
            for pos in positions:
                contract = pos.contract
                print(f"{contract.symbol} {pos.position} @ {pos.avgCost}")
    except Exception as exc:
        print(f"ERROR requesting positions: {exc}")


def print_nbbo(ib: IB, symbol: str) -> None:
    """
    Request and print a single snapshot of top-of-book NBBO
    for the given symbol.
    """
    try:
        print(f"\nRequesting NBBO for {symbol} ...")
        contract = Stock(symbol, "SMART", "USD")
        # Qualify contract to ensure IB has full details
        [qualified_contract] = ib.qualifyContracts(contract)

        # Request market data snapshot (no deep book)
        ticker = ib.reqMktData(qualified_contract, "", False, False)

        # Wait briefly for data to arrive
        ib.sleep(2.0)

        bid = ticker.bid
        ask = ticker.ask
        last = ticker.last

        print(f"Symbol: {symbol}")
        print(f"  Bid:  {bid}")
        print(f"  Ask:  {ask}")
        print(f"  Last: {last}")

        # Cancel market data to be polite
        ib.cancelMktData(qualified_contract)

    except Exception as exc:
        print(f"ERROR requesting NBBO for {symbol}: {exc}")


def main() -> None:
    ib = connect_ib()
    if ib is None:
        return

    try:
        print_server_time(ib)
        print_account_summary(ib)
        print_positions(ib)
        print_nbbo(ib, TEST_SYMBOL)
    finally:
        print("\nDisconnecting from IBKR...")
        ib.disconnect()
        print("Disconnected. Phase 0 check complete.")


if __name__ == "__main__":
    main()
