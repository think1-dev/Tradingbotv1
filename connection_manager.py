"""
connection_manager.py

Manages IBKR connection with automatic reconnection for 24/7 operation.

Reconnection behavior:
- On disconnect, attempt to reconnect with exponential backoff
- Backoff: 5s, 10s, 20s, 40s, 80s, 120s (max 2 minutes)
- Keep trying forever until successful
- After reconnect, re-subscribe to market data
- No retroactive entries - if limit crossed while blind, just continue monitoring
"""

from __future__ import annotations

import logging
import threading
import time
from typing import TYPE_CHECKING, Optional, Callable, Set

if TYPE_CHECKING:
    from ib_insync import IB, Contract
    from strategy_engine import StrategyEngine


# Reconnection timing
INITIAL_BACKOFF_SECONDS = 5
MAX_BACKOFF_SECONDS = 120
BACKOFF_MULTIPLIER = 2


class ConnectionManager:
    """
    Manages IBKR connection with automatic reconnection.

    Philosophy: Treat a disconnect as temporary blindness.
    When vision returns, just keep watching. Don't guess what you missed.
    """

    def __init__(
        self,
        ib: "IB",
        logger: logging.Logger,
        host: str,
        port: int,
        client_id: int,
    ) -> None:
        self.ib = ib
        self.logger = logger
        self.host = host
        self.port = port
        self.client_id = client_id

        # Track reconnection state
        self._reconnecting = False
        self._reconnect_thread: Optional[threading.Thread] = None
        self._stop_requested = False
        self._lock = threading.Lock()

        # Callback to re-subscribe after reconnect
        self._on_reconnected: Optional[Callable[[], None]] = None

        # Track symbols to re-subscribe
        self._subscribed_symbols: Set[str] = set()

    def set_reconnected_callback(self, callback: Callable[[], None]) -> None:
        """Set callback to be called after successful reconnection."""
        self._on_reconnected = callback

    def set_subscribed_symbols(self, symbols: Set[str]) -> None:
        """Set the list of symbols to re-subscribe after reconnection."""
        self._subscribed_symbols = symbols.copy()

    def start(self) -> None:
        """
        Start monitoring the connection.

        Registers the disconnect handler with ib_insync.
        """
        self.ib.disconnectedEvent += self._on_disconnect
        self.logger.info("[CONN] ConnectionManager started. Monitoring for disconnects.")

    def stop(self) -> None:
        """Stop the connection manager and any reconnection attempts."""
        self._stop_requested = True
        if self._reconnect_thread is not None:
            self._reconnect_thread.join(timeout=5.0)
        self.logger.info("[CONN] ConnectionManager stopped.")

    def _on_disconnect(self) -> None:
        """Handle disconnect event from ib_insync."""
        with self._lock:
            if self._reconnecting:
                # Already handling reconnection
                return
            self._reconnecting = True

        self.logger.warning("[CONN] Disconnected from IBKR. Starting reconnection loop...")

        # Start reconnection in a background thread
        self._reconnect_thread = threading.Thread(
            target=self._reconnect_loop,
            daemon=True,
            name="IBKR-Reconnect",
        )
        self._reconnect_thread.start()

    def _reconnect_loop(self) -> None:
        """
        Attempt to reconnect with exponential backoff.

        Keeps trying forever until successful or stop is requested.
        """
        backoff = INITIAL_BACKOFF_SECONDS
        attempt = 0

        while not self._stop_requested:
            attempt += 1
            self.logger.info(
                "[CONN] Reconnection attempt #%d (backoff: %ds)...",
                attempt, backoff,
            )

            try:
                # Wait before attempting
                time.sleep(backoff)

                if self._stop_requested:
                    break

                # Attempt to connect
                self.ib.connect(self.host, self.port, clientId=self.client_id)

                if self.ib.isConnected():
                    self.logger.info(
                        "[CONN] Reconnected to IBKR successfully after %d attempts.",
                        attempt,
                    )

                    # Reset state
                    with self._lock:
                        self._reconnecting = False

                    # Notify callback (e.g., to re-subscribe to market data)
                    if self._on_reconnected:
                        try:
                            self._on_reconnected()
                        except Exception as e:
                            self.logger.exception(
                                "[CONN] Error in reconnected callback: %s", e
                            )

                    return  # Success - exit the loop

            except Exception as e:
                self.logger.warning(
                    "[CONN] Reconnection attempt #%d failed: %s",
                    attempt, e,
                )

            # Increase backoff for next attempt (capped at max)
            backoff = min(backoff * BACKOFF_MULTIPLIER, MAX_BACKOFF_SECONDS)

        # Stop was requested
        with self._lock:
            self._reconnecting = False
        self.logger.info("[CONN] Reconnection loop stopped by request.")

    def is_connected(self) -> bool:
        """Check if currently connected to IBKR."""
        return self.ib.isConnected()

    def is_reconnecting(self) -> bool:
        """Check if currently attempting to reconnect."""
        with self._lock:
            return self._reconnecting
