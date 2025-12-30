"""
state_manager.py

Phase 4: JSON persistence of caps and open-position counts.

Implements:
- Per-strategy caps (Day: 5/day, Swing: 5/week).
- Global caps (DayOpen <= 20, SwingOpen <= 15).
- SwingReserved: Reserved slots for re-entry candidates.
- Per-day and per-week buckets in logs/state.json:

  {
    "day": {
      "YYYY-MM-DD": {
        "strategy_fills": { "StrategyName": int, ... },
        "DayOpen": int,
        "StoppedToday": [ "SYM1", "SYM2", ... ],
        "SkippedToday": [ ["SYM", "reason"], ... ]
      },
      ...
    },
    "swing": {
      "YYYY-MM-DD": {
        "strategy_fills": { "StrategyName": int, ... },
        "SwingOpen": int,
        "SwingReserved": int
      },
      ...
    }
  }

Day keys = PT dates.
Swing keys = Monday of each PT week.

SwingReserved Slot System:
- Reserved slots are earmarked for specific re-entry candidates
- New swing entries use AVAILABLE slots (cap - open - reserved)
- Re-entry uses its RESERVED slot (always available if candidate exists)
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

STATE_PATH = Path("logs") / "state.json"

# Caps from spec
DAY_STRATEGY_CAP = 5
SWING_STRATEGY_CAP = 5
DAY_GLOBAL_CAP = 20
SWING_GLOBAL_CAP = 15


def monday_of_week(d: date) -> date:
    """
    Return the Monday (PT calendar) for the week containing date d.
    Monday has weekday() == 0.
    """
    return d - timedelta(days=d.weekday())


class BlockedEntries:
    """
    Tracks symbol+strategy combinations that are blocked from entry.

    Blocked entries are stored in state.json under "blocked_entries":
    {
        "week": {
            "SYMBOL_STRATEGYID": "2025-01-06"  # Monday of blocked week
        },
        "day": {
            "SYMBOL_STRATEGYID": "2025-01-08"  # Blocked date
        }
    }

    Used when:
    - flatten_position() fails: block symbol+strategy for rest of week
    - Day bracket placement fails: block for rest of week (swing re-entry)
      AND block for rest of day (day entry)
    - Gap trade placement fails: same as Day bracket failure
    """

    def __init__(self, state: Dict[str, Any], save_callback, logger: logging.Logger):
        self.state = state
        self._save = save_callback
        self.logger = logger

    def _get_blocked_bucket(self) -> Dict[str, Any]:
        """Get or create the blocked_entries section."""
        if "blocked_entries" not in self.state:
            self.state["blocked_entries"] = {"week": {}, "day": {}}
        blocked = self.state["blocked_entries"]
        if "week" not in blocked:
            blocked["week"] = {}
        if "day" not in blocked:
            blocked["day"] = {}
        return blocked

    def _make_key(self, symbol: str, strategy_id: str) -> str:
        """Create a unique key for symbol+strategy."""
        return f"{symbol.upper()}_{strategy_id}"

    def block_for_week(self, symbol: str, strategy_id: str, d: date) -> None:
        """Block symbol+strategy for the rest of the trading week."""
        blocked = self._get_blocked_bucket()
        key = self._make_key(symbol, strategy_id)
        monday = monday_of_week(d)
        blocked["week"][key] = monday.isoformat()
        self._save()
        self.logger.warning(
            "[BLOCKED] %s %s blocked for rest of week (week=%s)",
            symbol, strategy_id, monday,
        )

    def block_for_day(self, symbol: str, strategy_id: str, d: date) -> None:
        """Block symbol+strategy for the rest of the trading day."""
        blocked = self._get_blocked_bucket()
        key = self._make_key(symbol, strategy_id)
        blocked["day"][key] = d.isoformat()
        self._save()
        self.logger.warning(
            "[BLOCKED] %s %s blocked for rest of day (date=%s)",
            symbol, strategy_id, d,
        )

    def is_blocked_for_week(self, symbol: str, strategy_id: str, d: date) -> bool:
        """Check if symbol+strategy is blocked for the current week."""
        blocked = self._get_blocked_bucket()
        key = self._make_key(symbol, strategy_id)
        blocked_monday = blocked["week"].get(key)
        if blocked_monday is None:
            return False
        current_monday = monday_of_week(d)
        return blocked_monday == current_monday.isoformat()

    def is_blocked_for_day(self, symbol: str, strategy_id: str, d: date) -> bool:
        """Check if symbol+strategy is blocked for today."""
        blocked = self._get_blocked_bucket()
        key = self._make_key(symbol, strategy_id)
        blocked_date = blocked["day"].get(key)
        if blocked_date is None:
            return False
        return blocked_date == d.isoformat()

    def is_blocked(self, symbol: str, strategy_id: str, d: date, check_week: bool = True, check_day: bool = True) -> Tuple[bool, Optional[str]]:
        """
        Check if symbol+strategy is blocked.

        Args:
            symbol: Stock symbol
            strategy_id: Strategy identifier
            d: Current date
            check_week: Whether to check week-level blocks
            check_day: Whether to check day-level blocks

        Returns:
            (is_blocked, reason_or_None)
        """
        if check_week and self.is_blocked_for_week(symbol, strategy_id, d):
            return True, f"blocked for week (bracket/gap failure)"
        if check_day and self.is_blocked_for_day(symbol, strategy_id, d):
            return True, f"blocked for day (bracket/gap failure)"
        return False, None

    def cleanup_expired(self, d: date) -> None:
        """Remove expired blocks (from previous weeks/days)."""
        blocked = self._get_blocked_bucket()
        current_monday = monday_of_week(d)

        # Clean up week blocks from previous weeks
        expired_week = [
            key for key, monday_str in blocked["week"].items()
            if date.fromisoformat(monday_str) < current_monday
        ]
        for key in expired_week:
            del blocked["week"][key]

        # Clean up day blocks from previous days
        expired_day = [
            key for key, date_str in blocked["day"].items()
            if date.fromisoformat(date_str) < d
        ]
        for key in expired_day:
            del blocked["day"][key]

        if expired_week or expired_day:
            self._save()
            self.logger.info(
                "[BLOCKED] Cleaned up %d expired week blocks, %d expired day blocks",
                len(expired_week), len(expired_day),
            )


class PendingFlattens:
    """
    Tracks positions that need to be flattened but haven't been closed yet.

    Used when:
    - flatten_position() fails after all intraday retries
    - Day timed exit MKT order is cancelled/fails
    - Position needs to be flattened at next market open

    Stored in state.json under "pending_flattens":
    {
        "positions": [
            {
                "symbol": "AAPL",
                "side": "LONG",
                "qty": 100,
                "kind": "DAY",
                "strategy_id": "DayShort1",
                "reason": "timed_exit_cancelled",
                "created_at": "2025-01-08T13:00:00"
            },
            ...
        ]
    }
    """

    def __init__(self, state: Dict[str, Any], save_callback, logger: logging.Logger):
        self.state = state
        self._save = save_callback
        self.logger = logger

    def _get_bucket(self) -> Dict[str, Any]:
        """Get or create the pending_flattens section."""
        if "pending_flattens" not in self.state:
            self.state["pending_flattens"] = {"positions": []}
        bucket = self.state["pending_flattens"]
        if "positions" not in bucket:
            bucket["positions"] = []
        return bucket

    def add_pending_flatten(
        self,
        symbol: str,
        side: str,
        qty: int,
        kind: str,
        strategy_id: str,
        reason: str,
        parent_order_id: int = 0,
    ) -> None:
        """Add a position that needs to be flattened."""
        from time_utils import now_pt

        bucket = self._get_bucket()

        # Check if already exists (avoid duplicates)
        for pos in bucket["positions"]:
            if pos["symbol"] == symbol and pos["strategy_id"] == strategy_id:
                # Update existing entry
                pos["qty"] = qty
                pos["reason"] = reason
                pos["parent_order_id"] = parent_order_id
                pos["created_at"] = now_pt().isoformat()
                self._save()
                self.logger.info(
                    "[PENDING_FLATTEN] Updated pending flatten for %s %s (qty=%d, reason=%s)",
                    symbol, strategy_id, qty, reason,
                )
                return

        # Add new entry
        bucket["positions"].append({
            "symbol": symbol.upper(),
            "side": side.upper(),
            "qty": qty,
            "kind": kind.upper(),
            "strategy_id": strategy_id,
            "reason": reason,
            "parent_order_id": parent_order_id,
            "created_at": now_pt().isoformat(),
        })
        self._save()
        self.logger.warning(
            "[PENDING_FLATTEN] Added pending flatten for %s %s (qty=%d, reason=%s, parent_order_id=%d)",
            symbol, strategy_id, qty, reason, parent_order_id,
        )

    def remove_pending_flatten(self, symbol: str, strategy_id: str) -> bool:
        """Remove a pending flatten (position was closed). Returns True if found."""
        bucket = self._get_bucket()
        for i, pos in enumerate(bucket["positions"]):
            if pos["symbol"] == symbol.upper() and pos["strategy_id"] == strategy_id:
                del bucket["positions"][i]
                self._save()
                self.logger.info(
                    "[PENDING_FLATTEN] Removed pending flatten for %s %s (position closed)",
                    symbol, strategy_id,
                )
                return True
        return False

    def get_all_pending(self) -> List[Dict[str, Any]]:
        """Get all pending flattens with schema validation. Removes invalid entries."""
        bucket = self._get_bucket()
        required_keys = {"symbol", "side", "qty", "kind", "strategy_id"}
        valid_positions = []
        invalid_count = 0

        for pos in bucket["positions"]:
            # Validate required fields exist
            if not required_keys.issubset(pos.keys()):
                missing = required_keys - set(pos.keys())
                self.logger.warning(
                    "[PENDING_FLATTEN] Removing invalid entry (missing keys: %s): %s",
                    missing, pos.get("symbol", "unknown"),
                )
                invalid_count += 1
                continue
            valid_positions.append(pos)

        # Remove invalid entries from state if any were found
        if invalid_count > 0:
            bucket["positions"] = valid_positions
            self._save()
            self.logger.info(
                "[PENDING_FLATTEN] Removed %d invalid entries from state",
                invalid_count,
            )

        return valid_positions

    def has_pending(self, symbol: str, strategy_id: str) -> bool:
        """Check if a symbol+strategy has a pending flatten."""
        bucket = self._get_bucket()
        for pos in bucket["positions"]:
            if pos["symbol"] == symbol.upper() and pos["strategy_id"] == strategy_id:
                return True
        return False


class PendingGapOrders:
    """
    Tracks gap MKT orders awaiting complete fill to finish bracket.

    After a gap condition triggers, a single-leg MKT order is placed.
    Once filled, stop and timed exit orders are added to complete the bracket.

    Stored in state.json under "pending_gap_orders":
    {
        "orders": [
            {
                "order_id": 123,
                "symbol": "AAPL",
                "strategy_id": "DayLong1",
                "signal_type": "DAY",
                "direction": "LONG",
                "stop_distance": 1.25,
                "shares": 100,
                "created_at": "2025-01-08T06:30:00"
            }
        ]
    }
    """

    def __init__(self, state: Dict[str, Any], save_callback, logger: logging.Logger):
        self.state = state
        self._save = save_callback
        self.logger = logger

    def _get_bucket(self) -> Dict[str, Any]:
        """Get or create the pending_gap_orders section."""
        if "pending_gap_orders" not in self.state:
            self.state["pending_gap_orders"] = {"orders": []}
        bucket = self.state["pending_gap_orders"]
        if "orders" not in bucket:
            bucket["orders"] = []
        return bucket

    def add_pending(
        self,
        order_id: int,
        symbol: str,
        strategy_id: str,
        signal_type: str,
        direction: str,
        stop_distance: float,
        shares: int,
    ) -> None:
        """Add a pending gap order awaiting fill."""
        from time_utils import now_pt

        bucket = self._get_bucket()

        # Check for duplicates
        for order in bucket["orders"]:
            if order["order_id"] == order_id:
                self.logger.warning(
                    "[PENDING_GAP] Order %d already pending, skipping add", order_id
                )
                return

        bucket["orders"].append({
            "order_id": order_id,
            "symbol": symbol.upper(),
            "strategy_id": strategy_id,
            "signal_type": signal_type.upper(),
            "direction": direction.upper(),
            "stop_distance": stop_distance,
            "shares": shares,
            "created_at": now_pt().isoformat(),
        })
        self._save()
        self.logger.warning(
            "[PENDING_GAP] Added pending gap order %d for %s %s (UNPROTECTED until fill)",
            order_id, symbol, strategy_id,
        )

    def remove_pending(self, order_id: int) -> Optional[Dict[str, Any]]:
        """Remove and return a pending gap order. Returns None if not found."""
        bucket = self._get_bucket()
        for i, order in enumerate(bucket["orders"]):
            if order["order_id"] == order_id:
                removed = bucket["orders"].pop(i)
                self._save()
                self.logger.info(
                    "[PENDING_GAP] Removed pending gap order %d for %s",
                    order_id, removed["symbol"],
                )
                return removed
        return None

    def get_pending(self, order_id: int) -> Optional[Dict[str, Any]]:
        """Get a pending gap order by order_id without removing it."""
        bucket = self._get_bucket()
        for order in bucket["orders"]:
            if order["order_id"] == order_id:
                return order
        return None

    def get_all_pending(self) -> List[Dict[str, Any]]:
        """Get all pending gap orders."""
        bucket = self._get_bucket()
        return list(bucket["orders"])

    def is_pending_gap_order(self, order_id: int) -> bool:
        """Check if an order_id is a pending gap order."""
        return self.get_pending(order_id) is not None

    def cleanup_stale(self, max_age_hours: int = 24) -> List[Dict[str, Any]]:
        """Remove pending orders older than max_age_hours. Returns removed orders."""
        from time_utils import now_pt
        from datetime import datetime

        bucket = self._get_bucket()
        now = now_pt()
        stale = []
        remaining = []

        for order in bucket["orders"]:
            try:
                created = datetime.fromisoformat(order["created_at"])
                age_hours = (now - created).total_seconds() / 3600
                if age_hours > max_age_hours:
                    stale.append(order)
                else:
                    remaining.append(order)
            except (ValueError, KeyError):
                stale.append(order)

        if stale:
            bucket["orders"] = remaining
            self._save()
            for order in stale:
                self.logger.warning(
                    "[PENDING_GAP] Cleaned up stale pending order %d for %s (age exceeded)",
                    order.get("order_id", 0), order.get("symbol", "?"),
                )

        return stale


class SignalCache:
    """
    Caches loaded signals with pre-calculated stop distances.

    Signals are cached per-date (day signals) or per-week (swing signals)
    and persist until a new CSV replaces them.

    Stored in state.json under "signal_cache":
    {
        "day": {
            "YYYY-MM-DD": {
                "source_file": "daytrades_20250108.csv",
                "source_hash": "abc123...",
                "loaded_at": "2025-01-08T06:00:00",
                "signals": [
                    {
                        "strategy_id": "DayLong1",
                        "symbol": "AAPL",
                        "entry_price": 150.0,
                        "stop_price": 149.0,
                        "stop_distance": 1.0,
                        ...
                    },
                    ...
                ]
            }
        },
        "swing": {
            "YYYY-MM-DD": {  # Monday of week
                "source_file": "swingtrades_20250106.csv",
                "source_hash": "def456...",
                "loaded_at": "2025-01-06T06:00:00",
                "signals": [...]
            }
        }
    }
    """

    def __init__(self, state: Dict[str, Any], save_callback, logger: logging.Logger):
        self.state = state
        self._save = save_callback
        self.logger = logger

    def _get_cache_bucket(self) -> Dict[str, Any]:
        """Get or create the signal_cache section."""
        if "signal_cache" not in self.state:
            self.state["signal_cache"] = {"day": {}, "swing": {}}
        cache = self.state["signal_cache"]
        if "day" not in cache:
            cache["day"] = {}
        if "swing" not in cache:
            cache["swing"] = {}
        return cache

    def _compute_file_hash(self, file_path: Path) -> str:
        """Compute a simple hash of CSV file contents for change detection."""
        import hashlib
        try:
            with file_path.open("rb") as f:
                return hashlib.md5(f.read()).hexdigest()[:16]
        except Exception:
            return ""

    def get_cached_day_signals(self, d: date, source_file: Path) -> Optional[List[Dict[str, Any]]]:
        """
        Get cached day signals if the cache is valid.

        Returns None if:
        - No cache exists for this date
        - Source file has changed (hash mismatch)
        - Cache is from a different source file
        """
        cache = self._get_cache_bucket()
        key = d.isoformat()
        entry = cache["day"].get(key)

        if entry is None:
            return None

        # Check if source file matches
        if entry.get("source_file") != source_file.name:
            self.logger.info(
                "[CACHE] Day cache source mismatch for %s: cached=%s, current=%s",
                key, entry.get("source_file"), source_file.name,
            )
            return None

        # Check if file hash matches (file hasn't changed)
        current_hash = self._compute_file_hash(source_file)
        if entry.get("source_hash") != current_hash:
            self.logger.info(
                "[CACHE] Day cache hash mismatch for %s (file changed), reloading",
                key,
            )
            return None

        self.logger.info(
            "[CACHE] Using cached day signals for %s (%d signals)",
            key, len(entry.get("signals", [])),
        )
        return entry.get("signals", [])

    def get_cached_swing_signals(self, d: date, source_file: Path) -> Optional[List[Dict[str, Any]]]:
        """
        Get cached swing signals if the cache is valid.

        Returns None if cache is invalid or missing.
        """
        cache = self._get_cache_bucket()
        monday = monday_of_week(d)
        key = monday.isoformat()
        entry = cache["swing"].get(key)

        if entry is None:
            return None

        # Check if source file matches
        if entry.get("source_file") != source_file.name:
            self.logger.info(
                "[CACHE] Swing cache source mismatch for %s: cached=%s, current=%s",
                key, entry.get("source_file"), source_file.name,
            )
            return None

        # Check if file hash matches
        current_hash = self._compute_file_hash(source_file)
        if entry.get("source_hash") != current_hash:
            self.logger.info(
                "[CACHE] Swing cache hash mismatch for %s (file changed), reloading",
                key,
            )
            return None

        self.logger.info(
            "[CACHE] Using cached swing signals for %s (%d signals)",
            key, len(entry.get("signals", [])),
        )
        return entry.get("signals", [])

    def cache_day_signals(self, d: date, source_file: Path, signals: List[Any]) -> None:
        """Cache day signals with their pre-calculated stop distances."""
        from dataclasses import asdict
        from time_utils import now_pt

        cache = self._get_cache_bucket()
        key = d.isoformat()

        # Convert signals to dicts, handling date serialization
        signal_dicts = []
        for sig in signals:
            sig_dict = asdict(sig)
            # Convert date to string for JSON serialization
            if "trade_date" in sig_dict and hasattr(sig_dict["trade_date"], "isoformat"):
                sig_dict["trade_date"] = sig_dict["trade_date"].isoformat()
            signal_dicts.append(sig_dict)

        cache["day"][key] = {
            "source_file": source_file.name,
            "source_hash": self._compute_file_hash(source_file),
            "loaded_at": now_pt().isoformat(),
            "signals": signal_dicts,
        }
        self._save()
        self.logger.info(
            "[CACHE] Cached %d day signals for %s (file=%s)",
            len(signals), key, source_file.name,
        )

    def cache_swing_signals(self, d: date, source_file: Path, signals: List[Any]) -> None:
        """Cache swing signals with their pre-calculated stop distances."""
        from dataclasses import asdict
        from time_utils import now_pt

        cache = self._get_cache_bucket()
        monday = monday_of_week(d)
        key = monday.isoformat()

        # Convert signals to dicts
        signal_dicts = []
        for sig in signals:
            sig_dict = asdict(sig)
            signal_dicts.append(sig_dict)

        cache["swing"][key] = {
            "source_file": source_file.name,
            "source_hash": self._compute_file_hash(source_file),
            "loaded_at": now_pt().isoformat(),
            "signals": signal_dicts,
        }
        self._save()
        self.logger.info(
            "[CACHE] Cached %d swing signals for week of %s (file=%s)",
            len(signals), key, source_file.name,
        )

    def get_day_signal_by_key(self, d: date, symbol: str, strategy_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific cached day signal by symbol and strategy."""
        cache = self._get_cache_bucket()
        key = d.isoformat()
        entry = cache["day"].get(key)
        if entry is None:
            return None

        for sig in entry.get("signals", []):
            if sig.get("symbol") == symbol and sig.get("strategy_id") == strategy_id:
                return sig
        return None

    def get_swing_signal_by_key(self, d: date, symbol: str, strategy_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific cached swing signal by symbol and strategy."""
        cache = self._get_cache_bucket()
        monday = monday_of_week(d)
        key = monday.isoformat()
        entry = cache["swing"].get(key)
        if entry is None:
            return None

        for sig in entry.get("signals", []):
            if sig.get("symbol") == symbol and sig.get("strategy_id") == strategy_id:
                return sig
        return None

    def cleanup_old_cache(self, d: date, keep_days: int = 7) -> None:
        """Remove cache entries older than keep_days."""
        cache = self._get_cache_bucket()
        cutoff = d - timedelta(days=keep_days)

        # Clean day cache
        expired_day = [
            key for key in cache["day"]
            if date.fromisoformat(key) < cutoff
        ]
        for key in expired_day:
            del cache["day"][key]

        # Clean swing cache
        expired_swing = [
            key for key in cache["swing"]
            if date.fromisoformat(key) < cutoff
        ]
        for key in expired_swing:
            del cache["swing"][key]

        if expired_day or expired_swing:
            self._save()
            self.logger.info(
                "[CACHE] Cleaned up %d day cache entries, %d swing cache entries",
                len(expired_day), len(expired_swing),
            )

    # === Signal Fill Tracking ===

    def is_signal_filled(
        self, d: date, symbol: str, strategy_id: str, signal_type: str
    ) -> bool:
        """
        Check if a signal has been filled (order placed).

        A filled signal should not be traded again until the CSV is replaced.

        Args:
            d: Trade date (day signals) or any date in the week (swing signals)
            symbol: Stock symbol
            strategy_id: Strategy identifier
            signal_type: "DAY" or "SWING"

        Returns:
            True if signal was already filled, False otherwise
        """
        cache = self._get_cache_bucket()

        if signal_type.upper() == "DAY":
            key = d.isoformat()
            entry = cache["day"].get(key)
        else:
            monday = monday_of_week(d)
            key = monday.isoformat()
            entry = cache["swing"].get(key)

        if entry is None:
            return False

        for sig in entry.get("signals", []):
            if sig.get("symbol") == symbol.upper() and sig.get("strategy_id") == strategy_id:
                return sig.get("filled", False)

        return False

    def mark_signal_filled(
        self,
        d: date,
        symbol: str,
        strategy_id: str,
        signal_type: str,
        filled: bool = True,
    ) -> bool:
        """
        Mark a signal as filled (order placed) or unfilled (placement failed).

        Used for optimistic locking - mark BEFORE placing order, unmark if fails.

        Args:
            d: Trade date (day signals) or any date in the week (swing signals)
            symbol: Stock symbol
            strategy_id: Strategy identifier
            signal_type: "DAY" or "SWING"
            filled: True to mark as filled, False to unmark

        Returns:
            True if signal was found and updated, False if not found
        """
        cache = self._get_cache_bucket()
        symbol = symbol.upper()

        if signal_type.upper() == "DAY":
            key = d.isoformat()
            entry = cache["day"].get(key)
        else:
            monday = monday_of_week(d)
            key = monday.isoformat()
            entry = cache["swing"].get(key)

        if entry is None:
            self.logger.warning(
                "[CACHE] Cannot mark signal - no cache entry for %s %s on %s",
                signal_type, symbol, key,
            )
            return False

        for sig in entry.get("signals", []):
            if sig.get("symbol") == symbol and sig.get("strategy_id") == strategy_id:
                sig["filled"] = filled
                self._save()
                self.logger.info(
                    "[CACHE] Marked %s %s %s as %s",
                    signal_type, symbol, strategy_id,
                    "FILLED" if filled else "UNFILLED",
                )
                return True

        self.logger.warning(
            "[CACHE] Cannot mark signal - not found: %s %s %s on %s",
            signal_type, symbol, strategy_id, key,
        )
        return False

    def get_unfilled_signals_count(self, d: date, signal_type: str) -> int:
        """Get count of signals that haven't been filled yet."""
        cache = self._get_cache_bucket()

        if signal_type.upper() == "DAY":
            key = d.isoformat()
            entry = cache["day"].get(key)
        else:
            monday = monday_of_week(d)
            key = monday.isoformat()
            entry = cache["swing"].get(key)

        if entry is None:
            return 0

        return sum(1 for sig in entry.get("signals", []) if not sig.get("filled", False))


class StateManager:
    """
    Manages persistent caps & open-position counts in logs/state.json.

    Public methods (used later by strategy engine / fills):
      - get_day_state(date) -> dict
      - get_swing_state(date) -> (monday_date, dict)

      - can_open_day(strategy_id, date) -> (bool, reason_or_None)
      - can_open_swing(strategy_id, date) -> (bool, reason_or_None)

      - register_day_entry(strategy_id, date)
      - register_swing_entry(strategy_id, date)

      - register_day_exit(date)
      - register_swing_exit(date)

      - record_day_stop(symbol, date)
      - record_day_skip(symbol, reason, date)
    """

    def __init__(self, logger: logging.Logger) -> None:
        self.logger = logger
        STATE_PATH.parent.mkdir(exist_ok=True)
        self.path = STATE_PATH
        self.state: Dict[str, Any] = {"day": {}, "swing": {}}
        self._load()
        # Ensure there is always a state.json file after init
        self._ensure_file_exists()
        # Initialize blocked entries tracker
        self.blocked = BlockedEntries(self.state, self._save, self.logger)
        # Initialize pending flattens tracker
        self.pending_flattens = PendingFlattens(self.state, self._save, self.logger)
        # Initialize pending gap orders tracker
        self.pending_gap_orders = PendingGapOrders(self.state, self._save, self.logger)
        # Initialize signal cache for pre-calculated stop distances
        self.signal_cache = SignalCache(self.state, self._save, self.logger)
        # Run startup cleanup
        self._startup_cleanup()

    def _startup_cleanup(self) -> None:
        """
        Clean up stale data on startup.

        - Pending gap orders older than 9 days (orphaned from crashes)
        - Expired blocked entries (from previous weeks/days)
        - Old signal cache entries (older than 7 days)
        """
        from time_utils import now_pt

        today = now_pt().date()

        # Clean up stale pending gap orders (9 days = 216 hours)
        stale_gap_orders = self.pending_gap_orders.cleanup_stale(max_age_hours=216)
        if stale_gap_orders:
            self.logger.warning(
                "[STARTUP] Cleaned up %d stale pending gap orders (>9 days old)",
                len(stale_gap_orders),
            )

        # Clean up expired blocked entries
        self.blocked.cleanup_expired(today)

        # Clean up old signal cache entries
        self.signal_cache.cleanup_old_cache(today, keep_days=7)

    # --- internal load/save ---

    def _load(self) -> None:
        if self.path.exists():
            try:
                with self.path.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    self.state["day"] = data.get("day", {}) or {}
                    self.state["swing"] = data.get("swing", {}) or {}
                    self.state["blocked_entries"] = data.get("blocked_entries", {"week": {}, "day": {}})
                    self.state["pending_flattens"] = data.get("pending_flattens", {"positions": []})
                    self.state["pending_gap_orders"] = data.get("pending_gap_orders", {"orders": []})
                    self.state["signal_cache"] = data.get("signal_cache", {"day": {}, "swing": {}})
                else:
                    raise ValueError("state.json root is not a dict")
                self.logger.info("[SYNC] Loaded state from %s.", self.path)
            except Exception as exc:
                self.logger.warning(
                    "[WARN] Failed to load state file %s, starting fresh: %s",
                    self.path,
                    exc,
                )
                self.state = {"day": {}, "swing": {}, "blocked_entries": {"week": {}, "day": {}}, "pending_flattens": {"positions": []}, "pending_gap_orders": {"orders": []}, "signal_cache": {"day": {}, "swing": {}}}
        else:
            self.logger.info("[SYNC] No existing state file; starting fresh.")
            self.state = {"day": {}, "swing": {}, "blocked_entries": {"week": {}, "day": {}}, "pending_flattens": {"positions": []}, "pending_gap_orders": {"orders": []}, "signal_cache": {"day": {}, "swing": {}}}

    def _ensure_file_exists(self) -> None:
        """
        If state.json does not exist yet, write the current (possibly empty) state.
        This guarantees the file shows up even before any trades occur.
        """
        if not self.path.exists():
            self._save()

    def _save(self) -> None:
        """Atomically save state to file using write-to-temp-then-rename pattern."""
        try:
            # Write to a temporary file in the same directory (ensures same filesystem)
            dir_path = self.path.parent
            dir_path.mkdir(parents=True, exist_ok=True)

            # Create temp file in same directory for atomic rename
            fd, tmp_path = tempfile.mkstemp(dir=str(dir_path), suffix=".tmp")
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    json.dump(self.state, f, indent=2, sort_keys=True)
                # Atomic rename (on POSIX systems)
                os.replace(tmp_path, str(self.path))
            except Exception:
                # Clean up temp file on failure
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                raise
        except Exception as exc:
            self.logger.warning(
                "[WARN] Failed to write state file %s: %s", self.path, exc
            )

    # --- bucket helpers ---

    def _get_day_bucket(self, d: date) -> Dict[str, Any]:
        day_state = self.state.setdefault("day", {})
        key = d.isoformat()
        if key not in day_state:
            day_state[key] = {
                "strategy_fills": {},
                "DayOpen": 0,
                "StoppedToday": [],
                "SkippedToday": [],
            }
        return day_state[key]

    def _get_swing_bucket(self, d: date) -> Dict[str, Any]:
        swing_state = self.state.setdefault("swing", {})
        monday = monday_of_week(d)
        key = monday.isoformat()
        if key not in swing_state:
            swing_state[key] = {
                "strategy_fills": {},
                "SwingOpen": 0,
                "SwingReserved": 0,
            }
        else:
            # Ensure SwingReserved exists for older buckets
            if "SwingReserved" not in swing_state[key]:
                swing_state[key]["SwingReserved"] = 0
        return swing_state[key]

    # --- public accessors for inspection/logging ---

    def get_day_state(self, d: date) -> Dict[str, Any]:
        return self._get_day_bucket(d)

    def get_swing_state(self, d: date) -> Tuple[date, Dict[str, Any]]:
        monday = monday_of_week(d)
        bucket = self._get_swing_bucket(d)
        return monday, bucket

    # --- caps / open-count checks ---

    def can_open_day(self, strategy_id: str, d: date) -> Tuple[bool, Optional[str]]:
        bucket = self._get_day_bucket(d)
        fills = int(bucket["strategy_fills"].get(strategy_id, 0))
        open_count = int(bucket.get("DayOpen", 0))

        if fills >= DAY_STRATEGY_CAP:
            return False, "strategy cap reached"
        if open_count >= DAY_GLOBAL_CAP:
            return False, "global cap reached"
        return True, None

    def can_open_swing(self, strategy_id: str, d: date) -> Tuple[bool, Optional[str]]:
        """Check if a NEW swing entry is allowed (uses available slots)."""
        bucket = self._get_swing_bucket(d)
        fills = int(bucket["strategy_fills"].get(strategy_id, 0))
        open_count = int(bucket.get("SwingOpen", 0))
        reserved = int(bucket.get("SwingReserved", 0))

        if fills >= SWING_STRATEGY_CAP:
            return False, "strategy cap reached"

        # New entries use AVAILABLE slots: cap - open - reserved
        available = SWING_GLOBAL_CAP - open_count - reserved
        if available <= 0:
            return False, "global cap reached (no available slots)"
        return True, None

    def can_reentry_swing(self, d: date) -> Tuple[bool, Optional[str]]:
        """Check if a re-entry can use its reserved slot."""
        bucket = self._get_swing_bucket(d)
        reserved = int(bucket.get("SwingReserved", 0))

        if reserved <= 0:
            return False, "no reserved slots available"
        return True, None

    def get_swing_available(self, d: date) -> int:
        """Return number of available slots for NEW swing entries."""
        bucket = self._get_swing_bucket(d)
        open_count = int(bucket.get("SwingOpen", 0))
        reserved = int(bucket.get("SwingReserved", 0))
        return max(0, SWING_GLOBAL_CAP - open_count - reserved)

    def get_swing_reserved(self, d: date) -> int:
        """Return number of reserved slots for re-entry."""
        bucket = self._get_swing_bucket(d)
        return int(bucket.get("SwingReserved", 0))

    # --- mutating ops: slot reservation ---

    def reserve_swing_slot(self, d: date) -> None:
        """Reserve a swing slot for a re-entry candidate."""
        bucket = self._get_swing_bucket(d)
        bucket["SwingReserved"] = int(bucket.get("SwingReserved", 0)) + 1
        self._save()
        self.logger.info(
            "[STATE] Reserved swing slot. SwingReserved=%d, SwingOpen=%d, Available=%d",
            bucket["SwingReserved"],
            bucket.get("SwingOpen", 0),
            self.get_swing_available(d),
        )

    def release_swing_slot(self, d: date) -> None:
        """Release a reserved swing slot (candidate expired or filled)."""
        bucket = self._get_swing_bucket(d)
        reserved = int(bucket.get("SwingReserved", 0))
        bucket["SwingReserved"] = max(0, reserved - 1)
        self._save()
        self.logger.info(
            "[STATE] Released swing slot. SwingReserved=%d, SwingOpen=%d, Available=%d",
            bucket["SwingReserved"],
            bucket.get("SwingOpen", 0),
            self.get_swing_available(d),
        )

    def convert_reserved_to_open(self, d: date) -> None:
        """Convert a reserved slot to an open position (re-entry filled)."""
        bucket = self._get_swing_bucket(d)
        reserved = int(bucket.get("SwingReserved", 0))
        if reserved > 0:
            bucket["SwingReserved"] = reserved - 1
            bucket["SwingOpen"] = int(bucket.get("SwingOpen", 0)) + 1
            self._save()
            self.logger.info(
                "[STATE] Converted reserved to open. SwingReserved=%d, SwingOpen=%d, Available=%d",
                bucket["SwingReserved"],
                bucket["SwingOpen"],
                self.get_swing_available(d),
            )

    # --- mutating ops: entries / exits ---

    def register_day_entry(self, strategy_id: str, d: date) -> None:
        bucket = self._get_day_bucket(d)
        fills = int(bucket["strategy_fills"].get(strategy_id, 0)) + 1
        bucket["strategy_fills"][strategy_id] = fills
        bucket["DayOpen"] = int(bucket.get("DayOpen", 0)) + 1
        self._save()

    def register_swing_entry(self, strategy_id: str, d: date) -> None:
        bucket = self._get_swing_bucket(d)
        fills = int(bucket["strategy_fills"].get(strategy_id, 0)) + 1
        bucket["strategy_fills"][strategy_id] = fills
        bucket["SwingOpen"] = int(bucket.get("SwingOpen", 0)) + 1
        self._save()

    def register_day_exit(self, d: date) -> None:
        bucket = self._get_day_bucket(d)
        bucket["DayOpen"] = max(0, int(bucket.get("DayOpen", 0)) - 1)
        self._save()

    def register_swing_exit(self, d: date) -> None:
        bucket = self._get_swing_bucket(d)
        bucket["SwingOpen"] = max(0, int(bucket.get("SwingOpen", 0)) - 1)
        self._save()

    # --- stops & skips (Day) ---

    def record_day_stop(self, symbol: str, d: date) -> None:
        bucket = self._get_day_bucket(d)
        stopped: List[str] = bucket.get("StoppedToday", [])
        if symbol not in stopped:
            stopped.append(symbol)
            bucket["StoppedToday"] = stopped
            self._save()

    def record_day_skip(self, symbol: str, reason: str, d: date) -> None:
        bucket = self._get_day_bucket(d)
        skipped: List[Any] = bucket.get("SkippedToday", [])
        skipped.append([symbol, reason])
        bucket["SkippedToday"] = skipped
        self._save()
