"""
signals.py

Phase 3: Signal dataclasses + CSV loader for Day and Swing trades.

- Reads CSVs from BasketLists/ (creating the folder if missing).
- Normalizes rows into DaySignal and SwingSignal objects.
- Applies fixed-per-position budgets and computes integer share size.
- Skips rows where shares == 0 (too expensive) with [SKIP] logs.
- No IBKR interaction here; pure data layer.

Day CSV layout (per spec):
  Filename: daytrades_YYYYMMDD.csv
  Columns: Strategy, Date, Symbol, LmtPrice, AuxPrice

Swing CSV layout (per spec):
  Filename: swingtrades_YYYYMMDD.csv
  Columns: Strategy, Symbol, Entry, Stop
"""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import List


# === Config ===

BASKET_DIR = Path("BasketLists")
BASKET_DIR.mkdir(exist_ok=True)

DAY_BUDGET_PER_POSITION = 5000.0   # USD, per the spec
SWING_BUDGET_PER_POSITION = 5000.0 # USD, per the spec


@dataclass
class DaySignal:
    strategy_id: str
    symbol: str
    trade_date: date
    direction: str          # "LONG" or "SHORT"
    entry_price: float      # LmtPrice
    stop_price: float       # AuxPrice
    shares: int             # integer share size
    source_file: str        # CSV filename for audit
    stop_distance: float = 0.0  # Pre-calculated |entry - stop|, persisted for gap trades


@dataclass
class SwingSignal:
    strategy_id: str        # "MOMO" | "Mean Reversion" | "Pullback"
    symbol: str
    direction: str          # always "LONG" for Swing per spec
    entry_price: float      # Entry
    stop_price: float       # Stop
    shares: int
    source_file: str        # CSV filename for audit
    stop_distance: float = 0.0  # Pre-calculated |entry - stop|, persisted for gap trades


class CsvLoader:
    """
    CSV loader and normalizer.

    It logs using the provided logger with tags:
      [CSV] for load operations
      [SKIP] for rows skipped due to qty=0 or parse errors
    """

    def __init__(self, logger: logging.Logger) -> None:
        self.logger = logger

    # --- internal helpers ---

    def _log(self, tag: str, msg: str) -> None:
        self.logger.info(f"[{tag}] {msg}")

    @staticmethod
    def _infer_direction_from_strategy(strategy: str) -> str:
        """
        Infer LONG/SHORT from the strategy name.
        If it contains 'short' (case-insensitive) => SHORT, else LONG.
        """
        if "short" in strategy.lower():
            return "SHORT"
        return "LONG"

    @staticmethod
    def _parse_float(value: str) -> float:
        return float(value.strip())

    # --- public API ---

    def load_day_signals(self, trade_date: date) -> List[DaySignal]:
        """
        Load Day signals for a given PT trade date.

        Expects a file:
          BasketLists/daytrades_YYYYMMDD.csv

        Returns a list of DaySignal objects.
        """
        filename = f"daytrades_{trade_date.strftime('%Y%m%d')}.csv"
        path = BASKET_DIR / filename

        if not path.exists():
            self._log(
                "CSV",
                f"Day CSV not found for {trade_date.isoformat()} ({filename}); 0 signals loaded.",
            )
            return []

        signals: List[DaySignal] = []
        loaded_rows = 0
        skipped_rows = 0

        with path.open(newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row_num, row in enumerate(reader, start=2):  # start=2 to account for header row
                loaded_rows += 1
                try:
                    strategy = (row.get("Strategy") or "").strip()
                    date_str = (row.get("Date") or "").strip()
                    symbol = (row.get("Symbol") or "").strip().upper()
                    lmt_str = (row.get("LmtPrice") or "").strip()
                    aux_str = (row.get("AuxPrice") or "").strip()

                    if not (strategy and date_str and symbol and lmt_str and aux_str):
                        skipped_rows += 1
                        self._log(
                            "SKIP",
                            f"Day CSV row {row_num}: missing required fields; skipping.",
                        )
                        continue

                    parsed_date = datetime.strptime(date_str, "%Y-%m-%d").date()
                    entry_price = self._parse_float(lmt_str)
                    stop_price = self._parse_float(aux_str)

                    # Validate prices before division
                    if entry_price <= 0:
                        skipped_rows += 1
                        self._log(
                            "SKIP",
                            f"Day {symbol} (strategy={strategy}) invalid entry_price={entry_price}; skipping.",
                        )
                        continue

                    direction = self._infer_direction_from_strategy(strategy)

                    # Position sizing
                    shares = int(DAY_BUDGET_PER_POSITION // entry_price)
                    if shares <= 0:
                        skipped_rows += 1
                        self._log(
                            "SKIP",
                            f"Day {symbol} (strategy={strategy}) qty=0 (too expensive); skipping.",
                        )
                        continue

                    # Pre-calculate stop distance for gap trades
                    stop_distance = abs(entry_price - stop_price)

                    signal = DaySignal(
                        strategy_id=strategy,
                        symbol=symbol,
                        trade_date=parsed_date,
                        direction=direction,
                        entry_price=entry_price,
                        stop_price=stop_price,
                        shares=shares,
                        source_file=filename,
                        stop_distance=stop_distance,
                    )
                    signals.append(signal)

                except Exception as exc:
                    skipped_rows += 1
                    self._log(
                        "SKIP",
                        f"Day CSV row {row_num} in {filename} parse error: {exc}; skipping.",
                    )

        self._log(
            "CSV",
            f"Loaded {len(signals)} Day signals from {filename} "
            f"({loaded_rows} rows, {skipped_rows} skipped).",
        )
        return signals

    def load_swing_signals(self, week_date: date) -> List[SwingSignal]:
        """
        Load Swing signals for a given date.

        Expects a file:
          BasketLists/swingtrades_YYYYMMDD.csv

        The date is used only to determine the filename; the spec
        arms these weekly on Mondays, but this loader is date-agnostic.
        """
        filename = f"swingtrades_{week_date.strftime('%YMMDD')}.csv"
        # NOTE: The above has MM vs mm — we'll fix to %Y%m%d now:
        filename = f"swingtrades_{week_date.strftime('%Y%m%d')}.csv"
        path = BASKET_DIR / filename

        if not path.exists():
            self._log(
                "CSV",
                f"Swing CSV not found for {week_date.isoformat()} ({filename}); 0 signals loaded.",
            )
            return []

        signals: List[SwingSignal] = []
        loaded_rows = 0
        skipped_rows = 0

        with path.open(newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row_num, row in enumerate(reader, start=2):
                loaded_rows += 1
                try:
                    strategy = (row.get("Strategy") or "").strip()
                    symbol = (row.get("Symbol") or "").strip().upper()
                    entry_str = (row.get("Entry") or "").strip()
                    stop_str = (row.get("Stop") or "").strip()

                    if not (strategy and symbol and entry_str and stop_str):
                        skipped_rows += 1
                        self._log(
                            "SKIP",
                            f"Swing CSV row {row_num}: missing required fields; skipping.",
                        )
                        continue

                    entry_price = self._parse_float(entry_str)
                    stop_price = self._parse_float(stop_str)

                    # Validate prices before division
                    if entry_price <= 0:
                        skipped_rows += 1
                        self._log(
                            "SKIP",
                            f"Swing {symbol} (strategy={strategy}) invalid entry_price={entry_price}; skipping.",
                        )
                        continue

                    # Swings are long-only per spec
                    direction = "LONG"

                    # Position sizing
                    shares = int(SWING_BUDGET_PER_POSITION // entry_price)
                    if shares <= 0:
                        skipped_rows += 1
                        self._log(
                            "SKIP",
                            f"Swing {symbol} (strategy={strategy}) qty=0 (too expensive); skipping.",
                        )
                        continue

                    # Pre-calculate stop distance for gap trades
                    stop_distance = abs(entry_price - stop_price)

                    signal = SwingSignal(
                        strategy_id=strategy,
                        symbol=symbol,
                        direction=direction,
                        entry_price=entry_price,
                        stop_price=stop_price,
                        shares=shares,
                        source_file=filename,
                        stop_distance=stop_distance,
                    )
                    signals.append(signal)

                except Exception as exc:
                    skipped_rows += 1
                    self._log(
                        "SKIP",
                        f"Swing CSV row {row_num} in {filename} parse error: {exc}; skipping.",
                    )

        self._log(
            "CSV",
            f"Loaded {len(signals)} Swing signals from {filename} "
            f"({loaded_rows} rows, {skipped_rows} skipped).",
        )
        return signals


# === Cache restoration helpers ===

def restore_day_signal_from_cache(cached: dict) -> DaySignal:
    """Restore a DaySignal object from cached dict data."""
    trade_date = cached.get("trade_date")
    if isinstance(trade_date, str):
        trade_date = date.fromisoformat(trade_date)

    return DaySignal(
        strategy_id=cached["strategy_id"],
        symbol=cached["symbol"],
        trade_date=trade_date,
        direction=cached["direction"],
        entry_price=float(cached["entry_price"]),
        stop_price=float(cached["stop_price"]),
        shares=int(cached["shares"]),
        source_file=cached.get("source_file", ""),
        stop_distance=float(cached.get("stop_distance", 0.0)),
    )


def restore_swing_signal_from_cache(cached: dict) -> SwingSignal:
    """Restore a SwingSignal object from cached dict data."""
    return SwingSignal(
        strategy_id=cached["strategy_id"],
        symbol=cached["symbol"],
        direction=cached["direction"],
        entry_price=float(cached["entry_price"]),
        stop_price=float(cached["stop_price"]),
        shares=int(cached["shares"]),
        source_file=cached.get("source_file", ""),
        stop_distance=float(cached.get("stop_distance", 0.0)),
    )
