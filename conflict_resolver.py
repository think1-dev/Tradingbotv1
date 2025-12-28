"""
conflict_resolver.py

Handles position conflict rules for Day vs Day and Weekly vs Day trades.

This module determines what actions to take when a new signal wants to enter
a position that may conflict with existing positions. It does NOT interact
with IBKR directly - it only returns decisions about what actions to take.

Position data is sourced from FillTracker (single source of truth).

Conflict Rules (Complete Matrix):
- Same-side positions: Always allow co-existence
- Opposite-side positions: Flatten existing before new entry

Examples:
1. DAY LONG exists → DAY LONG signal: Allow both
2. DAY LONG exists → DAY SHORT signal: Flatten DAY LONG, allow SHORT
3. SWING LONG exists → DAY SHORT signal: Flatten SWING LONG, allow SHORT
4. DAY LONG exists → SWING LONG signal: Allow both
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional

if TYPE_CHECKING:
    from fill_tracker import FillTracker, FilledPosition


@dataclass
class FlattenInstruction:
    """Instruction to flatten (close) a specific position."""
    symbol: str
    kind: str  # "DAY" or "SWING"
    side: str  # "LONG" or "SHORT"
    qty: int
    strategy_id: str
    parent_order_id: int
    stop_order_id: Optional[int] = None
    timed_order_id: Optional[int] = None

    def __str__(self) -> str:
        return f"Flatten {self.symbol} {self.kind} {self.side} qty={self.qty} strategy={self.strategy_id}"


@dataclass
class ConflictDecision:
    """
    Decision returned by ConflictResolver.decide().

    Attributes:
        positions_to_flatten: List of positions that must be closed before entry
        allow_entry: Whether the new entry is allowed (from conflict perspective)
        reason: Human-readable explanation for logging
    """
    positions_to_flatten: List[FlattenInstruction] = field(default_factory=list)
    allow_entry: bool = True
    reason: str = "No conflict"

    @property
    def requires_flatten(self) -> bool:
        """Returns True if positions need to be flattened before entry."""
        return len(self.positions_to_flatten) > 0

    def __str__(self) -> str:
        if not self.allow_entry:
            return f"BLOCKED: {self.reason}"
        if self.requires_flatten:
            flatten_strs = [str(f) for f in self.positions_to_flatten]
            return f"FLATTEN FIRST: {', '.join(flatten_strs)} | Then: {self.reason}"
        return f"ALLOWED: {self.reason}"


class ConflictResolver:
    """
    Resolves position conflicts between Day and Swing trades.

    Queries FillTracker for current filled positions (single source of truth)
    and determines what actions to take when a new signal wants to enter.

    Does NOT interact with IBKR - only returns decisions.

    Usage:
        resolver = ConflictResolver(logger, fill_tracker)

        # Before placing a new order, check for conflicts
        decision = resolver.decide(signal)

        if decision.requires_flatten:
            for instr in decision.positions_to_flatten:
                executor.flatten_position(instr)

        if decision.allow_entry:
            executor.place_bracket(signal)
    """

    def __init__(self, logger: logging.Logger, fill_tracker: "FillTracker") -> None:
        self.logger = logger
        self.fill_tracker = fill_tracker

    def decide(self, signal) -> ConflictDecision:
        """
        Decide what to do when a new signal wants to enter.

        Args:
            signal: A DaySignal or SwingSignal with attributes:
                - symbol: str
                - direction: str ("LONG" or "SHORT")
                - strategy_id: str
                - shares: int

        Returns:
            ConflictDecision with flatten instructions and allow_entry flag
        """
        symbol = signal.symbol.upper()
        new_side = getattr(signal, "direction", "LONG").upper()
        new_kind = self._infer_kind(signal)

        # Query FillTracker for existing positions on this symbol
        existing = self.fill_tracker.get_filled_positions_by_symbol(symbol)

        if not existing:
            return ConflictDecision(
                allow_entry=True,
                reason=f"No existing positions for {symbol}"
            )

        # Separate by side
        same_side = [p for p in existing if p.side == new_side]
        opposite_side = [p for p in existing if p.side != new_side]

        # Same-side positions: allow co-existence
        if not opposite_side:
            return ConflictDecision(
                allow_entry=True,
                reason=f"Same-side {new_side} - allowing co-existence with {len(same_side)} existing position(s)"
            )

        # Opposite-side exists: need to flatten
        flatten_list = []
        for pos in opposite_side:
            flatten_list.append(FlattenInstruction(
                symbol=pos.symbol,
                kind=pos.kind,
                side=pos.side,
                qty=pos.qty,
                strategy_id=pos.strategy_id,
                parent_order_id=pos.parent_order_id,
                stop_order_id=pos.stop_order_id,
                timed_order_id=pos.timed_order_id,
            ))

        reason = self._build_reason(new_side, new_kind, opposite_side)

        self.logger.info(
            "[CONFLICT] %s %s %s signal requires flattening %d opposite-side position(s)",
            symbol, new_kind, new_side, len(flatten_list)
        )

        return ConflictDecision(
            positions_to_flatten=flatten_list,
            allow_entry=True,
            reason=reason,
        )

    def _infer_kind(self, signal) -> str:
        """Infer whether signal is DAY or SWING based on class name or attribute."""
        if hasattr(signal, "kind"):
            return signal.kind.upper()

        class_name = signal.__class__.__name__.lower()
        if "swing" in class_name:
            return "SWING"
        return "DAY"

    def _build_reason(
        self,
        new_side: str,
        new_kind: str,
        opposite_positions: List["FilledPosition"],
    ) -> str:
        """Build a human-readable reason string."""
        if not opposite_positions:
            return "No conflicts"

        opposite_desc = []
        for p in opposite_positions:
            opposite_desc.append(f"{p.kind} {p.side}")

        return (
            f"Flatten {len(opposite_positions)} opposite-side position(s) "
            f"[{', '.join(opposite_desc)}] before entering {new_kind} {new_side}"
        )

    # ---------- Utility methods ---------- #

    def get_positions_for_symbol(self, symbol: str) -> List["FilledPosition"]:
        """Get all filled positions for a symbol (delegates to FillTracker)."""
        return self.fill_tracker.get_filled_positions_by_symbol(symbol)

    def has_position(self, symbol: str, kind: Optional[str] = None, side: Optional[str] = None) -> bool:
        """Check if any position exists for a symbol."""
        return self.fill_tracker.has_position(symbol, kind, side)

    def get_summary(self) -> dict:
        """Get a summary of current position state."""
        return self.fill_tracker.get_position_summary()
