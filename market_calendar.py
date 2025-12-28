"""
market_calendar.py

Market calendar utilities for NYSE trading schedule awareness.

Uses pandas_market_calendars for comprehensive NYSE holiday and
early close detection. Provides helpers for:
- Trading day detection
- Week-ending day calculation (accounting for Friday holidays)
- Market close time (normal 13:00 PT vs early 10:00 PT)
- Next trading day calculation

All times are in Pacific Time (PT) to match bot conventions.
"""

from __future__ import annotations

import logging
from datetime import date, time, datetime, timedelta
from functools import lru_cache
from typing import Optional, Tuple
from zoneinfo import ZoneInfo

import pandas_market_calendars as mcal

PT = ZoneInfo("America/Los_Angeles")
ET = ZoneInfo("America/New_York")

# Cache the NYSE calendar instance
_nyse_calendar = None


def _get_nyse_calendar():
    """Get or create the NYSE calendar instance."""
    global _nyse_calendar
    if _nyse_calendar is None:
        _nyse_calendar = mcal.get_calendar("NYSE")
    return _nyse_calendar


@lru_cache(maxsize=128)
def is_trading_day(d: date) -> bool:
    """
    Return True if the given date is a valid NYSE trading day.

    Returns False for weekends and market holidays.
    """
    cal = _get_nyse_calendar()
    # Get schedule for just this day
    schedule = cal.schedule(start_date=d, end_date=d)
    return len(schedule) > 0


@lru_cache(maxsize=128)
def is_early_close(d: date) -> bool:
    """
    Return True if the given date is an early close day (1:00 PM ET).

    Early close days include:
    - Day before Independence Day (usually Jul 3)
    - Day after Thanksgiving (Black Friday)
    - Christmas Eve (Dec 24)
    - New Year's Eve (Dec 31, some years)
    """
    if not is_trading_day(d):
        return False

    cal = _get_nyse_calendar()
    schedule = cal.schedule(start_date=d, end_date=d)

    if len(schedule) == 0:
        return False

    # Get the market close time for this day
    close_time = schedule.iloc[0]["market_close"]
    # Early close is 1:00 PM ET (18:00 UTC typically)
    # Normal close is 4:00 PM ET (21:00 UTC typically)
    # Check if close hour is before 16:00 ET
    close_et = close_time.tz_convert(ET)
    return close_et.hour < 16


@lru_cache(maxsize=128)
def get_market_close_time_pt(d: date) -> time:
    """
    Return the market close time in PT for the given date.

    - Normal days: 13:00 PT (4:00 PM ET)
    - Early close days: 10:00 PT (1:00 PM ET)

    Raises ValueError if not a trading day.
    """
    if not is_trading_day(d):
        raise ValueError(f"{d} is not a trading day")

    if is_early_close(d):
        return time(10, 0)  # 1:00 PM ET = 10:00 AM PT
    else:
        return time(13, 0)  # 4:00 PM ET = 1:00 PM PT


def get_day_stop_time_pt(d: date) -> time:
    """
    Return the Day stop GTD time in PT for the given date.

    5 minutes before market close:
    - Normal days: 12:55 PT
    - Early close days: 9:55 PT
    """
    close = get_market_close_time_pt(d)
    # Subtract 5 minutes
    close_dt = datetime.combine(d, close)
    stop_dt = close_dt - timedelta(minutes=5)
    return stop_dt.time()


def get_day_exit_time_pt(d: date) -> time:
    """
    Return the Day timed exit GAT time in PT for the given date.

    2 minutes before market close:
    - Normal days: 12:58 PT
    - Early close days: 9:58 PT
    """
    close = get_market_close_time_pt(d)
    # Subtract 2 minutes
    close_dt = datetime.combine(d, close)
    exit_dt = close_dt - timedelta(minutes=2)
    return exit_dt.time()


def get_weekly_flatten_time_pt(d: date) -> time:
    """
    Return the weekly flatten time in PT for the given date.

    5 minutes before market close on week-ending day:
    - Normal days: 12:55 PT
    - Early close days: 9:55 PT
    """
    return get_day_stop_time_pt(d)


@lru_cache(maxsize=128)
def get_next_trading_day(d: date) -> date:
    """
    Return the next trading day after the given date.

    Skips weekends and holidays.
    """
    cal = _get_nyse_calendar()
    next_d = d + timedelta(days=1)

    # Look ahead up to 10 days (handles long weekends)
    for _ in range(10):
        if is_trading_day(next_d):
            return next_d
        next_d += timedelta(days=1)

    # Fallback (shouldn't happen with normal market schedule)
    raise ValueError(f"No trading day found within 10 days of {d}")


@lru_cache(maxsize=128)
def get_previous_trading_day(d: date) -> date:
    """
    Return the previous trading day before the given date.

    Skips weekends and holidays.
    """
    prev_d = d - timedelta(days=1)

    # Look back up to 10 days
    for _ in range(10):
        if is_trading_day(prev_d):
            return prev_d
        prev_d -= timedelta(days=1)

    raise ValueError(f"No trading day found within 10 days before {d}")


@lru_cache(maxsize=128)
def get_week_ending_day(d: date) -> date:
    """
    Return the week-ending trading day for the week containing date d.

    Normally Friday, but could be Thursday (or earlier) if Friday is a holiday.

    Examples:
    - Normal week: Returns Friday
    - Good Friday week: Returns Thursday
    - Thanksgiving week: Returns Friday (early close)
    """
    # Find the Friday of this week
    days_until_friday = (4 - d.weekday()) % 7
    if d.weekday() > 4:  # Saturday or Sunday
        days_until_friday = 4 - d.weekday() + 7

    friday = d + timedelta(days=days_until_friday)
    if d.weekday() == 4:  # d is Friday
        friday = d
    elif d.weekday() > 4:  # Weekend - use previous Friday's week
        friday = d - timedelta(days=(d.weekday() - 4))

    # Check if Friday is a trading day
    if is_trading_day(friday):
        return friday

    # Friday is a holiday, walk back to find the last trading day of the week
    check_date = friday - timedelta(days=1)
    week_start = friday - timedelta(days=4)  # Monday of this week

    while check_date >= week_start:
        if is_trading_day(check_date):
            return check_date
        check_date -= timedelta(days=1)

    # Edge case: entire week is holidays (very rare)
    raise ValueError(f"No trading day found in week containing {d}")


def get_week_ending_datetime_pt(d: date, minutes_before_close: int = 5) -> datetime:
    """
    Return the week-ending datetime in PT.

    Args:
        d: Any date in the target week
        minutes_before_close: Minutes before market close (default 5)

    Returns:
        datetime of (week_ending_day, close_time - minutes_before_close) in PT
    """
    end_day = get_week_ending_day(d)
    close = get_market_close_time_pt(end_day)
    close_dt = datetime.combine(end_day, close, tzinfo=PT)
    return close_dt - timedelta(minutes=minutes_before_close)


def get_swing_exit_day(trade_date: date) -> date:
    """
    Return the day when swing positions should exit for a trade entered on trade_date.

    This is the week-ending day of the CURRENT week (not next week).
    For Monday-Thursday entries, exits on Friday (or Thursday if Friday holiday).
    For Friday entries, exits same day at close.
    """
    return get_week_ending_day(trade_date)


def get_swing_stop_gtd_datetime_pt(trade_date: date) -> datetime:
    """
    Return the GTD datetime for swing stop orders in PT.

    Set to 5 minutes before close on the week-ending day.
    """
    return get_week_ending_datetime_pt(trade_date, minutes_before_close=5)


def get_swing_timed_exit_gat_datetime_pt(trade_date: date) -> datetime:
    """
    Return the GAT datetime for swing timed exit orders in PT.

    Set to market open (6:30 PT) on the week-ending day.
    This allows the timed exit to activate at open on the last day.
    """
    end_day = get_week_ending_day(trade_date)
    return datetime.combine(end_day, time(6, 30), tzinfo=PT)


def is_week_ending_day(d: date) -> bool:
    """
    Return True if d is the week-ending trading day.
    """
    if not is_trading_day(d):
        return False
    return get_week_ending_day(d) == d


def can_enter_weekly_position(d: date, current_time: time) -> Tuple[bool, Optional[str]]:
    """
    Check if a new weekly position can be entered at the given date/time.

    Returns (True, None) if allowed.
    Returns (False, reason) if blocked.

    Blocked if:
    - It's the week-ending day and current_time >= (close - 5 minutes)
    """
    if not is_trading_day(d):
        return False, "Not a trading day"

    if is_week_ending_day(d):
        cutoff = get_day_stop_time_pt(d)  # 5 min before close
        if current_time >= cutoff:
            return False, f"Past weekly cutoff ({cutoff}) on week-ending day"

    return True, None


def format_datetime_for_ib(dt: datetime) -> str:
    """
    Format a datetime for IBKR GTD/GAT fields.

    Returns: 'YYYYMMDD HH:MM:SS'
    """
    return dt.strftime("%Y%m%d %H:%M:%S")


# === Logging/Debug Helpers ===

def log_market_schedule(logger: logging.Logger, start_date: date, days: int = 5) -> None:
    """
    Log market schedule for debugging.
    """
    logger.info("[CALENDAR] Market schedule for next %d days from %s:", days, start_date)

    d = start_date
    for _ in range(days):
        if is_trading_day(d):
            close = get_market_close_time_pt(d)
            early = " (EARLY CLOSE)" if is_early_close(d) else ""
            week_end = " [WEEK END]" if is_week_ending_day(d) else ""
            logger.info(
                "[CALENDAR]   %s (%s): Open, close %s PT%s%s",
                d,
                d.strftime("%a"),
                close,
                early,
                week_end,
            )
        else:
            logger.info(
                "[CALENDAR]   %s (%s): CLOSED (holiday/weekend)",
                d,
                d.strftime("%a"),
            )
        d += timedelta(days=1)
