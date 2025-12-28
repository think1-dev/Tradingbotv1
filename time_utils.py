"""
time_utils.py

Phase 2+ : Time and calendar utilities for the trading bot.

All logic is expressed in America/Los_Angeles (PT), to match the spec:
- RTH: 09:30–16:00 ET  =>  06:30–13:00 PT.
- Day stop ~12:55 PT (or 9:55 PT on early close days)
- Day timed exit ~12:58 PT (or 9:58 PT on early close days)

Now integrates with market_calendar.py for holiday and early close awareness.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time, timedelta, date
from zoneinfo import ZoneInfo

PT = ZoneInfo("America/Los_Angeles")

# Import market calendar functions (lazy import to avoid circular deps)
_market_calendar = None


def _get_market_calendar():
    """Lazy import of market_calendar module."""
    global _market_calendar
    if _market_calendar is None:
        import market_calendar as mc
        _market_calendar = mc
    return _market_calendar


def now_pt() -> datetime:
    """
    Return timezone-aware current time in PT.
    """
    return datetime.now(PT)


def today_pt_date_str() -> str:
    """
    Return today's PT date as 'YYYY-MM-DD'.
    """
    return now_pt().date().isoformat()


@dataclass(frozen=True)
class RthWindow:
    start: time  # PT
    end: time    # PT


def get_rth_window_pt(d: date | None = None) -> RthWindow:
    """
    Regular US stock RTH in PT for the given date.

    Normal days: 06:30–13:00 PT (09:30–16:00 ET)
    Early close days: 06:30–10:00 PT (09:30–13:00 ET)

    If d is None, returns the normal window (for backwards compatibility).
    """
    if d is None:
        return RthWindow(start=time(6, 30), end=time(13, 0))

    mc = _get_market_calendar()
    try:
        close_time = mc.get_market_close_time_pt(d)
        return RthWindow(start=time(6, 30), end=close_time)
    except ValueError:
        # Not a trading day - return normal window
        return RthWindow(start=time(6, 30), end=time(13, 0))


def is_rth(dt: datetime | None = None) -> bool:
    """
    Return True if dt (or now_pt()) is within US regular trading hours
    in PT **on a valid trading day**.

    Now incorporates:
    - Weekend check (Saturday/Sunday always closed)
    - Holiday check via market_calendar
    - Early close awareness (10:00 PT on early close days)
    """
    if dt is None:
        dt = now_pt()

    if dt.tzinfo is None:
        # Assume naive datetimes are PT; attach PT tz
        dt = dt.replace(tzinfo=PT)

    # Weekend guard: Saturday=5, Sunday=6 => never RTH
    if dt.weekday() >= 5:
        return False

    # Check if it's a trading day (not a holiday)
    mc = _get_market_calendar()
    if not mc.is_trading_day(dt.date()):
        return False

    # Get the appropriate RTH window for this day
    rth = get_rth_window_pt(dt.date())
    t = dt.timetz()
    return rth.start <= t <= rth.end


def get_day_stop_time_pt(date_pt: datetime | None = None) -> datetime:
    """
    Return the PT datetime for the Day-stop GTD time for a given PT date.

    5 minutes before market close:
    - Normal days: 12:55 PT
    - Early close days: 9:55 PT
    """
    if date_pt is None:
        date_pt = now_pt()

    d = date_pt.date() if isinstance(date_pt, datetime) else date_pt

    mc = _get_market_calendar()
    try:
        stop_time = mc.get_day_stop_time_pt(d)
        return datetime(d.year, d.month, d.day, stop_time.hour, stop_time.minute, tzinfo=PT)
    except ValueError:
        # Not a trading day - use default
        return datetime(d.year, d.month, d.day, 12, 55, tzinfo=PT)


def get_day_exit_time_pt(date_pt: datetime | None = None) -> datetime:
    """
    Return the PT datetime for the Day timed-exit GAT time for a given PT date.

    2 minutes before market close:
    - Normal days: 12:58 PT
    - Early close days: 9:58 PT
    """
    if date_pt is None:
        date_pt = now_pt()

    d = date_pt.date() if isinstance(date_pt, datetime) else date_pt

    mc = _get_market_calendar()
    try:
        exit_time = mc.get_day_exit_time_pt(d)
        return datetime(d.year, d.month, d.day, exit_time.hour, exit_time.minute, tzinfo=PT)
    except ValueError:
        # Not a trading day - use default
        return datetime(d.year, d.month, d.day, 12, 58, tzinfo=PT)


def to_ib_time_string(dt: datetime) -> str:
    """
    Convert a timezone-aware datetime to an IBKR time string:
    'YYYYMMDD HH:MM:SS' in the dt's local timezone.

    We will use this for goodTillDate / goodAfterTime later.
    """
    if dt.tzinfo is None:
        # Assume PT if missing
        dt = dt.replace(tzinfo=PT)

    return dt.strftime("%Y%m%d %H:%M:%S")


def week_monday_for(d: date) -> date:
    """
    Given any date d, return the Monday (PT week start) for that week.
    Monday = 0, Sunday = 6.
    """
    return d - timedelta(days=d.weekday())


# === Market Calendar Helper Exports ===
# These provide convenient access to market_calendar functions from time_utils


def is_trading_day(d: date) -> bool:
    """
    Return True if d is a valid NYSE trading day (not weekend/holiday).
    """
    mc = _get_market_calendar()
    return mc.is_trading_day(d)


def is_early_close(d: date) -> bool:
    """
    Return True if d is an early close day (1:00 PM ET / 10:00 AM PT).
    """
    mc = _get_market_calendar()
    return mc.is_early_close(d)


def get_week_ending_day(d: date) -> date:
    """
    Return the week-ending trading day for the week containing d.

    Normally Friday, but Thursday (or earlier) if Friday is a holiday.
    """
    mc = _get_market_calendar()
    return mc.get_week_ending_day(d)


def is_week_ending_day(d: date) -> bool:
    """
    Return True if d is the last trading day of its week.
    """
    mc = _get_market_calendar()
    return mc.is_week_ending_day(d)


def get_next_trading_day(d: date) -> date:
    """
    Return the next trading day after d.
    """
    mc = _get_market_calendar()
    return mc.get_next_trading_day(d)


def get_market_close_time_pt(d: date) -> time:
    """
    Return market close time in PT for date d.

    Normal: 13:00 PT, Early close: 10:00 PT
    """
    mc = _get_market_calendar()
    return mc.get_market_close_time_pt(d)
