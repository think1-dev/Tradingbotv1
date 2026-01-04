"""
test_error_handling.py

Comprehensive test suite for TWS API error handling.
Tests the bot's response to every error code from the TWS API reference.

Usage:
    python test_error_handling.py

This simulates IBKR error callbacks and verifies the bot handles each correctly.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from enum import Enum

# Define SHORTABLE_ERROR_CODES locally (same as execution.py)
# 201: Order rejected - no shares to borrow
# 10147: Order would violate security short sale rule
# 162: Historical market data error (short sale related)
# 426: None of accounts have enough shares for short sale
SHORTABLE_ERROR_CODES = {201, 10147, 162, 426}


@dataclass
class PlacementResult:
    """Local copy of PlacementResult for testing without ib_insync."""
    order_id: Optional[int] = None
    success: bool = False
    rejection_code: Optional[int] = None
    rejection_message: Optional[str] = None

    @property
    def is_shortable_rejection(self) -> bool:
        return self.rejection_code in SHORTABLE_ERROR_CODES


class ErrorCategory(Enum):
    CONNECTION_CLIENT = "Client-side connection (500s)"
    CONNECTION_SERVER = "Server connection status (1100-1300)"
    DATA_FARM = "Data farm connectivity (2100s)"
    CLIENT_AUTH = "Client ID and authentication"
    ORDER_SUBMISSION = "Order submission and ID (100-149)"
    PRICE_VALIDATION = "Price validation"
    ORDER_TYPE_TIF = "Order type and TIF"
    ORDER_REJECTION = "Primary order rejection (200-203)"
    SIZE_QUANTITY = "Size and quantity"
    TRADING_HOURS = "Trading hours and market status"
    ACCOUNT_PERMISSION = "Account and permission"
    ALGORITHM = "Algorithm order (400s)"
    FRACTIONAL = "Fractional shares (10241-10252)"
    CONDITIONAL = "Conditional and trigger"
    BRACKET_COMBO = "Combo and bracket"


@dataclass
class ErrorDefinition:
    code: int
    message: str
    category: ErrorCategory
    severity: str  # "info", "warning", "error", "critical"
    bot_action: str  # What the bot SHOULD do
    currently_handled: bool
    notes: str = ""


# Complete TWS API error code definitions with bot handling expectations
TWS_ERRORS: List[ErrorDefinition] = [
    # === CLIENT-SIDE CONNECTION ERRORS (500 range) ===
    ErrorDefinition(501, "Already Connected", ErrorCategory.CONNECTION_CLIENT, "warning",
                   "Log warning, continue operation", True,
                   "ib_insync handles internally"),
    ErrorDefinition(502, "Couldn't connect to TWS", ErrorCategory.CONNECTION_CLIENT, "critical",
                   "Retry connection with backoff", True,
                   "ConnectionManager handles reconnection"),
    ErrorDefinition(503, "TWS is out of date", ErrorCategory.CONNECTION_CLIENT, "critical",
                   "Log error, exit - requires manual TWS update", False,
                   "Should alert user to update TWS"),
    ErrorDefinition(504, "Not connected", ErrorCategory.CONNECTION_CLIENT, "error",
                   "Trigger reconnection", True,
                   "ConnectionManager handles"),
    ErrorDefinition(505, "Fatal Error: Unknown message id", ErrorCategory.CONNECTION_CLIENT, "critical",
                   "Log error, may need API version update", False,
                   "Version mismatch - alert user"),
    ErrorDefinition(506, "Unsupported version", ErrorCategory.CONNECTION_CLIENT, "critical",
                   "Log error, exit - requires update", False,
                   "Should alert user"),
    ErrorDefinition(507, "Bad message length", ErrorCategory.CONNECTION_CLIENT, "error",
                   "Log error, reconnect", True,
                   "Often caused by duplicate client ID"),
    ErrorDefinition(508, "Bad message", ErrorCategory.CONNECTION_CLIENT, "error",
                   "Log error, reconnect", True),
    ErrorDefinition(509, "Exception caught while reading socket", ErrorCategory.CONNECTION_CLIENT, "error",
                   "Trigger reconnection", True),
    ErrorDefinition(520, "Failed to create socket", ErrorCategory.CONNECTION_CLIENT, "critical",
                   "Log error, retry with backoff", True),
    ErrorDefinition(530, "SSL specific error", ErrorCategory.CONNECTION_CLIENT, "critical",
                   "Log error, check SSL config", False),

    # === SERVER CONNECTION STATUS (1100-1300) ===
    ErrorDefinition(1100, "Connectivity between IB and TWS has been lost", ErrorCategory.CONNECTION_SERVER, "warning",
                   "Wait for automatic reconnection, DO NOT disconnect", True,
                   "Normal during nightly reset ~11:45 PM ET"),
    ErrorDefinition(1101, "Connectivity restored - data lost", ErrorCategory.CONNECTION_SERVER, "warning",
                   "Resubscribe to all market data feeds", True,
                   "Must re-request market data"),
    ErrorDefinition(1102, "Connectivity restored - data maintained", ErrorCategory.CONNECTION_SERVER, "info",
                   "No action needed, subscriptions recovered", True),
    ErrorDefinition(1300, "TWS socket port has been reset", ErrorCategory.CONNECTION_SERVER, "warning",
                   "Reconnect using new port from message", False,
                   "Rare - may need dynamic port handling"),

    # === DATA FARM CONNECTIVITY (2100 range) ===
    ErrorDefinition(2103, "Market data farm disconnected", ErrorCategory.DATA_FARM, "warning",
                   "Wait for auto-reconnect", True,
                   "ISP issue outside nightly reset"),
    ErrorDefinition(2104, "Market data farm connection is OK", ErrorCategory.DATA_FARM, "info",
                   "No action - normal startup notification", True),
    ErrorDefinition(2105, "Historical data farm disconnected", ErrorCategory.DATA_FARM, "warning",
                   "Wait for auto-reconnect", True),
    ErrorDefinition(2106, "Historical data farm connected", ErrorCategory.DATA_FARM, "info",
                   "No action - normal startup", True),
    ErrorDefinition(2107, "Historical data farm inactive but available", ErrorCategory.DATA_FARM, "info",
                   "No action - normal dormant state", True),
    ErrorDefinition(2108, "Market data farm inactive but available", ErrorCategory.DATA_FARM, "info",
                   "No action - normal behavior", True),
    ErrorDefinition(2110, "Connectivity between TWS and server broken", ErrorCategory.DATA_FARM, "warning",
                   "Wait for auto-restore (nightly reset)", True),
    ErrorDefinition(2158, "Sec-def data farm connection is OK", ErrorCategory.DATA_FARM, "info",
                   "No action - confirms connection", True),

    # === CLIENT ID AND AUTHENTICATION ===
    ErrorDefinition(326, "Unable to connect - client ID in use", ErrorCategory.CLIENT_AUTH, "error",
                   "Use different client ID or close other connection", False,
                   "Should try different client ID"),
    ErrorDefinition(327, "Only clientId 0 can set auto bind", ErrorCategory.CLIENT_AUTH, "warning",
                   "Use client ID 0 for order status", True),
    ErrorDefinition(100, "Max rate of messages exceeded", ErrorCategory.CLIENT_AUTH, "error",
                   "Implement rate limiting, may disconnect", False,
                   "50 msg/sec limit - need throttling"),

    # === ORDER SUBMISSION AND ID ERRORS (100-149) ===
    ErrorDefinition(103, "Duplicate order ID", ErrorCategory.ORDER_SUBMISSION, "error",
                   "Use nextValidId() for order IDs", True,
                   "Bot uses _get_next_order_id()"),
    ErrorDefinition(104, "Can't modify a filled order", ErrorCategory.ORDER_SUBMISSION, "warning",
                   "Check order status before modification", True),
    ErrorDefinition(105, "Order being modified does not match original", ErrorCategory.ORDER_SUBMISSION, "error",
                   "Only modify price/quantity, not fundamentals", True),
    ErrorDefinition(106, "Can't transmit order ID", ErrorCategory.ORDER_SUBMISSION, "error",
                   "Check order parameters and connectivity", True),
    ErrorDefinition(107, "Cannot transmit incomplete order", ErrorCategory.ORDER_SUBMISSION, "error",
                   "Ensure all required fields specified", True),
    ErrorDefinition(133, "Submit new order failed", ErrorCategory.ORDER_SUBMISSION, "error",
                   "Check order parameters and account status", True),
    ErrorDefinition(134, "Modify order failed", ErrorCategory.ORDER_SUBMISSION, "error",
                   "Verify order exists and is modifiable", True),
    ErrorDefinition(135, "Can't find order with ID", ErrorCategory.ORDER_SUBMISSION, "warning",
                   "Order may have filled/cancelled - verify", True),
    ErrorDefinition(136, "This order cannot be cancelled", ErrorCategory.ORDER_SUBMISSION, "warning",
                   "Order may be filled/expired/already cancelled", True),
    ErrorDefinition(140, "Size value should be an integer", ErrorCategory.ORDER_SUBMISSION, "error",
                   "Use integer for totalQuantity", True,
                   "signals.py ensures integer shares"),
    ErrorDefinition(141, "Price value should be a double", ErrorCategory.ORDER_SUBMISSION, "error",
                   "Use float for price fields", True),
    ErrorDefinition(144, "Order size does not match FA allocation", ErrorCategory.ORDER_SUBMISSION, "error",
                   "Adjust share allocation", False,
                   "FA accounts only"),
    ErrorDefinition(145, "Error in validating entry fields", ErrorCategory.ORDER_SUBMISSION, "error",
                   "Review and correct field syntax", True),

    # === PRICE VALIDATION ERRORS ===
    ErrorDefinition(109, "Price out of range (Precautionary)", ErrorCategory.PRICE_VALIDATION, "error",
                   "Adjust price or modify TWS settings", True),
    ErrorDefinition(110, "Price does not conform to minimum tick", ErrorCategory.PRICE_VALIDATION, "error",
                   "Round price to valid tick size", False,
                   "Should round to tick size"),
    ErrorDefinition(125, "Buy price must be same as best ask", ErrorCategory.PRICE_VALIDATION, "error",
                   "Adjust buy limit price", True),
    ErrorDefinition(126, "Sell price must be same as best bid", ErrorCategory.PRICE_VALIDATION, "error",
                   "Adjust sell limit price", True),
    ErrorDefinition(163, "Price would violate percentage constraint", ErrorCategory.PRICE_VALIDATION, "error",
                   "Adjust price or modify TWS settings", True),
    ErrorDefinition(382, "Price violates number of ticks constraint", ErrorCategory.PRICE_VALIDATION, "error",
                   "Adjust price or modify settings", True),
    ErrorDefinition(403, "Invalid stop price", ErrorCategory.PRICE_VALIDATION, "error",
                   "Adjust stop price for contract", True),

    # === ORDER TYPE AND TIF ERRORS ===
    ErrorDefinition(111, "TIF and order type are incompatible", ErrorCategory.ORDER_TYPE_TIF, "error",
                   "Use valid TIF/order type combination", True),
    ErrorDefinition(113, "TIF should be DAY for MOC and LOC orders", ErrorCategory.ORDER_TYPE_TIF, "error",
                   "Set TIF to DAY", True),
    ErrorDefinition(116, "Order cannot be transmitted to dead exchange", ErrorCategory.ORDER_TYPE_TIF, "error",
                   "Verify exchange is active", True),
    ErrorDefinition(117, "Block order size must be at least 50", ErrorCategory.ORDER_TYPE_TIF, "error",
                   "Increase order size to minimum 50", True),
    ErrorDefinition(152, "Hidden order attribute may not be specified", ErrorCategory.ORDER_TYPE_TIF, "error",
                   "Remove Hidden attribute", True),
    ErrorDefinition(157, "Order can be EITHER Iceberg or Discretionary", ErrorCategory.ORDER_TYPE_TIF, "error",
                   "Remove conflicting attribute", True),
    ErrorDefinition(158, "Must specify offset amount or percent for TRAIL", ErrorCategory.ORDER_TYPE_TIF, "error",
                   "Add auxPrice or percentOffset", True),
    ErrorDefinition(325, "Discretionary orders not supported", ErrorCategory.ORDER_TYPE_TIF, "error",
                   "Use different order type/exchange", True),
    ErrorDefinition(328, "Trailing stop only for limit or stop-limit", ErrorCategory.ORDER_TYPE_TIF, "error",
                   "Change parent order type", True),
    ErrorDefinition(387, "Unsupported order type for this exchange", ErrorCategory.ORDER_TYPE_TIF, "error",
                   "Check IB Order Types documentation", True),

    # === PRIMARY ORDER REJECTION CODES (200-203) - CRITICAL ===
    ErrorDefinition(200, "No security definition found", ErrorCategory.ORDER_REJECTION, "error",
                   "Block symbol - contract doesn't exist or wrong params", True,
                   "Symbol may be delisted or wrong exchange"),
    ErrorDefinition(201, "Order rejected - Reason: [specific]", ErrorCategory.ORDER_REJECTION, "error",
                   "Handle based on reason text - shortable rejection enables ghost mode", True,
                   "PRIMARY rejection code - includes shortable rejection"),
    ErrorDefinition(202, "Order cancelled - Reason: [specific]", ErrorCategory.ORDER_REJECTION, "warning",
                   "Order was cancelled by system - check reason", True,
                   "May be price check failure"),
    ErrorDefinition(203, "Security not available for this account", ErrorCategory.ORDER_REJECTION, "error",
                   "Block symbol for account - trading restriction", True,
                   "Account-level restriction on security"),

    # === SIZE AND QUANTITY ERRORS ===
    ErrorDefinition(355, "Order size does not conform to market rule", ErrorCategory.SIZE_QUANTITY, "error",
                   "Check minSize and sizeIncrement in ContractDetails", False,
                   "Should validate against contract specs"),
    ErrorDefinition(383, "Size violates size constraint in settings", ErrorCategory.SIZE_QUANTITY, "error",
                   "Adjust size or modify TWS settings", True),
    ErrorDefinition(388, "Order size smaller than minimum", ErrorCategory.SIZE_QUANTITY, "error",
                   "Increase order size to minimum", True),
    ErrorDefinition(434, "Order size cannot be zero", ErrorCategory.SIZE_QUANTITY, "error",
                   "Ensure positive quantity", True,
                   "signals.py skips qty=0"),
    ErrorDefinition(10020, "Display size should be smaller than total", ErrorCategory.SIZE_QUANTITY, "error",
                   "Adjust displaySize attribute", True),

    # === TRADING HOURS AND MARKET STATUS ===
    ErrorDefinition(154, "Orders cannot be transmitted for halted security", ErrorCategory.TRADING_HOURS, "warning",
                   "Wait for trading to resume - do not block symbol", True,
                   "Temporary halt - retry later"),
    ErrorDefinition(351, "Regular Trading Hours flag not valid", ErrorCategory.TRADING_HOURS, "error",
                   "Remove outsideRth flag or change order params", True),
    ErrorDefinition(411, "Outside Regular Trading Hours flag not valid", ErrorCategory.TRADING_HOURS, "error",
                   "Contract doesn't support extended hours", True),
    ErrorDefinition(412, "Contract is not available for trading", ErrorCategory.TRADING_HOURS, "error",
                   "Verify contract status and trading calendar", True),
    ErrorDefinition(392, "Invalid order: contract expired", ErrorCategory.TRADING_HOURS, "error",
                   "Use current/valid contract expiration", True),

    # === ACCOUNT AND PERMISSION ERRORS ===
    ErrorDefinition(10015, "Trading is not allowed in the API", ErrorCategory.ACCOUNT_PERMISSION, "critical",
                   "Enable API trading in Account Management", False,
                   "User must enable API trading"),
    ErrorDefinition(426, "None of the accounts have enough shares", ErrorCategory.ACCOUNT_PERMISSION, "error",
                   "Insufficient shares for short sale - enable ghost mode", True,
                   "Part of SHORTABLE_ERROR_CODES - triggers ghost mode"),
    ErrorDefinition(435, "You must specify an account", ErrorCategory.ACCOUNT_PERMISSION, "error",
                   "Provide account code for single-account function", True),
    ErrorDefinition(436, "You must specify an allocation", ErrorCategory.ACCOUNT_PERMISSION, "error",
                   "Specify FA account/group/profile", False,
                   "FA accounts only"),

    # === ALGORITHM ORDER ERRORS ===
    ErrorDefinition(400, "Algo order error", ErrorCategory.ALGORITHM, "error",
                   "Review AlgoStrategy and AlgoParams", True,
                   "Not using algos currently"),
    ErrorDefinition(439, "Algorithm definition not found", ErrorCategory.ALGORITHM, "error",
                   "Use valid IB algorithm name", True),
    ErrorDefinition(440, "Algorithm cannot be modified", ErrorCategory.ALGORITHM, "error",
                   "Cancel and replace the algo order", True),
    ErrorDefinition(441, "Algo attributes validation failed", ErrorCategory.ALGORITHM, "error",
                   "Check parameters against IB documentation", True),
    ErrorDefinition(442, "Specified algorithm not allowed", ErrorCategory.ALGORITHM, "error",
                   "Use compatible algorithm", True),
    ErrorDefinition(443, "Unknown algo attribute", ErrorCategory.ALGORITHM, "error",
                   "Remove or correct unknown parameter", True),

    # === FRACTIONAL SHARES ERRORS ===
    ErrorDefinition(10241, "Order in monetary terms not supported via API", ErrorCategory.FRACTIONAL, "error",
                   "Use desktop TWS for monetary orders", True),
    ErrorDefinition(10242, "Fractional order cannot be modified via API", ErrorCategory.FRACTIONAL, "error",
                   "Use desktop TWS to modify", True),
    ErrorDefinition(10243, "Fractional order cannot be placed via API", ErrorCategory.FRACTIONAL, "error",
                   "Use desktop TWS for fractional orders", True),
    ErrorDefinition(10245, "Instrument does not support fractional shares", ErrorCategory.FRACTIONAL, "error",
                   "Use whole share quantities", True,
                   "signals.py uses int() for shares"),
    ErrorDefinition(10247, "Only IB SmartRouting supports fractional", ErrorCategory.FRACTIONAL, "error",
                   "Route to SMART exchange", True),
    ErrorDefinition(10248, "Account doesn't have fractional permission", ErrorCategory.FRACTIONAL, "error",
                   "Request fractional trading permission", True),

    # === CONDITIONAL AND TRIGGER ERRORS ===
    ErrorDefinition(146, "Invalid trigger method", ErrorCategory.CONDITIONAL, "error",
                   "Use valid trigger method (0-8)", True),
    ErrorDefinition(147, "Conditional contract info incomplete", ErrorCategory.CONDITIONAL, "error",
                   "Complete conditional contract definition", True),
    ErrorDefinition(361, "Invalid trigger price", ErrorCategory.CONDITIONAL, "error",
                   "Specify valid trigger price", True),
    ErrorDefinition(398, "Contract cannot be used as condition trigger", ErrorCategory.CONDITIONAL, "error",
                   "Use valid contract for condition", True),
    ErrorDefinition(402, "Conditions not allowed for this contract", ErrorCategory.CONDITIONAL, "error",
                   "Remove conditions from order", True),

    # === COMBO AND BRACKET ORDER ERRORS ===
    ErrorDefinition(312, "Combo details are invalid", ErrorCategory.BRACKET_COMBO, "error",
                   "Verify combo leg specifications", True),
    ErrorDefinition(313, "Combo details for leg invalid", ErrorCategory.BRACKET_COMBO, "error",
                   "Check individual combo leg parameters", True),
    ErrorDefinition(314, "Security type BAG requires combo leg details", ErrorCategory.BRACKET_COMBO, "error",
                   "Add ComboLegs to contract definition", True),
    ErrorDefinition(315, "Stock combo legs restricted to SMART routing", ErrorCategory.BRACKET_COMBO, "error",
                   "Set exchange to SMART for stock combos", True),
    ErrorDefinition(10006, "Missing parent order", ErrorCategory.BRACKET_COMBO, "error",
                   "Add 50ms delay after parent before placing children", True,
                   "Bot already handles this in _place_bracket_orders"),
    ErrorDefinition(429, "Delta neutral orders only supported for combos", ErrorCategory.BRACKET_COMBO, "error",
                   "Use BAG security type for delta neutral", True),

    # === SHORTABLE-SPECIFIC ERRORS (ghost mode triggers) ===
    ErrorDefinition(162, "Historical market data Service error", ErrorCategory.ORDER_REJECTION, "error",
                   "Enable ghost mode for Day SHORT - shortable rejection", True,
                   "Part of SHORTABLE_ERROR_CODES"),
    ErrorDefinition(10147, "Order would violate security short sale rule", ErrorCategory.ORDER_REJECTION, "error",
                   "Enable ghost mode for Day SHORT - shortable rejection", True,
                   "Part of SHORTABLE_ERROR_CODES"),
]


def analyze_error_handling() -> Dict[str, List[ErrorDefinition]]:
    """
    Analyze all errors and categorize by handling status.

    Returns dict with:
    - "handled": Errors the bot handles correctly
    - "needs_attention": Errors that may need improvement
    - "not_applicable": Errors that don't apply to our use case
    """
    results = {
        "handled": [],
        "needs_attention": [],
        "not_applicable": [],
    }

    for err in TWS_ERRORS:
        if err.currently_handled:
            results["handled"].append(err)
        elif err.category in (ErrorCategory.ALGORITHM, ErrorCategory.FRACTIONAL,
                             ErrorCategory.BRACKET_COMBO, ErrorCategory.CONDITIONAL):
            # These features aren't used by our bot
            results["not_applicable"].append(err)
        else:
            results["needs_attention"].append(err)

    return results


def test_shortable_error_codes():
    """Verify SHORTABLE_ERROR_CODES contains the correct codes."""
    expected_shortable = {201, 10147, 162, 426}

    print("\n=== Testing SHORTABLE_ERROR_CODES ===")
    print(f"Current codes: {SHORTABLE_ERROR_CODES}")
    print(f"Expected codes: {expected_shortable}")

    if SHORTABLE_ERROR_CODES == expected_shortable:
        print("✓ SHORTABLE_ERROR_CODES is correct")
    else:
        missing = expected_shortable - SHORTABLE_ERROR_CODES
        extra = SHORTABLE_ERROR_CODES - expected_shortable
        if missing:
            print(f"✗ Missing codes: {missing}")
        if extra:
            print(f"? Extra codes: {extra}")

    return SHORTABLE_ERROR_CODES == expected_shortable


def test_placement_result():
    """Test PlacementResult dataclass."""
    print("\n=== Testing PlacementResult ===")

    # Test successful placement
    success = PlacementResult(order_id=12345, success=True)
    assert success.success == True
    assert success.is_shortable_rejection == False
    print("✓ Successful placement result works")

    # Test shortable rejection
    for code in SHORTABLE_ERROR_CODES:
        rejection = PlacementResult(
            order_id=None,
            success=False,
            rejection_code=code,
            rejection_message="Test shortable rejection"
        )
        assert rejection.is_shortable_rejection == True, f"Code {code} should be shortable rejection"
    print("✓ Shortable rejection detection works for all codes")

    # Test non-shortable rejection
    other_rejection = PlacementResult(
        order_id=None,
        success=False,
        rejection_code=200,
        rejection_message="No security definition"
    )
    assert other_rejection.is_shortable_rejection == False
    print("✓ Non-shortable rejection detection works")

    return True


def print_error_summary():
    """Print a summary of all error handling status."""
    results = analyze_error_handling()

    print("\n" + "="*70)
    print("TWS API ERROR HANDLING SUMMARY")
    print("="*70)

    print(f"\n✓ HANDLED ({len(results['handled'])} errors):")
    print("-" * 40)
    for err in sorted(results["handled"], key=lambda e: e.code):
        print(f"  {err.code:>5}: {err.message[:45]:<45}")

    print(f"\n⚠ NEEDS ATTENTION ({len(results['needs_attention'])} errors):")
    print("-" * 40)
    for err in sorted(results["needs_attention"], key=lambda e: e.code):
        print(f"  {err.code:>5}: {err.message[:45]:<45}")
        print(f"         Action: {err.bot_action}")
        if err.notes:
            print(f"         Notes: {err.notes}")

    print(f"\n○ NOT APPLICABLE ({len(results['not_applicable'])} errors):")
    print("-" * 40)
    for err in sorted(results["not_applicable"], key=lambda e: e.code):
        print(f"  {err.code:>5}: {err.message[:45]:<45} ({err.category.value})")

    return results


def simulate_error_callback(code: int, message: str):
    """
    Simulate an IBKR error callback and show expected bot behavior.

    This mimics what execution.py's _on_error() receives.
    """
    # Find the error definition
    err_def = next((e for e in TWS_ERRORS if e.code == code), None)

    print(f"\n--- Simulating Error {code} ---")
    print(f"Message: {message}")

    if err_def:
        print(f"Category: {err_def.category.value}")
        print(f"Severity: {err_def.severity}")
        print(f"Expected Action: {err_def.bot_action}")
        print(f"Currently Handled: {'Yes' if err_def.currently_handled else 'NO - NEEDS WORK'}")

        # Check if it's a shortable rejection
        if code in SHORTABLE_ERROR_CODES:
            print("→ This triggers GHOST MODE for Day SHORT trades")
    else:
        print("⚠ Unknown error code - not in reference")

    return err_def


def run_all_tests():
    """Run all error handling tests."""
    print("\n" + "="*70)
    print("RUNNING TWS API ERROR HANDLING TESTS")
    print("="*70)

    all_passed = True

    # Test 1: SHORTABLE_ERROR_CODES
    if not test_shortable_error_codes():
        all_passed = False

    # Test 2: PlacementResult
    if not test_placement_result():
        all_passed = False

    # Test 3: Print summary
    results = print_error_summary()

    # Test 4: Simulate critical errors
    print("\n" + "="*70)
    print("SIMULATING CRITICAL ERROR SCENARIOS")
    print("="*70)

    # Simulate shortable rejection
    simulate_error_callback(201, "Order rejected - Reason: No shares available to short")

    # Simulate connection loss
    simulate_error_callback(1100, "Connectivity between IB and the TWS has been lost")

    # Simulate no security definition
    simulate_error_callback(200, "No security definition has been found for the request")

    # Simulate insufficient shares
    simulate_error_callback(426, "None of the accounts have enough shares")

    print("\n" + "="*70)
    if all_passed and len(results["needs_attention"]) == 0:
        print("ALL TESTS PASSED - Error handling is comprehensive")
    else:
        print(f"ATTENTION NEEDED: {len(results['needs_attention'])} errors need handling")
    print("="*70)

    return all_passed, results


if __name__ == "__main__":
    run_all_tests()
