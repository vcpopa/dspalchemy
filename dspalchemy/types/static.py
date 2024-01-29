"""
This module defines custom validators for various data types used in applications.

Validators:
- PostCode: Represents a validator for UK postcodes.
- Email: Represents a validator for email addresses.
- NHSNumber: Represents a validator for NHS numbers.
- CurrentFinancialYear: Represents a validator for dates falling within the current financial year.
"""
from datetime import datetime
from dspalchemy.types.dynamic import Regex, DateValidator


class PostCode(Regex):
    """
    Represents a PostCode validator that inherits from the `Regex` module.

    The regular expression pattern is designed to match UK postcodes.

    Examples:
    - "SW1A 1AA"
    - "EC1A 1BB"
    - "W1A 0AX"
    - "B33 8TH"
    - "ASCN 1ZZ"
    - "BFPO 1234"
    - "KY11 2ZZ"
    """

    regex = (
        r"(?:"
        r"([A-Z]{1,2}[0-9][A-Z0-9]?|ASCN|STHL|TDCU|BBND|[BFS]IQQ|PCRN|TKCA) ?"
        r"([0-9][A-Z]{2})|"
        r"(BFPO) ?([0-9]{1,4})|"
        r"(KY[0-9]|MSR|VG|AI)[ -]?[0-9]{4}|"
        r"([A-Z]{2}) ?([0-9]{2})|"
        r"(GE) ?(CX)|"
        r"(GIR) ?(0A{2})|"
        r"(SAN) ?(TA1)"
        r")"
    )


class Email(Regex):
    """
    Represents an Email validator that inherits from the `Regex` module.

    The regular expression pattern is designed to match valid email addresses.

    Examples:
    - "user@example.com"
    - "user.name@example.co.uk"
    - "john_doe123@example.org"
    """

    regex = (
        r"(^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$)"
        r"|"
        r"(\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b)"
    )


class NHSNumber(Regex):
    """
    Represents an NHS Number validator that inherits from the `Regex` module.

    The regular expression pattern is designed to match valid NHS Numbers.

    Examples:
    - "123-456-7890"
    - "9876543210"
    """

    regex = r"^\d{3}-\d{3}-\d{4}$|^\d{10}$"


class CurrentFinancialYear(DateValidator):
    """
    Represents a Current Financial Year date validator that inherits from the `DateValidator` module.

    Validates that a date falls within the current financial year, considering a custom date format.

    Examples:
    - "2022-01-15" (Valid if within the financial year starting from April 1, 2021, to March 31, 2022)
    - "2021-06-30" (Valid if within the same financial year)
    - "2023-02-28" (Invalid if after March 31, 2023)
    """

    allowed_format = "%Y-%m-%d"
    min_date = datetime(datetime.now().year - 1, 4, 1).date()
    max_date = datetime(datetime.now().year, 3, 31).date()
