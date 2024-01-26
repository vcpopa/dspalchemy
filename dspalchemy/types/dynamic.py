# pylint: disable = unused-argument
import typing
import re
from datetime import datetime, date
from pydantic import field_validator, GetCoreSchemaHandler
from pydantic_core import CoreSchema, core_schema
from dspalchemy.types.utils import translate_optional_type_hint

T = typing.TypeVar("T")


class LookupMeta(type):
    """
    Metaclass for Lookup classes.

    Validates the allowed_values attribute to ensure it matches the specified type hint.
    """

    def __init__(cls, name, bases, attrs):
        super().__init__(name, bases, attrs)

        if "allowed_values" in attrs:
            field_type, nulls_allowed = translate_optional_type_hint(
                cls.__orig_bases__[0].__args__[0]
            )
            non_none_values = [val for val in cls.allowed_values if val is not None]

            # Check for mixed types in non-None values
            if len(set(map(type, non_none_values))) > 1:
                raise TypeError("Lookup does not accept mixed types in allowed_values")
            invalid_allowed_values = [
                (val, type(val).__name__)
                for val in cls.allowed_values
                if not (isinstance(val, field_type) or (nulls_allowed and val is None))
            ]
            if invalid_allowed_values:
                invalid_values = ", ".join(
                    f"{val}:{type_name}" for val, type_name in invalid_allowed_values
                )
                expected_type = (
                    f"{field_type.__name__} or None"
                    if nulls_allowed
                    else field_type.__name__
                )
                raise TypeError(
                    f"Expected {expected_type}, got {invalid_values.replace('<class', '').replace('>', '')}"
                )


class Lookup(typing.Generic[T], metaclass=LookupMeta):
    """
    Custom lookup class.

    Ensures that the allowed_values attribute matches the specified type hint.

    Attributes:
    - allowed_values: List of allowed values.
    """

    allowed_values: typing.List[T]

    @field_validator("allowed_values")
    def validate_allowed_values(cls, v: T, info: core_schema.ValidationInfo):
        """
        Pydantic field validator for allowed_values.

        Validates that the provided value is of the correct type and within the allowed values.

        Args:
        - v (T): The value to be validated.

        Raises:
        - ValueError: If the value is not of the correct type or not within the allowed values.
        """

        if not isinstance(v, cls.__orig_bases__[0].__args__[0]):  # type: ignore [attr-defined]
            raise ValueError(
                f"{v} must be of type {cls.__orig_bases__[0].__args__[0]}, got {type(v)}"  # type: ignore [attr-defined]
            )
        if v not in cls.allowed_values:
            raise ValueError(
                f"{v} must be one of {', '.join(map(str, cls.allowed_values))}"
            )
        return v

    def __repr__(self):
        return f"Lookup({super().__repr__()})"

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source_type: typing.Type[T], handler: GetCoreSchemaHandler
    ) -> CoreSchema:
        """
        Get Pydantic CoreSchema for the Lookup class.

        Args:
        - source_type (Type[T]): The source type.
        - handler (GetCoreSchemaHandler): The core schema handler.

        Returns:
        - CoreSchema: The Pydantic CoreSchema for the Lookup class.
        """
        return core_schema.chain_schema(
            [
                core_schema.with_info_plain_validator_function(
                    function=cls.validate_allowed_values
                ),
            ]
        )


class RegexMeta(type):
    """
    Metaclass for Regex classes.

    Validates the regex attribute to ensure it is a valid regular expression.
    """

    def __init__(cls, name, bases, attrs):
        super().__init__(name, bases, attrs)

        if "regex" in attrs:
            if not isinstance(cls.regex, str):
                raise TypeError("Regex must be a string")
            try:
                re.compile(cls.regex)
            except re.error as err:
                raise ValueError("Invalid regular expression") from err


class Regex(typing.Generic[T], metaclass=RegexMeta):
    """
    Custom regex class.

    Ensures that the regex attribute is a valid regular expression.

    Attributes:
    - regex: The regular expression pattern.
    """

    regex: str

    @field_validator("regex")
    def validate_regex(cls, v: T, values: dict):
        """
        Pydantic field validator for regex.

        Validates that the provided value matches the specified regex.

        Args:
        - v (T): The value to be validated.

        Raises:
        - ValueError: If the value does not match the regex pattern.
        """
        if not re.fullmatch(cls.regex, str(v)):
            raise ValueError(f"{v} does not match the regex pattern")
        return v

    def __repr__(self):
        return f"Regex({super().__repr__()})"

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source_type: typing.Type[T], handler: GetCoreSchemaHandler
    ) -> CoreSchema:
        """
        Get Pydantic CoreSchema for the Regex class.

        Args:
        - source_type (Type[T]): The source type.
        - handler (GetCoreSchemaHandler): The core schema handler.

        Returns:
        - CoreSchema: The Pydantic CoreSchema for the Regex class.
        """
        return core_schema.chain_schema(
            [
                core_schema.with_info_plain_validator_function(
                    function=cls.validate_regex
                ),
            ]
        )


class DateValidatorMeta(type):
    """
    Metaclass for DateValidator classes.

    This metaclass validates the attributes of DateValidator to ensure correct usage.

    Attributes:
    - min_date (T): The minimum allowed date.
    - max_date (T): The maximum allowed date.
    - allowed_format (str): The allowed date format.
    """

    def __init__(cls, name, bases, attrs):
        super().__init__(name, bases, attrs)

        if "min_date" in attrs:
            if not isinstance(cls.min_date, (date, datetime, str, type(None))):
                raise TypeError("min_date must be a date, datetime, string, or None")

        if "max_date" in attrs:
            if not isinstance(cls.max_date, (date, datetime, str, type(None))):
                raise TypeError("max_date must be a date, datetime, string, or None")

        if "allowed_format" in attrs:
            if not isinstance(cls.allowed_format, str):
                raise TypeError("allowed_format must be a string")


class DateValidator(typing.Generic[T], metaclass=DateValidatorMeta):
    """
    A custom date validator class.

    This class validates date values based on specified constraints.

    Attributes:
    - allowed_format (str): The allowed date format. Defaults to "%Y-%m-%d".
    - min_date (T): The minimum allowed date.
    - max_date (T): The maximum allowed date.
    """

    allowed_format: str = "%Y-%m-%d"
    min_date: T = None
    max_date: T = None

    @field_validator("allowed_format", "min_date", "max_date")
    def validate_date(cls, v: T, info: core_schema.ValidationInfo):
        if isinstance(v, str):
            try:
                parsed_date = datetime.strptime(v, cls.allowed_format).date()
            except ValueError as exc:
                raise ValueError(f"Failed to convert {v} to a valid date") from exc

            if cls.min_date is not None:
                cls.min_date = (
                    datetime.strptime(cls.min_date, cls.allowed_format).date()
                    if isinstance(cls.min_date, str)
                    else cls.min_date
                )
                if parsed_date < cls.min_date:
                    raise ValueError(
                        f"{v} is before the allowed minimum date: {cls.min_date}"
                    )

            if cls.max_date is not None:
                cls.max_date = (
                    datetime.strptime(cls.max_date, cls.allowed_format).date()
                    if isinstance(cls.max_date, str)
                    else cls.max_date
                )
                if parsed_date > cls.max_date:
                    raise ValueError(
                        f"{v} is after the allowed maximum date: {cls.max_date}"
                    )

            return parsed_date

        if not isinstance(v, (date, datetime, type(None))):
            raise ValueError(
                f"{v} must be a date, datetime, string convertible to date, or None, got {type(v)}"
            )

        return v

    def __repr__(self):
        return f"DateValidator({super().__repr__()})"

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source_type: typing.Type[T], handler: GetCoreSchemaHandler
    ) -> core_schema.CoreSchema:
        return core_schema.chain_schema(
            [
                core_schema.with_info_plain_validator_function(
                    function=cls.validate_date
                ),
            ]
        )
