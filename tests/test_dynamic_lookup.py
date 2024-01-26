#pylint: skip-file
import pytest
import random
from pydantic import BaseModel, ValidationError
from dspalchemy.types.dynamic import Lookup, Regex, DateValidator


def test_lookup_allowed_values():
    class CustomLookup(Lookup[int]):
        allowed_values = [1, 2]

    class Model(BaseModel):
        value: CustomLookup

    Model(value=1)
    Model(value=2)


@pytest.mark.xfail(raises=ValidationError, strict=True)
def test_lookup_allowed_values_xfail():
    class CustomLookup(Lookup[int]):
        allowed_values = [1, 2]

    class Model(BaseModel):
        value: CustomLookup

    Model(value=3)


@pytest.mark.xfail(
    raises=TypeError,
    strict=True,
    reason="Integers specified in the Lookup typing, allowed values are strings",
)
def test_lookup_mismatched_dtypes_in_allowed():
    class CustomLookup(Lookup[int]):
        allowed_values = ["1", "2"]


def test_lookup_dynamic_allowed_values():
    even_values = [random.randint(0, 100) * 2 for _ in range(5)]

    class CustomLookup(Lookup[int]):
        allowed_values = even_values

    class Model(BaseModel):
        value: CustomLookup

    Model(value=even_values[0])
    Model(value=even_values[1])


@pytest.mark.xfail(
    raises=ValidationError,
    strict=True,
    reason="Lookup specifies dynamic even values, passed value is odd",
)
def test_lookup_dynamic_allowed_values_validation_failure():
    even_values = [random.randint(0, 100) * 2 for _ in range(5)]

    class CustomLookup(Lookup[int]):
        allowed_values = even_values

    class Model(BaseModel):
        value: CustomLookup

    Model(value=1)
    Model(value=3)
