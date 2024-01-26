#pylint: skip-file
import pytest
from pydantic import BaseModel, ValidationError
from dspalchemy.types.static import PostCode, CurrentFinancialYear, Email


@pytest.mark.parametrize(
    "postcode",
    [
        "SW1A 1AA",
        "EC1A 1BB",
        "W1A 0AX",
        "B33 8TH",
        "ASCN 1ZZ",
        "BFPO 1234",
        "KY11 2ZZ",
        # Add more valid postcodes here
    ],
)
def test_postcode_validator(postcode):
    class Model(BaseModel):
        value: PostCode

    Model(value=postcode)


@pytest.mark.parametrize(
    "invalid_postcode",
    [
        "XYZ",  # Invalid format
        "12345",  # Invalid length
        "EC1A-1BB",  # Invalid separator
        # Add more invalid postcodes here
    ],
)
@pytest.mark.xfail(strict=True, raises=ValidationError)
def test_postcode_validator_invalid(invalid_postcode):
    class Model(BaseModel):
        value: PostCode

    Model(value=invalid_postcode)


@pytest.mark.parametrize(
    "email",
    [
        "user@example.com",
        "user.name@example.co.uk",
        "john_doe123@example.org",
        # Add more valid emails here
    ],
)
def test_email_validator(email):
    class Model(BaseModel):
        value: Email

    Model(value=email)
