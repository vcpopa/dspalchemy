import pytest
import pandas as pd
import timeit
from pydantic import BaseModel
from dspalchemy.frame.data_model import Frame
from dspalchemy.types.dynamic import Lookup


@pytest.fixture
def test_model_data():
    class TestModel(BaseModel):
        col1: int
        col2: str

    data = pd.DataFrame({"col1": [1, 2], "col2": ["a", "b"]})
    return TestModel, data


@pytest.fixture
def test_lookup_model_data():
    class TestLookup(Lookup[int]):
        allowed_values = [1, 2]

    class TestModel(BaseModel):
        col1: TestLookup
        col2: str

    data = pd.DataFrame({"col1": [1, 2], "col2": ["a", "b"]})
    return TestModel, data


@pytest.fixture
def invalid_test_model():
    class InvalidTestModel:
        col1: int
        col2: str

    return InvalidTestModel


@pytest.fixture
def empty_data():
    return pd.DataFrame()


@pytest.fixture
def valid_test_model():
    class TestModel(BaseModel):
        col1: int
        col2: str

    return TestModel


@pytest.fixture
def test_invalid_frame():
    class InvalidTestModel(BaseModel):
        col1: int
        col2: str

    invalid_data = pd.DataFrame({"col1": ["a", "b"], "col2": [1, 2]})
    return Frame(model=InvalidTestModel, data=invalid_data)


@pytest.fixture
def test_frame(test_model_data):
    TestModel, data = test_model_data
    return Frame(model=TestModel, data=data)


@pytest.fixture
def large_frame():
    # Create a larger Frame for testing
    class TestModel(BaseModel):
        col1: int
        col2: str

    data = pd.DataFrame({"col1": range(10000), "col2": ["a", "b"] * 5000})
    return Frame(model=TestModel, data=data)


def test_is_frame(test_frame, test_model_data):
    TestModel, data = test_model_data
    assert isinstance(data, pd.DataFrame), "Not a valid dataframe"
    assert issubclass(TestModel, BaseModel), "Not a valid pydantic BaseModel"
    assert isinstance(test_frame, Frame), "Not a valid frame object"


def test_frame_with_custom_types(test_lookup_model_data):
    TestModel, data = test_lookup_model_data
    frame = Frame(model=TestModel, data=data)
    assert isinstance(data, pd.DataFrame), "Not a valid dataframe"
    assert issubclass(TestModel, BaseModel), "Not a valid pydantic BaseModel"
    assert isinstance(frame, Frame), "Not a valid frame object"
    assert issubclass(TestModel.__annotations__["col1"], Lookup), "Not a valid lookup"


@pytest.mark.xfail(
    raises=TypeError, strict=True, reason="A frame requires a pydantic BaseModel object"
)
def test_init_frame_with_nonpydantic_model(invalid_test_model, test_model_data):
    Frame(model=invalid_test_model, data=test_model_data[1])


@pytest.mark.xfail(
    raises=ValueError, strict=True, reason="Cannot create a frame with empty data"
)
def test_init_frame_with_no_datal(empty_data, valid_test_model):
    Frame(model=valid_test_model, data=empty_data)


def test_validation_function(test_frame):
    test_frame._validate()


def test_validation_function_invalid_data(test_invalid_frame):
    assert (
        not test_invalid_frame.error_report_df.empty
    ), "Error report empty despite validation failure"
    assert (
        test_invalid_frame.error_count == 4
    ), "This data should raise exactly 4 errors"
    assert test_invalid_frame.error_percentage == 100


def test_error_report_caching_large_frame(large_frame):
    # First call to error_report
    first_call_time = timeit.timeit(lambda: large_frame.error_report, number=1)

    # Second call to error_report
    second_call_time = timeit.timeit(lambda: large_frame.error_report, number=1)

    # Ensure that the second call is faster (indicating caching)
    assert second_call_time < first_call_time, "Error report caching is not working"


def test_summary_report_dict(test_frame):
    report = test_frame.summary_report
    assert isinstance(report, dict)
    assert set(report.keys()) == set(
        [
            "error_counts_per_column",
            "percentage_invalid_data_per_column",
            "error_counts_per_message",
        ]
    )
    for key, value in report.items():
        assert isinstance(value, pd.DataFrame), f"{key} dataframe is not a valid df"
