import pytest
from pydantic import BaseModel
import pandas as pd
from dspalchemy.frame.data_model import (
    Frame,
    join_frames_and_models,
    concat_frames_and_models,
)


@pytest.fixture()
def base_join_dfs():
    class LeftModel(BaseModel):
        col1: int
        col2: str

    class RightModel(BaseModel):
        col2: str
        col3: int

    left_data = pd.DataFrame({"col1": [1, 2], "col2": ["a", "b"]})
    right_data = pd.DataFrame({"col2": ["a", "c"], "col3": [3, 4]})
    left_frame = Frame(model=LeftModel, data=left_data)
    right_frame = Frame(model=RightModel, data=right_data)
    return left_frame, right_frame


def test_inner_join(base_join_dfs):
    left_frame, right_frame = base_join_dfs
    joined = join_frames_and_models(
        left_df=left_frame, right_df=right_frame, on="col2", how="inner"
    )
    assert issubclass(
        joined.model, BaseModel
    ), "Joined Frame model is not a valid pydantic model"
    assert isinstance(joined, Frame), "Joined Frame is not a valid Frame object"
    assert joined.shape == (
        1,
        3,
    ), f"Expected frame with 1 row and 3 columns, got {joined.shape[0]} rows and {joined.shape[1]} columns"
    assert (
        not joined.error_report
    ), f"Expected 0 validation errors, got {len(joined.error_report)}"
    assert joined.model.model_fields["col1"].annotation == int, "col1 should be an int"
    assert joined.model.model_fields["col2"].annotation == str, "col2 should be str"
    assert joined.model.model_fields["col3"].annotation == int, "col3 should be an int"


def test_left_join(base_join_dfs):
    left_frame, right_frame = base_join_dfs
    joined = join_frames_and_models(
        left_df=left_frame, right_df=right_frame, on="col2", how="left"
    )
    assert issubclass(
        joined.model, BaseModel
    ), "Joined Frame model is not a valid pydantic model"
    assert isinstance(joined, Frame), "Joined Frame is not a valid Frame object"
    assert joined.shape == (
        2,
        3,
    ), f"Expected frame with 1 row and 3 columns, got {joined.shape[0]} rows and {joined.shape[1]} columns"
    assert (
        len(joined.error_report) == 1
    ), f"Expected 1 validation errors, got {len(joined.error_report)}"
    assert joined.error_report[0]["column"] == "col3", "Expected an error in col3"
    assert (
        joined.error_report[0]["error_message"] == "Input should be a finite number"
    ), "Expected different error"
    assert joined.model.model_fields["col1"].annotation == int, "col1 should be an int"
    assert joined.model.model_fields["col2"].annotation == str, "col2 should be str"
    assert joined.model.model_fields["col3"].annotation == int, "col3 should be an int"


@pytest.mark.xfail(
    raises=KeyError,
    strict=True,
    reason="Join column given is not present in both frames",
)
def test_join_on_invalid_col(base_join_dfs):
    left_frame, right_frame = base_join_dfs
    joined = join_frames_and_models(
        left_df=left_frame, right_df=right_frame, on="col3", how="left"
    )


@pytest.mark.xfail(raises=ValueError, strict=True, reason="Type mismatch on join key")
def test_join_on_mismatched_annots():
    class LeftModel(BaseModel):
        col1: int
        col2: str

    class RightModel(BaseModel):
        col2: int
        col3: int

    left_data = pd.DataFrame({"col1": [1, 2], "col2": ["a", "b"]})
    right_data = pd.DataFrame({"col2": ["a", "c"], "col3": [3, 4]})
    left_frame = Frame(model=LeftModel, data=left_data)
    right_frame = Frame(model=RightModel, data=right_data)
    joined = join_frames_and_models(
        left_df=left_frame, right_df=right_frame, on="col2", how="left"
    )
