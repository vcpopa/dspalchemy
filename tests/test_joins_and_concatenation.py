import pytest
import pandas as pd
from dspalchemy.frame.data_model import (
    Frame,
    join_frames_and_models,
    concat_frames_and_models,
)
from dspalchemy.types.dynamic import Lookup
from pydantic import BaseModel
import typing


@pytest.fixture
def test_model_left():
    class TestModelLeft(BaseModel):
        col1: int
        col2: str

    return TestModelLeft


@pytest.fixture
def test_model_right():
    class TestModelRight(BaseModel):
        col1: int
        col3: float

    return TestModelRight


@pytest.fixture
def test_lookup():
    class TestLookup(Lookup[int]):
        allowed_values = [1, 2]

    return TestLookup


@pytest.fixture
def data_and_frames(test_model_left, test_model_right):
    data_left = pd.DataFrame({"col1": [1, 2], "col2": ["a", "b"]})
    data_right = pd.DataFrame({"col1": [1, 2], "col3": [3.14, 2.71]})
    frame_left = Frame(model=test_model_left, data=data_left)
    frame_right = Frame(model=test_model_right, data=data_right)
    return data_left, data_right, frame_left, frame_right


@pytest.fixture
def joined_frame(data_and_frames):
    data_left, data_right, frame_left, frame_right = data_and_frames
    return join_frames_and_models(
        left_df=frame_left, right_df=frame_right, on="col1", how="left", sort=True
    )


@pytest.fixture
def concatenated_frame(data_and_frames):
    _, _, frame_left, frame_right = data_and_frames
    return concat_frames_and_models([frame_left, frame_right])


@pytest.mark.xfail(raises=ValueError, strict=True, reason="Invalid join type")
def test_join_frames_and_models_invalid_join_type(data_and_frames):
    _, _, frame_left, frame_right = data_and_frames
    join_frames_and_models(
        left_df=frame_left, right_df=frame_right, on="col1", how="invalid_join_type"
    )


@pytest.mark.xfail(
    raises=ValueError, strict=True, reason="Columns in DataFrames do not match"
)
def test_join_frames_and_models_columns_mismatch(data_and_frames):
    _, _, frame_left, frame_right = data_and_frames
    join_frames_and_models(
        left_df=frame_left, right_df=frame_right, on="col1", how="left", sort=True
    )


@pytest.mark.xfail(
    raises=TypeError, strict=True, reason="Annotations mismatched between models"
)
def test_concat_frames_and_models_annotations_mismatch(data_and_frames):
    _, _, frame_left, frame_right = data_and_frames
    concat_frames_and_models([frame_left, frame_right, frame_left])


@pytest.mark.xfail(
    raises=ValueError, strict=True, reason="Columns in DataFrames do not match"
)
def test_concat_frames_and_models_columns_mismatch(data_and_frames):
    _, _, frame_left, frame_right = data_and_frames
    concat_frames_and_models([frame_left, frame_right.iloc[:, :-1]])
