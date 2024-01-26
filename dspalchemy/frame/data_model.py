"""
DataFrame Bound to Pydantic Model

The `Frame` class is an extension of the `pandas.DataFrame` class that is bound to a Pydantic model.
It provides additional functionality for validating data against the model, generating error reports,
and creating visualizations. The data within the `Frame` is validated during instantiation, and various
properties and methods are available to analyze and report validation errors.

Attributes:
    - model (BaseModel): The Pydantic model associated with the DataFrame.

Properties:
    - validated_cols (set): Set of columns present in both the DataFrame and the Pydantic model.
    - cols_without_validation (set): Set of columns present in the DataFrame but not in the Pydantic model.
    - error_report (list): List of dictionaries containing error information for invalid entries.
    - error_report_df (pd.DataFrame): DataFrame summarizing the error report.
    - error_count (int): Total number of validation errors.
    - df_entries (int): Total number of entries in the DataFrame.
    - error_percentage (float): Percentage of entries in the DataFrame that contain validation errors.
    - summary_report (dict): Dictionary containing summary information about validation errors per column and error messages.
    - visual_report: Cached property to generate a visual representation of the error report using Matplotlib.

Methods:
    - generate_report(): Generate a visual report of the error overview using Matplotlib.
    - _validate(): Private method to perform validation on DataFrame data using the associated Pydantic model.

Example Usage:
```python
m = M()
print(m)
# Output: M s1=<__main__.MySequence object at ...>
print(m.s1.v)
# Output: [3]
"""
# pylint: disable =unnecessary-pass
import typing
import copy
from functools import cached_property
import pandas as pd
from pydantic import BaseModel, ValidationError, create_model
import matplotlib.pyplot as plt
from dspalchemy.frame.utils import extract_column_name, are_models_equal


class Frame(pd.DataFrame):
    """
    Initialize a Frame object with a Pydantic model and DataFrame data.

    Args:
        - model (BaseModel): The Pydantic model associated with the DataFrame.
        - data (pd.DataFrame): The DataFrame containing the data.
        - **kwargs: Additional keyword arguments to pass to the pandas.DataFrame constructor.

    Raises:
        - ValueError: If the DataFrame is empty or None.
    """

    def __init__(self, model: BaseModel, data: pd.DataFrame, **kwargs):
        if data is None or data.empty:
            raise ValueError("DataFrame cannot be empty or None.")

        super().__init__(data, **kwargs)  # type: ignore [call-arg]
        self.model = model

    @property
    def validated_cols(self) -> typing.Set[str]:
        """
        Get the set of columns present in both the DataFrame and the Pydantic model.

        Returns:
            Set[str]: Set of validated columns.
        """
        model_cols = set(self.model.model_fields.keys())
        data_cols = set(self.columns)
        return model_cols & data_cols

    @property
    def cols_without_validation(self) -> typing.Set[str]:
        """
        Get the set of columns present in the DataFrame but not in the Pydantic model.

        Returns:
            Set[str]: Set of columns that are not validated by a pydantic model.
        """
        model_cols = set(self.model.model_fields.keys())
        data_cols = set(self.columns)
        return data_cols - model_cols

    @cached_property
    def error_report(self) -> typing.List[typing.Dict[str, str]]:
        """
        Get a list of dictionaries containing error information for invalid entries.

        Returns:
            List[Dict[str,str]]: List of dictionaries with error information.
        """
        return self._validate()

    @cached_property
    def error_report_df(self) -> pd.DataFrame:
        """
        Get a DataFrame summarizing the error report.

        Returns:
            pd.DataFrame: DataFrame containing error information.
        """
        return pd.DataFrame(
            self.error_report, columns=["idx", "column", "error_message"]
        )

    @cached_property
    def error_count(self) -> int:
        """
        Get the total number of validation errors.

        Returns:
            int: Total number of validation errors.
        """
        return len(self.error_report)

    @cached_property
    def df_entries(self) -> int:
        """
        Get the total number of entries in the DataFrame.

        Returns:
            int: Total number of entries.
        """
        data_shape = self.shape
        return data_shape[0] * data_shape[1]

    @cached_property
    def error_percentage(self) -> float:
        """
        Get the percentage of entries in the DataFrame that contain validation errors.

        Returns:
            float: Percentage of entries with validation errors.
        """
        return (self.error_count / self.df_entries) * 100

    @cached_property
    def summary_report(self) -> typing.Dict[str, pd.DataFrame]:
        """
        Get a dictionary containing summary information about validation errors per column and error messages.

        Returns:
            Dict[str,pd.DataFrame]: Dictionary of summary dataframes.
        """
        error_counts_per_column = (
            self.error_report_df["column"].value_counts().to_dict()
        )
        percentage_invalid_data_per_column = (
            self.error_report_df.groupby("column").size() / len(self) * 100
        ).to_dict()

        # Create DataFrames with columns without errors for clarity
        columns_with_no_errors = set(self.columns) - set(self.error_report_df["column"])

        # Include columns with no errors in error_counts_per_column and percentage_invalid_data_per_column
        for col in columns_with_no_errors:
            error_counts_per_column[col] = 0
            percentage_invalid_data_per_column[col] = 0

        error_counts_per_column_df = pd.DataFrame.from_dict(
            {
                "column": list(error_counts_per_column.keys()),
                "error_counts": list(error_counts_per_column.values()),
            }
        )

        percentage_invalid_data_per_column_df = pd.DataFrame.from_dict(
            {
                "column": list(percentage_invalid_data_per_column.keys()),
                "percentage_invalid_data": list(
                    percentage_invalid_data_per_column.values()
                ),
            }
        )

        error_counts_per_message = (
            self.error_report_df["error_message"].value_counts().to_dict()
        )
        error_counts_per_message_df = pd.DataFrame.from_dict(
            {
                "error_message": list(error_counts_per_message.keys()),
                "error_counts": list(error_counts_per_message.values()),
            }
        )

        return {
            "error_counts_per_column": error_counts_per_column_df,
            "percentage_invalid_data_per_column": percentage_invalid_data_per_column_df,
            "error_counts_per_message": error_counts_per_message_df,
        }

    @cached_property
    def visual_report(self):
        """
        Cached property to generate a visual representation of the error report using Matplotlib.

        Returns:
            matplotlib.figure.Figure: Matplotlib figure containing visualizations of the error report.
        """
        return self.generate_report()

    def _validate(self) -> typing.List[typing.Dict[str, str]]:
        """
        Private method to perform validation on DataFrame data using the associated Pydantic model.

        Returns:
            list: List of dictionaries containing error information for invalid entries.
        """
        errors = []

        data = self.to_dict(orient="records")

        for idx, record in enumerate(data, start=1):
            try:
                self.model(**record)  # type: ignore [call-arg]
            except ValidationError as e:
                for error in e.errors():
                    error_msg = {
                        "idx": idx,
                        "column": extract_column_name(error["loc"]),
                        "error_message": error["msg"],
                    }
                    errors.append(error_msg)

        return errors

    def generate_report(self) -> typing.Any:
        """
        Generate a visual report of the error overview using Matplotlib.

        Returns:
            matplotlib.figure.Figure: Matplotlib figure containing visualizations of the error report.
        """
        error_report_df = self.error_report_df

        fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(8, 10))
        fig.suptitle("Error Report Overview", fontsize=16)

        # Pie chart: Total errors vs Valid rows (in percentages)
        pie_labels = ["Valid Rows", "Errors"]
        pie_values = [self.df_entries - self.error_count, self.error_count]
        axes[0].pie(pie_values, labels=pie_labels, autopct="%1.1f%%", startangle=90)
        axes[0].set_title("Total Errors vs Valid Rows")

        # Stacked bar chart: Errors and Valid entries per column (in percentages)
        total_rows = len(self)
        error_counts_per_column = error_report_df["column"].value_counts().to_dict()
        percentage_invalid_data_per_column = (
            error_report_df.groupby("column").size() / total_rows * 100
        ).to_dict()

        # Include columns with no errors
        all_columns = set(self.columns)
        columns_with_errors = set(error_report_df["column"])
        columns_with_no_errors = all_columns - columns_with_errors

        for col in columns_with_no_errors:
            error_counts_per_column[col] = 0
            percentage_invalid_data_per_column[col] = 0

        stacked_data = pd.DataFrame(
            {
                "Valid Rows": 100 - pd.Series(percentage_invalid_data_per_column),
                "Errors": pd.Series(percentage_invalid_data_per_column),
            }
        )
        stacked_data.plot(kind="bar", stacked=True, ax=axes[1])
        axes[1].set_title("Errors and Valid Entries per Column")
        axes[1].legend(title="Entry Status", loc="upper right")
        axes[1].set_ylim(0, 100)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        return fig

    def update_model(self, cols: typing.Dict[str, typing.Type]) -> None:
        """
        Update the Pydantic model based on the DataFrame columns and their types.

        Args:
            cols (dict): Dictionary mapping column names to their types.

        This method checks for specified or all columns in the DataFrame and updates the Pydantic model accordingly.
        """

        if cols is None:
            raise AttributeError("You must specify cols in order to update a model")

        fields = {}
        for name, field in self.model.__annotations__.items():
            if name in cols.keys():
                fields[name] = (cols[name], ...)
            else:
                fields[name] = (field, ...)

        for new_name, new_field in cols.items():
            if new_name not in self.model.__annotations__.keys():
                fields[new_name] = (new_field, ...)
        updated_model = create_model(self.model.__name__, __base__=self.model, **fields)  # type: ignore [attr-defined]
        # Assign the updated model to the Frame
        self.model = updated_model
        self._clear_cached_properties()

    def _clear_cached_properties(self) -> None:
        for name in dir(type(self)):
            if isinstance(getattr(type(self), name), cached_property):
                vars(self).pop(name, None)


def create_joined_model(
    left_model: BaseModel,
    right_model: BaseModel,
    join_columns: typing.Union[str, typing.List[str]],
) -> type[BaseModel]:
    """
    Create a combined Pydantic model by joining two existing Pydantic models.

    Args:
        - left_model (BaseModel): The Pydantic model for the left dataset.
        - right_model (BaseModel): The Pydantic model for the right dataset.
        - join_columns (Union[str, List[str]]): Columns to use as the join key.

    Returns:
        BaseModel: The combined Pydantic model.

    Raises:
        - ValueError: If join_column is not present in the combined model or if there is a type mismatch.

    Example:
    ```python
    left_model = MyLeftModel(...)
    right_model = MyRightModel(...)
    join_columns = "id"
    combined_model = create_joined_model(left_model, right_model, join_columns)
    ```
    """
    if isinstance(join_columns, str):
        join_columns = [join_columns]

    fields = {}

    # Add fields from ModelA
    for name, field in left_model.__annotations__.items():
        fields[name] = (field, ...)

    # Add fields from ModelB, excluding the join_columns (assumed to be present in ModelA)
    for name, field in right_model.__annotations__.items():
        if name not in join_columns:
            fields[name] = (field, ...)

    # Validate join column types
    for join_column in join_columns:
        if join_column not in fields:
            raise ValueError(f"{join_column} not found in the combined model.")
        if (
            left_model.__annotations__[join_column]
            != right_model.__annotations__[join_column]
        ):
            raise ValueError(f"Type mismatch for join column {join_column}.")

    combined_model = create_model(
        left_model.__name__ + right_model.__name__, __base__=BaseModel, **fields  # type: ignore [attr-defined]
    )

    return combined_model


def join_frames_and_models(
    left_df: Frame,
    right_df: Frame,
    on: typing.Union[str, typing.List[str]],
    how: typing.Literal["left", "right", "outer", "inner"] = "left",
    sort: bool = False,
) -> Frame:
    """
    Join two DataFrame instances based on specified columns and create a new Frame with a combined Pydantic model.

    Args:
        - left_df (Frame): The left DataFrame.
        - right_df (Frame): The right DataFrame.
        - on (Union[str, List[str]]): Columns to use as the join key.
        - how (Literal["left", "right", "outer", "inner"]): Type of join to perform.
        - sort (bool): Whether to sort the result DataFrame by the join keys.

    Returns:
        Frame: The joined DataFrame with a combined Pydantic model.
    """
    # Reset index to avoid multi-index issues
    left_df.reset_index(drop=True, inplace=True)
    right_df.reset_index(drop=True, inplace=True)

    # Perform DataFrame join without suffixes
    result_df = pd.merge(left_df, right_df, on=on, how=how, sort=sort)
    result_df = result_df.applymap(
        lambda x: None if pd.isna(x) else x
    )  # TODO come up with a better way to handle NaN for Optional typing

    # Create a new Pydantic model for the joined DataFrame
    joined_model = create_joined_model(
        left_model=left_df.model, right_model=right_df.model, join_columns=on
    )

    return Frame(model=joined_model, data=result_df)


def concat_frames_and_models(frames: typing.List[Frame]) -> Frame:
    """
    Concatenate multiple DataFrame instances along axis 0 with ignore_index=True and create a new Frame with a combined Pydantic model.

    Args:
        - frames (List[Frame]): List of DataFrame instances to concatenate.

    Returns:
        Frame: The concatenated DataFrame with a combined Pydantic model.

    Raises:
        - ValueError: If columns with the same name have different types among the DataFrames.
        - ValueError: If columns in DataFrames do not match.

    Example:
    ```python
    df1 = Frame(model=model1, data=data1)
    df2 = Frame(model=model2, data=data2)
    concatenated_df = concat_frames_and_models([df1, df2])
    """
    # Extract models and data from frames
    models = [frame.model for frame in frames]
    dfs = [frame.copy() for frame in frames]

    # Check if columns match among all DataFrames
    first_columns = set(dfs[0].columns)
    for data in dfs[1:]:
        if set(data.columns) != first_columns:
            raise ValueError("Columns in DataFrames do not match.")

    # Concatenate DataFrames
    result_df = pd.concat(dfs, axis=0, ignore_index=True)

    # Check annotations (types) match between all models
    for i in range(1, len(models)):
        models_match = are_models_equal(models[0], models[i])
        if models_match is False:
            raise TypeError("Annotations mismatched between models.")

    concatenated_model = copy.deepcopy(models[0])
    return Frame(model=concatenated_model, data=result_df)
