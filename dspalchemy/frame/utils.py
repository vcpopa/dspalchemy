"""
This module provides utility functions for working with Pydantic models.

Functions:
- extract_column_name: Extracts the column name from a tuple representing a database column.
- are_models_equal: Checks if two Pydantic models have identical fields.

Classes:
- None
"""

import re
from pydantic import BaseModel


def extract_column_name(column_tuple):
    """
    Extracts the column name from a tuple representing a database column.

    Args:
    - column_tuple: A tuple representing a database column.

    Returns:
    - str: The extracted column name.
    """
    match = re.match(r"\('(.+)',\)", str(column_tuple))
    if match:
        return match.group(1)
    return str(column_tuple)


def are_models_equal(model1: BaseModel, model2: BaseModel) -> bool:
    """
    Checks if two Pydantic models have identical fields.

    Args:
    - model1 (BaseModel): The first Pydantic model.
    - model2 (BaseModel): The second Pydantic model.

    Returns:
    - bool: True if the models have identical fields, False otherwise.
    """
    return all(
        model1.model_fields[field_name].annotation
        == model2.model_fields[field_name].annotation
        for field_name in model1.model_fields.keys()
    )
