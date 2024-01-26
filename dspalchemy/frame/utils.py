import re
from pydantic import BaseModel


def extract_column_name(column_tuple):
    # Extract the column name using a regex
    match = re.match(r"\('(.+)',\)", str(column_tuple))
    if match:
        return match.group(1)
    else:
        return str(column_tuple)


def are_models_equal(model1: BaseModel, model2: BaseModel) -> bool:
    return all(
        model1.model_fields[field_name].annotation
        == model2.model_fields[field_name].annotation
        for field_name in model1.model_fields.keys()
    )
