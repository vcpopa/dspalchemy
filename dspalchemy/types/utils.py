import typing


def translate_optional_type_hint(
    type_hint: typing.Type,
) -> typing.Tuple[typing.Type, bool]:
    """
    Translate an Optional type hint.

    Extracts the inner type if the type hint is Optional, and indicates whether None is allowed.

    Args:
    - type_hint (Type): The original type hint.

    Returns:
    - Tuple[Type, bool]: A tuple containing the translated type and a boolean indicating whether None is allowed.

    Example:
    ```python
    translate_optional_type_hint(Optional[int])  # Returns (int, True)
    translate_optional_type_hint(int)             # Returns (int, False)
    ```
    """
    if getattr(type_hint, "__origin__", None) is typing.Union:
        inner_type = type_hint.__args__[0]
        return inner_type, True
    return type_hint, False
