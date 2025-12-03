# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Serialization utilities for dataclasses.

Provides safe serialization/deserialization that works with torch.load(weights_only=True).
These utilities are used by the Config base class for policy configurations.
"""

import dataclasses
import operator
import types
from collections.abc import Mapping
from enum import Enum
from functools import reduce
from itertools import starmap
from typing import Union, get_args, get_origin, get_type_hints

# Type argument counts for generic type reconstruction
_MIN_DICT_TYPE_ARGS = 2  # dict[K, V] has 2 type args: key type and value type
_VAR_TUPLE_ARG_COUNT = 2  # tuple[T, ...] has 2 type args: item type and ellipsis

__all__ = ["dataclass_to_dict", "dict_to_dataclass"]


def dataclass_to_dict(obj: object) -> object:
    """Recursively convert a dataclass (or nested structure) to a plain dict.

    Args:
        obj: Object to convert (dataclass, dict, list, tuple, Enum, or primitive).

    Returns:
        Plain dict if obj is a dataclass, otherwise appropriately converted value.
    """
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        result = {}
        for field in dataclasses.fields(obj):
            value = getattr(obj, field.name)
            result[field.name] = dataclass_to_dict(value)
        return result

    if isinstance(obj, dict):
        return {k: dataclass_to_dict(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        # Always return list for JSON compatibility
        return [dataclass_to_dict(item) for item in obj]

    if isinstance(obj, Enum):
        return obj.value

    return obj


def dict_to_dataclass[T](cls: type[T], data: Mapping[str, object]) -> T:
    """Reconstruct a dataclass from a dict using type hints.

    Args:
        cls: The dataclass type to reconstruct.
        data: Dictionary containing field values.

    Returns:
        Reconstructed dataclass instance.

    Raises:
        TypeError: If cls is not a dataclass.
    """
    if not dataclasses.is_dataclass(cls):
        msg = f"Expected dataclass, got {cls}"
        raise TypeError(msg)

    # Get type hints for the class
    try:
        hints = get_type_hints(cls)
    except Exception:  # noqa: BLE001
        # Fallback if type hints can't be resolved
        hints = {}

    kwargs = {}
    for field in dataclasses.fields(cls):
        if field.name not in data:
            continue  # Use default value

        value = data[field.name]
        field_type = hints.get(field.name, field.type)

        # Reconstruct the value based on its type
        kwargs[field.name] = _reconstruct_value(value, field_type)

    return cls(**kwargs)  # type: ignore[return-value]


def _reconstruct_value(value: object, field_type: object) -> object:  # noqa: PLR0911
    """Reconstruct a value based on its expected type.

    Args:
        value: The value to reconstruct.
        field_type: The expected type of the field.

    Returns:
        Reconstructed value matching the expected type.
    """
    if value is None:
        return None

    # Handle Optional types (Union with None)
    origin = get_origin(field_type)
    args = get_args(field_type)

    if origin is type(None):  # NoneType
        return None

    # Handle Optional[X] which is Union[X, None]
    if _is_optional_type(field_type):
        # Get the non-None type
        inner_type = _get_optional_inner_type(field_type)
        return _reconstruct_value(value, inner_type)

    # Handle dict types
    if origin is dict and isinstance(value, dict):
        if len(args) >= _MIN_DICT_TYPE_ARGS:
            _, val_type = args[0], args[1]
            return {k: _reconstruct_value(v, val_type) for k, v in value.items()}
        return value

    # Handle list types
    if origin is list and isinstance(value, list):
        if args:
            item_type = args[0]
            return [_reconstruct_value(item, item_type) for item in value]
        return value

    # Handle tuple types (stored as lists)
    if origin is tuple and isinstance(value, list):
        if args:
            # Handle Tuple[int, ...] (variable length)
            if len(args) == _VAR_TUPLE_ARG_COUNT and args[1] is ...:
                item_type = args[0]
                return tuple(_reconstruct_value(item, item_type) for item in value)
            # Handle Tuple[int, str, float] (fixed length)
            return tuple(starmap(_reconstruct_value, zip(value, args, strict=False)))
        return tuple(value)

    # Handle dataclass types
    actual_type = origin or field_type
    if isinstance(actual_type, type) and dataclasses.is_dataclass(actual_type) and isinstance(value, dict):
        return dict_to_dataclass(actual_type, value)

    # Handle Enum types
    if isinstance(actual_type, type) and issubclass(actual_type, Enum) and not isinstance(value, Enum):
        return actual_type(value)

    # Return as-is for primitive types
    return value


def _is_optional_type(field_type: object) -> bool:
    """Check if a type is Optional[X] (i.e., Union[X, None]).

    Args:
        field_type: The type to check.

    Returns:
        True if the type is Optional[X], False otherwise.
    """
    origin = get_origin(field_type)
    # In Python 3.10+, Optional[X] has origin types.UnionType or typing.Union
    if origin is None:
        return False

    # Check for Python 3.10+ union type (X | None)
    if origin is types.UnionType:
        return type(None) in get_args(field_type)

    # Check for typing.Union
    if origin is Union:
        return type(None) in get_args(field_type)

    return False


def _get_optional_inner_type(field_type: object) -> object:
    """Get the inner type from Optional[X], returning X.

    Args:
        field_type: An Optional type (Union[X, None]).

    Returns:
        The inner type X from Optional[X], or a Union of non-None types.
    """
    args = get_args(field_type)
    non_none_args = [arg for arg in args if arg is not type(None)]
    if len(non_none_args) == 1:
        return non_none_args[0]
    # If multiple non-None types, return as Union using | operator (rare case)
    return reduce(operator.or_, non_none_args)
