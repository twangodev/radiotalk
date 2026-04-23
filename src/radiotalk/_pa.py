from __future__ import annotations

from datetime import datetime, timezone
from types import UnionType
from typing import Any, Literal, Union, get_args, get_origin

import pyarrow as pa
from pydantic import BaseModel

_PRIMITIVE_PA: dict[type, pa.DataType] = {
    str: pa.string(),
    int: pa.int64(),
    float: pa.float64(),
    bool: pa.bool_(),
    bytes: pa.binary(),
    datetime: pa.string(),
}


def _annotation_to_pa(annotation: Any) -> pa.DataType:
    origin = get_origin(annotation)
    args = get_args(annotation)

    if origin is Union or origin is UnionType:
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) != 1:
            raise TypeError(f"unsupported Union in model schema: {annotation!r}")
        return _annotation_to_pa(non_none[0])

    if origin is Literal:
        kinds = {type(a) for a in args}
        if len(kinds) != 1:
            raise TypeError(f"mixed-type Literal not supported: {annotation!r}")
        return _PRIMITIVE_PA[next(iter(kinds))]

    if origin is list:
        (inner,) = args
        return pa.list_(_annotation_to_pa(inner))

    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return _model_to_pa_struct(annotation)

    if annotation in _PRIMITIVE_PA:
        return _PRIMITIVE_PA[annotation]

    raise TypeError(f"no pyarrow mapping for annotation: {annotation!r}")


def _model_to_pa_struct(model: type[BaseModel]) -> pa.DataType:
    fields: list[tuple[str, pa.DataType]] = []
    for name, info in model.model_fields.items():
        fields.append((name, _annotation_to_pa(info.annotation)))
    for name, info in model.model_computed_fields.items():
        fields.append((name, _annotation_to_pa(info.return_type)))
    return pa.struct(fields)


def pydantic_to_pa_schema(model: type[BaseModel]) -> pa.Schema:
    struct = _model_to_pa_struct(model)
    return pa.schema([pa.field(f.name, f.type) for f in struct])


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
