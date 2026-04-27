"""Generic per-tool settings schema.

Each MLTool can declare a list of SettingField describing user-tunable knobs
(model variant, threshold, min lesion size, etc.).  The API layer renders
the schema for the frontend; user changes are persisted in the tool_settings
table and applied to the live tool instance via MLTool.apply_settings().
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class SettingType(StrEnum):
    SELECT = "select"
    TOGGLE = "toggle"
    NUMBER = "number"


@dataclass(frozen=True)
class SettingOption:
    value: str
    label: str
    description: str = ""


@dataclass(frozen=True)
class SettingField:
    key: str
    label: str
    type: SettingType
    default: Any
    description: str = ""
    options: list[SettingOption] = field(default_factory=list)  # SELECT only
    min: float | None = None  # NUMBER only
    max: float | None = None  # NUMBER only
    step: float | None = None  # NUMBER only


class SettingsValidationError(ValueError):
    """Raised when user-supplied settings fail schema validation."""


def validate_values(schema: list[SettingField], values: dict[str, Any]) -> dict[str, Any]:
    """Coerce/validate user-supplied values against a tool's schema.

    Unknown keys are rejected; missing keys are filled with defaults.
    """
    by_key = {f.key: f for f in schema}
    unknown = set(values) - set(by_key)
    if unknown:
        raise SettingsValidationError(f"Unknown setting keys: {sorted(unknown)}")

    out: dict[str, Any] = {}
    for f in schema:
        if f.key in values:
            v = values[f.key]
            out[f.key] = _validate_one(f, v)
        else:
            out[f.key] = f.default
    return out


def _validate_one(field_: SettingField, value: Any) -> Any:
    if field_.type == SettingType.SELECT:
        valid = {o.value for o in field_.options}
        if value not in valid:
            raise SettingsValidationError(f"{field_.key}: {value!r} not in {sorted(valid)}")
        return value
    if field_.type == SettingType.TOGGLE:
        if not isinstance(value, bool):
            raise SettingsValidationError(
                f"{field_.key}: expected bool, got {type(value).__name__}"
            )
        return value
    if field_.type == SettingType.NUMBER:
        try:
            v = float(value)
        except (TypeError, ValueError) as e:
            raise SettingsValidationError(f"{field_.key}: not a number ({value!r})") from e
        if field_.min is not None and v < field_.min:
            raise SettingsValidationError(f"{field_.key}: {v} < min {field_.min}")
        if field_.max is not None and v > field_.max:
            raise SettingsValidationError(f"{field_.key}: {v} > max {field_.max}")
        return v
    raise SettingsValidationError(f"{field_.key}: unknown type {field_.type!r}")


def schema_to_dict(schema: list[SettingField]) -> list[dict[str, Any]]:
    """Serialize a schema to JSON-friendly dicts for the API."""
    out: list[dict[str, Any]] = []
    for f in schema:
        item: dict[str, Any] = {
            "key": f.key,
            "label": f.label,
            "type": f.type.value,
            "default": f.default,
            "description": f.description,
        }
        if f.options:
            item["options"] = [
                {"value": o.value, "label": o.label, "description": o.description}
                for o in f.options
            ]
        if f.min is not None:
            item["min"] = f.min
        if f.max is not None:
            item["max"] = f.max
        if f.step is not None:
            item["step"] = f.step
        out.append(item)
    return out
