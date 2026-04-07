# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Shared UVC control metadata model."""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator


class CameraSetting(BaseModel):
    """Metadata about a single camera setting."""

    id: int | str = Field(description="Backend setting ID (int for V4L2, str for OmniCamera).")
    name: str = Field(description="Human-readable setting name.")
    setting_type: str = Field(description="Setting type (e.g. 'integer', 'boolean', 'menu').")
    min: int | None = Field(
        default=None, description="Minimum allowed value. None when unavailable or for boolean/button settings."
    )
    max: int | None = Field(
        default=None, description="Maximum allowed value. None when unavailable or for boolean/button settings."
    )
    step: int | None = Field(
        default=None, description="Step increment. None when unavailable or for boolean/button settings."
    )
    default: int | bool | str | None = Field(default=None, description="Default value. None when unavailable.")
    value: int | bool | str | None = Field(
        default=None, description="Current value. None when unavailable or for button settings."
    )
    inactive: bool = Field(default=False, description="Setting is currently locked by another setting.")
    read_only: bool = Field(default=False, description="Setting value cannot be changed.")
    menu_items: dict[int, str] | None = Field(
        default=None, description="Menu index-to-label mapping for menu settings."
    )

    @model_validator(mode="after")
    def _fill_value_from_default(self) -> CameraSetting:
        if self.value is None and self.setting_type != "button" and self.default is not None:
            self.value = self.default
        return self

    @model_validator(mode="after")
    def _normalize_and_validate_value_type(self) -> CameraSetting:
        if self.setting_type == "button":
            self.value = None
            return self

        # Allow None when the backend cannot report a current value.
        if self.value is None:
            return self

        if self.setting_type == "boolean":
            if isinstance(self.value, bool):
                return self
            if isinstance(self.value, int) and self.value in {0, 1}:
                self.value = bool(self.value)
                return self
            msg = f"{self.name}: boolean setting expects bool or 0/1"
            raise ValueError(msg)

        if self.setting_type == "string":
            if isinstance(self.value, str):
                return self
            msg = f"{self.name}: string setting expects str"
            raise ValueError(msg)

        return self

    @model_validator(mode="after")
    def _validate_menu_value(self) -> CameraSetting:
        if self.setting_type == "menu":
            if isinstance(self.value, str):
                if not self.menu_items:
                    msg = f"{self.name}: menu labels are unavailable"
                    raise ValueError(msg)
                label_to_index = {label: idx for idx, label in self.menu_items.items()}
                if self.value not in label_to_index:
                    msg = f"{self.name}: unknown menu label '{self.value}'"
                    raise ValueError(msg)
                self.value = label_to_index[self.value]
                return self
            if isinstance(self.value, int) and not isinstance(self.value, bool):
                return self
            msg = f"{self.name}: menu setting expects int index or label"
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def _validate_min_max_bounds(self) -> CameraSetting:
        if self.min is not None and self.max is not None:
            if self.min > self.max:
                msg = f"{self.name}: min ({self.min}) > max ({self.max})"
                raise ValueError(msg)

            if (
                isinstance(self.default, int)
                and not isinstance(self.default, bool)
                and not (self.min <= self.default <= self.max)
            ):
                msg = f"{self.name}: default ({self.default}) outside [{self.min}, {self.max}]"
                raise ValueError(msg)

            if (
                isinstance(self.value, int)
                and not isinstance(self.value, bool)
                and not (self.min <= self.value <= self.max)
            ):
                msg = f"{self.name}: value ({self.value}) outside [{self.min}, {self.max}]"
                raise ValueError(msg)

        return self


__all__ = ["CameraSetting"]
