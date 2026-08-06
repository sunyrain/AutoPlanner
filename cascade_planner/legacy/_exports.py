"""Shared loader for deprecated compatibility exports."""
from __future__ import annotations

from importlib import import_module
from typing import Any, Mapping
import warnings


def load_legacy_export(
    name: str,
    exports: Mapping[str, tuple[str, str]],
    *,
    replacement: str,
) -> Any:
    target = exports.get(name)
    if target is None:
        raise AttributeError(name)
    warnings.warn(
        f"{name} is a frozen V3 compatibility export; use {replacement}",
        DeprecationWarning,
        stacklevel=3,
    )
    module_name, attribute_name = target
    return getattr(import_module(module_name), attribute_name)


__all__ = ["load_legacy_export"]
