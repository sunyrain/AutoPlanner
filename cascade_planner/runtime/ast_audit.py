"""Focused AST helpers for repository audits."""
from __future__ import annotations

import ast


def type_checking_import_lines(tree: ast.AST) -> set[int]:
    lines: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        test = node.test
        if not (
            (isinstance(test, ast.Name) and test.id == "TYPE_CHECKING")
            or (isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING")
        ):
            continue
        lines.update(
            child.lineno
            for statement in node.body
            for child in ast.walk(statement)
            if isinstance(child, (ast.Import, ast.ImportFrom))
        )
    return lines


__all__ = ["type_checking_import_lines"]
