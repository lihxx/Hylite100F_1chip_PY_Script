#!/usr/bin/env python3
"""Minimal dependency-free .xlsx reader (zipfile + ElementTree, stdlib only)."""

from __future__ import annotations

import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

_MAIN_NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"


def read_rows(path: str | Path) -> list[list[str]]:
    """Read the first worksheet of an .xlsx as rows of cell strings."""
    path = Path(path)
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()

        shared_strings: list[str] = []
        if "xl/sharedStrings.xml" in names:
            root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            for item in root.iter(f"{_MAIN_NS}si"):
                shared_strings.append(
                    "".join(text.text or "" for text in item.iter(f"{_MAIN_NS}t"))
                )

        sheet = "xl/worksheets/sheet1.xml"
        if sheet not in names:
            candidates = sorted(
                name for name in names
                if name.startswith("xl/worksheets/sheet") and name.endswith(".xml")
            )
            if not candidates:
                raise ValueError(f"no worksheet found inside {path}")
            sheet = candidates[0]

        root = ET.fromstring(archive.read(sheet))
        rows: list[list[str]] = []
        for row in root.iter(f"{_MAIN_NS}row"):
            values: list[str] = []
            for cell in row:
                node = cell.find(f"{_MAIN_NS}v")
                if node is None:
                    values.append("")
                    continue
                text = (node.text or "").strip()
                if cell.get("t") == "s":
                    try:
                        text = shared_strings[int(text)]
                    except (ValueError, IndexError):
                        text = ""
                values.append(text)
            rows.append(values)
        return rows


def first_column_int(path: str | Path) -> list[int]:
    """Numeric values of column A, skipping empty and non-numeric cells."""
    values: list[int] = []
    for row in read_rows(path):
        if not row or not row[0].strip():
            continue
        try:
            values.append(int(float(row[0])))
        except ValueError:
            continue
    return values
