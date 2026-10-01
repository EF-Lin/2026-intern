import csv
import os
from pathlib import Path
from typing import Optional

import pytest

from src.utils.saving import dicts2csv


@pytest.mark.parametrize(
    "data, filename, fieldnames, expected_rows",
    [
        (
            [{"name": "Alice", "score": "95"}, {"name": "Bob", "score": "88"}],
            "normal.csv",
            None,
            [{"name": "Alice", "score": "95"}, {"name": "Bob", "score": "88"}],
        ),
        (
            [{"a": "1", "b": "2"}, {"b": "3", "c": "4"}],
            "inconsistent.csv",
            None,
            [{"a": "1", "b": "2", "c": ""}, {"a": "", "b": "3", "c": "4"}],
        ),
        (
            [{"a": "1", "b": "2", "c": "3"}],
            "custom_header.csv",
            ["b", "a"],
            [{"b": "2", "a": "1"}],
        ),
    ],
)
def test_dicts2csv(
    tmp_path: Path,
    data: list[dict],
    filename: str,
    fieldnames: Optional[list[str]],
    expected_rows: list[dict],
) -> None:
    filepath = str(tmp_path / filename)
    out_path = dicts2csv(data, filename=filepath, fieldnames=fieldnames)

    assert os.path.exists(out_path)

    with open(out_path, "r", encoding="utf-8-sig") as f:
        reader = list(csv.DictReader(f))
        assert reader == expected_rows


@pytest.mark.parametrize(
    "data, filename, should_exist",
    [
        (None, "none.csv", False),
        ([], "empty.csv", True),
    ],
)
def test_dicts2csv_edge_cases(
    tmp_path: Path,
    data: Optional[list[dict]],
    filename: str,
    should_exist: bool,
) -> None:
    filepath = str(tmp_path / filename)
    out_path = dicts2csv(data, filename=filepath)

    assert os.path.exists(out_path) == should_exist


def test_dicts2csv_duplicate_file(tmp_path: Path) -> None:
    filepath = str(tmp_path / "dup.csv")
    data = [{"x": "1"}]

    path1 = dicts2csv(data, filename=filepath)
    path2 = dicts2csv(data, filename=filepath)

    assert path1 != path2
    assert os.path.exists(path1)
    assert os.path.exists(path2)
