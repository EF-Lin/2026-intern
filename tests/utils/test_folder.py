import os

import pytest

from src.utils.folder import check_file


@pytest.fixture
def in_tmp(tmp_path, monkeypatch):
    (tmp_path / "main.py").touch()
    (tmp_path / "data_100Hz").mkdir()
    (tmp_path / "data_100Hz" / "download_errors.log").touch()
    (tmp_path / "taken.txt").touch()
    (tmp_path / "taken (1).txt").touch()
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.mark.parametrize(
    "path, expected",
    [
        (os.path.join("image", "figure.png"), os.path.join("image", "figure.png")),
        ("main.py", "main (1).py"),
        (os.path.join("data_100Hz", "download_errors.log"), os.path.join("data_100Hz", "download_errors (1).log")),
        ("taken.txt", "taken (2).txt"),
    ],
)
def test_check_file(in_tmp, path: str, expected: str) -> None:
    assert check_file(path) == expected
