from src.utils.folder import check_file
import pytest


@pytest.mark.parametrize("path, expected",[
    ("image\\figure.png", "image\\figure.png"),
    ("main.py", "main (1).py"),
    ("data_100Hz\\download_errors.log", "data_100Hz\\download_errors (1).log"),
])
def test_check_file(path: str, expected: str) -> None:
    assert check_file(path) == expected
