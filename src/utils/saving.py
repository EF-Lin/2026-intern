import csv
from typing import Optional

from .folder import check_file


def dicts2csv(
    data: list[dict],
    filename: str = "data.csv",
    fieldnames: Optional[list[str]] = None,
) -> str:
    if data is None:
        return filename

    filename = check_file(filename)

    if fieldnames is None:
        fieldnames = list(dict.fromkeys(k for row in data for k in row.keys()))

    with open(filename, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(data)

    return filename
