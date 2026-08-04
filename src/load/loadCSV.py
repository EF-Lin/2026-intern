import csv


def load_csv(path: str) -> list[dict]:
    with open(path, 'r', encoding="utf-8-sig") as f:
        file = list(csv.DictReader(f))
    return file[0:1]
# 45:45+18
