import csv


def load_csv() -> list[dict]:
    with open("asset/78.csv", 'r', encoding="utf-8-sig") as f:
        file = list(csv.DictReader(f))
    return file[45:49]
