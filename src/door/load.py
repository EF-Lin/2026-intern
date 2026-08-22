import csv

from obspy import Stream, read
from tqdm import tqdm


def load_mini(files: list | str) -> Stream:
    data = Stream()
    if isinstance(files, list):
        for f in tqdm(files, desc="Reading Data"):
            data += read(f)
    else:
        data = read(files)
    return data


def load_csv(path: str) -> list[dict]:
    with open(path, 'r', encoding="utf-8-sig") as f:
        file = list(csv.DictReader(f))
    return file
