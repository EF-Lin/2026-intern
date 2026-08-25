import csv
import os

import dascore as dc
import obspy as op
from tqdm import tqdm


def load_mini(files: list | str) -> op.Stream:
    data = op.Stream()
    if isinstance(files, list):
        for f in tqdm(files, desc="Reading Data"):
            data += op.read(f)
    else:
        data = op.read(files)
    return data


def load_csv(path: str) -> list[dict]:
    path = os.path.normpath(path)
    with open(path, 'r', encoding="utf-8-sig") as f:
        file = list(csv.DictReader(f))
    return file


def load_h5(path: str) -> dc.Patch:
    path = os.path.normpath(path)
    res = dc.read(path, "dasdae")
    return res[0] if hasattr(res, "__getitem__") and not isinstance(res, dc.Patch) else res
