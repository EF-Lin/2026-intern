import datetime
import os
from typing import Optional, Union

from obspy import UTCDateTime

DEFAULT_FILENAME = "TW.{num}..S.D.{year}.{julday}"


class Search:
    """
    Locate channel files in ``folder``.

    ``filename`` is a template; supported placeholders are ``{num}`` (channel, 5 digits),
    ``{year}`` (4 digits) and ``{julday}`` (day of year, 3 digits). Placeholders that are
    absent are simply ignored, so a fixed name like ``"TW.{num}..S.D.2026.209"`` still works.
    ``date`` (date / datetime / UTCDateTime) overrides ``year`` and ``julday`` when given.
    """

    def __init__(
        self,
        *,
        folder: str,
        filename: Optional[str] = DEFAULT_FILENAME,
        year: int = 2026,
        julday: int = 189,
        date: Optional[Union[datetime.date, UTCDateTime]] = None,
    ):
        self.folder = os.path.normpath(folder)
        self.filename = DEFAULT_FILENAME if filename is None else filename
        if date is not None:
            if isinstance(date, UTCDateTime):
                year, julday = date.year, date.julday
            else:
                year, julday = date.year, date.timetuple().tm_yday
        self.year = year
        self.julday = julday
        self.folder_files: list = sorted(os.listdir(self.folder))
        self._file_set = set(self.folder_files)

    def _name(self, number: int) -> str:
        return (
            self.filename.replace("{num}", f"{number:05d}")
            .replace("{year}", f"{self.year:04d}")
            .replace("{julday}", f"{self.julday:03d}")
        )

    def find(self, number: int) -> str:
        name = self._name(number)
        if name not in self._file_set:
            raise FileNotFoundError(f"{name} does not exist in {self.folder}")
        return os.path.normpath(f"{self.folder}/{name}")

    def multi_find(self, number: int, count: int) -> list[str]:
        """
        Return files for channels number, number + 1, ..., number + count (inclusive).
        Raises FileNotFoundError listing every missing channel.
        """
        names = [self._name(n) for n in range(number, number + count + 1)]
        missing = [n for n in names if n not in self._file_set]
        if missing:
            raise FileNotFoundError(f"{len(missing)} file(s) missing in {self.folder}: {missing}")
        return [os.path.normpath(f"{self.folder}/{n}") for n in names]
