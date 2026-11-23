import datetime
import os
from typing import Optional, Union

from obspy import UTCDateTime

DEFAULT_FILENAME = "TW.{num}..S.D.{year}.{julday}"
# layout of the MiDAS SeisComP archive on the data server, relative to the archive root
ARCHIVE_FILENAME = "{year}/TW/{num}/S.D/TW.{num}..S.D.{year}.{julday}"


class Search:
    """
    Locate channel files in ``folder``.

    ``filename`` is a template; supported placeholders are ``{num}`` (channel, 5 digits),
    ``{year}`` (4 digits) and ``{julday}`` (day of year, 3 digits). Placeholders that are
    absent are simply ignored, so a fixed name like ``"TW.{num}..S.D.2026.209"`` still works.
    The template may contain sub-folders (``/``), e.g. ``ARCHIVE_FILENAME`` for the server archive
    where every channel has its own folder.
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
        if not os.path.isdir(self.folder):
            raise FileNotFoundError(f"folder {self.folder} does not exist")
        self.filename = DEFAULT_FILENAME if filename is None else filename
        if date is not None:
            if isinstance(date, UTCDateTime):
                year, julday = date.year, date.julday
            else:
                year, julday = date.year, date.timetuple().tm_yday
        self.year = year
        self.julday = julday

    def _name(self, number: int) -> str:
        return (
            self.filename.replace("{num}", f"{number:05d}")
            .replace("{year}", f"{self.year:04d}")
            .replace("{julday}", f"{self.julday:03d}")
        )

    def _path(self, number: int) -> str:
        return os.path.normpath(os.path.join(self.folder, self._name(number)))

    def find(self, number: int) -> str:
        path = self._path(number)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"{self._name(number)} does not exist in {self.folder}")
        return path

    def multi_find(self, number: int, count: int) -> list[str]:
        """
        Return files for channels number, number + 1, ..., number + count (inclusive).
        Raises FileNotFoundError listing every missing channel.
        """
        paths = [self._path(n) for n in range(number, number + count + 1)]
        missing = [p for p in paths if not os.path.isfile(p)]
        if missing:
            raise FileNotFoundError(f"{len(missing)} file(s) missing in {self.folder}: {[os.path.relpath(p, self.folder) for p in missing]}")
        return paths
