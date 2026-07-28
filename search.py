import os


class Search:
    def __init__(self, folder: str, filename: str = "TW.{num}..S.D.2026.189"):
        self.folder = os.path.normpath(folder)
        self.filename = filename
        self.folder_files: list = sorted(os.listdir(self.folder))

    def find(self, number: int) -> str:
        try:
            return os.path.normpath(f"{self.folder}\\{self.folder_files[self.folder_files.index(self.filename.replace("{num}", f"{number:05d}"))]}")
        except ValueError as ex:
            return f"Error: the file dose not exist.\n{str(ex)}"
        except Exception as ex:
            return str(ex)

    def multi_find(self, number: int, range: int) -> list[str]:
        try:
            start = self.filename.replace("{num}", f"{number:05d}")
            end = self.filename.replace("{num}", f"{number+range:05d}")
            files = []
            flag = False
            for i in self.folder_files:
                if i == start:
                    flag = True
                    files.append(os.path.normpath(f"{self.folder}\\{i}"))
                elif i == end:
                    flag = False
                    files.append(os.path.normpath(f"{self.folder}\\{i}"))
                elif flag:
                    files.append(os.path.normpath(f"{self.folder}\\{i}"))
                else:
                    pass
            return files
        except Exception as ex:
            return [str(ex)]
