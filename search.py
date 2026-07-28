import os


class Search:
    def __init__(self, folder: str, number: tuple[int, int], filename: str = "TW.{num}..S.D.2026.189"):
        self.folder = os.path.normpath(folder)
        self.number = number
        self.filename = filename

        self.start = self.filename.replace("{num}", f"{self.number[0]:05d}")
        self.end = self.filename.replace("{num}", f"{self.number[1]:05d}")
        self.folder_files = sorted(os.listdir(self.folder))

    def find(self) -> list[str]:
        files = []
        flag = False
        for i in self.folder_files:
            if i == self.start:
                flag = True
                files.append(os.path.normpath(f"{self.folder}\\{i}"))
            elif i == self.end:
                flag = False
                files.append(os.path.normpath(f"{self.folder}\\{i}"))
            elif flag:
                files.append(os.path.normpath(f"{self.folder}\\{i}"))
            else:
                pass
        return files


if __name__ == "__main__":
    se = Search(folder="data_100Hz/", number=(55, 100))
    l = se.find()
    print(l)
