import os
import random
import string


def mkdir(path: str) -> str:
    path = os.path.normpath(path)
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def random_mkdir(parent: str = './') -> str:
    def _name() -> str:
        return os.path.normpath(os.path.join(parent, ''.join(random.choices(string.ascii_letters + string.digits, k=10))))

    path = _name()
    while os.path.exists(path):
        path = _name()
    os.makedirs(path)
    return path


def check_file(path: str):
    drive, tail = os.path.splitdrive(path)
    illegal = ['<', '>', ':', '"', '|', '?', '*']
    for i in illegal:
        tail = tail.replace(i, '')
    path = os.path.normpath(drive + tail)
    root, ext = os.path.splitext(path)

    i = 1
    while os.path.exists(path):
        path = f"{root} ({i}){ext}"
        i += 1
    return path
