import os
import random
import string


def mkdir(path: str) -> str:
    path = os.path.normpath(path)
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def random_mkdir(parent: str = './') -> str:
    path = parent + ''.join(random.choices(string.ascii_letters + string.digits, k=10))
    while os.path.exists(path):
        path = parent + ''.join(random.choices(string.ascii_letters + string.digits, k=10))
    path = os.path.normpath(path)
    os.makedirs(path)
    return path


def check_file(path: str):
    path = os.path.normpath(path)
    i = 1
    li = path.split('.')
    while os.path.exists(path):
        path = f"{''.join(li[:-1])} ({i}).{li[-1]}"
        i += 1
    return path
