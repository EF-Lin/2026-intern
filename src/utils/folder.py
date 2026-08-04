import os


def mkdir(path: str) -> bool:
    if not os.path.exists(path):
        os.makedirs(os.path.normpath(path))
        return True
    else:
        return False


def check_file(path: str):
    path = os.path.normpath(path)
    i = 1
    li = path.split('.')
    while os.path.exists(path):
        path = f"{''.join(li[:-1])} ({i}).{li[-1]}"
        i += 1
    return path
