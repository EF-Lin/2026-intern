import atexit
from functools import wraps
from time import time


def timer(func):
    @wraps(func)
    def wapper(*args, **kwargs):
        start_time = time()
        atexit.register(lambda: print(f'Run time: {time() - start_time:.8f}'))
        return func(*args, **kwargs)

    return wapper
