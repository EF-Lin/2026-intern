import atexit
from time import time
from functools import wraps


def timer(func):
    @wraps(func)
    def wapper(*args, **kwargs):
        start_time = time()
        result = func(*args, **kwargs)
        atexit.register(lambda: print(f'Run time: {time() - start_time:.8f}'))
        return result
    return wapper
