from functools import wraps
from time import perf_counter


def timer(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = perf_counter()
        try:
            return func(*args, **kwargs)
        finally:
            print(f"{func.__name__} run time: {perf_counter() - start_time:.4f} s")

    return wrapper
