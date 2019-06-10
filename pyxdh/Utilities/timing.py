from functools import wraps
from time import time
import os
import inspect

LOGLEVEL = int(os.getenv("LOGLEVEL", 0))


def timing(f):
    return timing_level(2)(f)


def timing_level(level):
    # Answer from https://codereview.stackexchange.com/a/169876
    #             https://stackoverflow.com/a/17065634
    # For parameterable decorators, refer to
    #             https://stackoverflow.com/a/10176276
    def actual_timing(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            start = time()
            result = f(*args, **kwargs)
            end = time()
            if LOGLEVEL >= level:
                stack = inspect.stack()
                the_file = "\"" + stack[1].filename + "\", line " + str(stack[1].lineno)
                if "self" in stack[1][0].f_locals:
                    the_func = stack[1][0].f_locals["self"].__class__.__qualname__ + "." + stack[1].function
                else:
                    the_func = stack[1].function
                print(" {0:40s}, Elapsed time: {1:17.10f} s | called by {2:40s}, in file {3:}"
                      .format(f.__qualname__, end-start, the_func, the_file))
                stack.clear()
            return result
        return wrapper
    return actual_timing
