import time
from typing import Tuple, Type

def retry(tries: int=3, delay: float=0.5, exceptions: Tuple[Type[BaseException], ...]=(Exception,)): 
    def deco(fn): 
        def wrap(*args, **kwargs): 
            t=tries
            while t > 0: 
                try: 
                    return fn(*args, **kwargs)
                except exceptions: 
                    t -= 1
                    if t== 0: 
                        raise
                    time.sleep(delay)
        return wrap
    return deco