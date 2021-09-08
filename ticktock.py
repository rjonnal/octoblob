import time

def tick():
    return time.time()

def tock(t0,verbose=False):
    dt = time.time()-t0
    if verbose:
        print('tock: %0.3f'%dt)
    return dt



