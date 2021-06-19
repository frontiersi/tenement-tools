import concurrent.futures

def _run_async(fn, **inputs):
    """runs the inputs asynchronously"""
    tp = concurrent.futures.ThreadPoolExecutor(1)
    future = tp.submit(fn=fn, **inputs)
    tp.shutdown(False)
    return future