import time

def timeit(method):
    def timed(*args, **kw):
        """
        Stores the number of milliseconds it took to run a function.
        Output is the result dict -> keys are function names, 
        values are dicts -> in which keys are iteration number and values are ms.
        """
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())

            if name in kw['log_time']:
                iter_number = len(kw['log_time'][name].keys())
                kw['log_time'][name][iter_number] = int((te - ts) * 1000)
            else:
                kw['log_time'][name] = {}
                kw['log_time'][name][0] = int((te - ts) * 1000)
        else:
            print '%r  %2.2f ms' % \
                  (method.__name__, (te - ts) * 1000)
        return result
    return timed