import time

############### profiling ##################

class Timer(object):
    def __init__(self, msg, verbose=True):
        self.msg, self.verbose = msg, verbose
    def __enter__(self):
        self.tstart = time.time()
    def __exit__(self, type, value, traceback):
        if self.verbose:
            print('Time spent %s : %s' % (self.msg, time.time() - self.tstart))
