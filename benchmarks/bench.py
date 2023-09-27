import time
import numpy as np
#from astropy.stats._stats import ks_2samp
from cython_stats.badstats import ks_2samp_opt as ks_2samp_opt_cy
from cython_stats.badstats import ks_2samp_unopt as ks_2samp_unopt_cy

from astrustpy import ks_2samp as ks_2samp_rust
from astrustpy import ks_2samp_buffer

R = 100
dtypes = ['f8'] # ['i4', 'f4', 'f8']
Ns = [10_000, 100_000, 1_000_000, 10_000_000]
which_kinds = ['cython', 'rust_buffer']


times = {}

print((' '*12).join(which_kinds))

for N in Ns:
    print("N=", N)
    for dtype in dtypes:
        print('dtype=', dtype)
        x = (np.random.random(N) * (2**32 - 1)).astype(dtype)
        xb = x.data
        y = (np.random.random(N) * (2**31 - 1)).astype(dtype)
        yb = y.data

        if 'cython' in which_kinds:
            start = time.time()
            for repeat in range(R):
                ks_2samp_opt_cy(x, y)
            end = time.time()
            times[('cython', N, dtype)] = end - start

        if 'cython_unopt' in which_kinds:
            start = time.time()
            for repeat in range(R):
                ks_2samp_unopt_cy(x, y)
            end = time.time()
            times[('cython_unopt', N, dtype)] = end - start

        if 'rust' in which_kinds:
            start = time.time()
            for repeat in range(R):
                ks_2samp_rust(x, y)
            end = time.time()
            times[('rust', N, dtype)] = end - start

        if 'rust_buffer' in which_kinds:
            start = time.time()
            for repeat in range(R):
                ks_2samp_buffer(xb, yb)
            end = time.time()
            times[('rust_buffer', N, dtype)] = end - start


        line = ''
        for kind in which_kinds:
            line += f'{times[(kind, N, dtype)]*1000:10.2f}ms\t'
        print(line)
