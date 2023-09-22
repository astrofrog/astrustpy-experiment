import time
import numpy as np
from astropy.stats._stats import ks_2samp

from astropy.stats._stats import ks_2samp as ks_2samp_rust
# from astrustpy import ks_2samp as ks_2samp_rust

R = 100

print("          cython        rust")
for N in [100_000, 1_000_000, 10_000_000]:
    print(f'N={N}')
    for dtype in ['i4', 'f4', 'f8']:
        x = (np.random.random(N) * (2**32 - 1)).astype(dtype)
        y = (np.random.random(N) * (2**31 - 1)).astype(dtype)
        start1 = time.time()
        for repeat in range(R):
            ks_2samp(x, y)
        end1 = time.time()
        start2 = time.time()
        for repeat in range(R):
            ks_2samp_rust(x, y)
        end2 = time.time()
        print(f' {dtype} {1000 * (end1 - start1) / R:10.2f}ms {1000 * (end2 - start2) / R:10.2f}ms')
