

import numba as nb
from numba import jit, f8, int32,b1
# Numpy library
import numpy as np


# Static conduction Kaviani
@jit(f8(  f8, f8, f8),nopython=True)
def kStat(er, fK, mK):
    return fK * ((1 - 10 ** (0.935844e0 - 0.6778e1 * er)) * (er * (0.8e0 + 0.1e0 * er) + (-er * (0.8e0 + 0.1e0 * er) + 1) * mK / fK) / (1 - er * (0.2e0 - 0.1e0 * er) + mK / fK * er * (0.2e0 - 0.1e0 * er)) + 10 ** (0.935844e0 - 0.6778e1 * er) * (2 * mK ** 2 / fK ** 2 * (1 - er) + (1 + 2 * er) * mK / fK) / ((2 + er) * mK / fK + 1 - er))