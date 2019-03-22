
import numba as nb
from numba import jit, f8, int32,b1
# Numpy library
import numpy as np

# Dynamic conduction in the fluid based on Paulo's work
@jit(f8 (   f8, f8,  f8, f8,   f8, f8),nopython=True)
def kDyn_P(Dsp, er, fCp, fK, fRho, Ud):
    PeNum = fCp / fK * np.abs(Ud) * Dsp * fRho
    if PeNum < 0.01 :
        kd = fK + fRho ** 2 * fCp ** 2 / fK * np.sqrt(0.2e1) * (np.abs(Ud)**2) / (er** 2) * Dsp**2 * ((1 - er) ** (-0.1e1 / 0.2e1)) / 0.240e3
    else:
        kd = fK + 0.375 * fRho * fCp * np.abs(Ud) * Dsp
    return kd