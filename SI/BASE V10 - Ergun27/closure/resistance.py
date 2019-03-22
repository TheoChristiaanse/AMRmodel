
import numba as nb
from numba import jit, f8, int32,b1
# Numpy library
import numpy as np



# This calculates the thermal resistance for the regenerator.
@jit(f8(              f8,  f8,  f8,   f8,   f8, f8,   f8, f8, f8, f8),nopython=True)
def ThermalResistance(Dsp, Ud, fMu, fRho, kair, kf, kg10, r1, r2, r3):
    return 0.1e1 / (0.5882352941e1 * (fRho * np.abs(Ud) * Dsp / fMu) ** (-0.79e0) / kf * Dsp + 0.1e1 / kg10 * r1 * np.log(r2 / r1) + 0.1e1 / kair * r1 * np.log(r3 / r2))


# This calculates the thermal resistance for the void space.
@jit(f8(                  f8,  f8,  f8,   f8,   f8, f8, f8, f8),nopython=True)
def ThermalResistanceVoid(kair, kf, kg10, kult, r0, r1, r2, r3):
    return 0.1e1 / (0.4587155963e0 / kf * r0 + 0.1e1 / kult * r0 * np.log(r1 / r0) + 0.1e1 / kg10 * r0 * np.log(r2 / r1) + 0.1e1 / kair * r0 * np.log(r3 / r2))