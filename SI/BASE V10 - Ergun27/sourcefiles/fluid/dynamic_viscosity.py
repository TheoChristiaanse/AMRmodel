import numba as nb
from numba import jit, f8, int32,b1

# Fluid Dynamic Viscosity
percGly=20
@jit(f8 (f8 ,f8 ),nopython=True)
def fMu(Tref,percglyvol=percGly):
    return (0.79913e0 - 0.97631e-2 * Tref + 0.4496e-4 * Tref ** 2 - 0.92347e-7 * Tref ** 3 + 0.71318e-10 * Tref ** 4) * (-0.4808578086e0 + 0.31652e-2 * Tref - 0.55754e-5 * Tref ** 2 - 0.4861406877e-3 * percglyvol + 0.4779908718e-6 * percglyvol ** 2 - 0.83104e-1 * (0.6336420000e-3 + 0.1107710000e-1 * percglyvol - 0.1089140000e-4 * percglyvol ** 2) ** 2) / (0.1000016871e1 - 0.10521e-1 * Tref + 0.36260e-4 * Tref ** 2 - 0.42298e-7 * Tref ** 3 + 0.2949277875e-3 * percglyvol - 0.2899835250e-6 * percglyvol ** 2)
