import numba as nb
from numba import jit, f8, int32,b1

# Fluid Density
percGly=20
@jit(f8 (f8 ,f8 ),nopython=True)
def fRho(Tref,percglyvol=percGly):
    if(Tref<230):
        return 1000
    else:
        return (-0.517378e3 + 0.143157e2 * Tref - 0.4405e-1 * Tref ** 2 + 0.4384e-4 * Tref ** 3) * (0.1088984040e1 - 0.10227e-1 * Tref + 0.31258e-4 * Tref ** 2 - 0.32614e-7 * Tref ** 3 - 0.2790099948e-3 * percglyvol + 0.2743325832e-6 * percglyvol ** 2 - 0.60141e-3 * (0.6336420000e-3 + 0.1107710000e-1 * percglyvol - 0.1089140000e-4 * percglyvol ** 2) ** 2) / (0.9999884348e0 - 0.9426e-2 * Tref + 0.28927e-4 * Tref ** 2 - 0.30449e-7 * Tref ** 3 - 0.2021792292e-3 * percglyvol + 0.1987898328e-6 * percglyvol ** 2)