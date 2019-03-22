import numba as nb
from numba import jit, f8, int32,b1

# Fluid Conduction Coefficient
percGly=20
@jit(f8 (f8 ,f8 ),nopython=True)
def fK(Tref,percglyvol=percGly):
    return (-0.680944e0 + 0.68761e-2 * Tref - 0.87127e-5 * Tref ** 2) * (0.7675368334e0 - 0.64896e-2 * Tref - 0.3169712165e-1 * percglyvol + 0.3116574110e-4 * percglyvol ** 2 - 0.40541e-5 * Tref ** 2 + 0.60202e-1 * (0.6336420000e-3 + 0.1107710000e-1 * percglyvol - 0.1089140000e-4 * percglyvol ** 2) ** 2 + 0.10605e-1 * Tref * (0.6336420000e-3 + 0.1107710000e-1 * percglyvol - 0.1089140000e-4 * percglyvol ** 2)) / (0.1000069631e1 - 0.78039e-2 * Tref + 0.1217262519e-2 * percglyvol - 0.1196855946e-5 * percglyvol ** 2 - 0.19933e-5 * Tref ** 2 - 0.20530e0 * (0.6336420000e-3 + 0.1107710000e-1 * percglyvol - 0.1089140000e-4 * percglyvol ** 2) ** 2 - 0.29614e-2 * Tref * (0.6336420000e-3 + 0.1107710000e-1 * percglyvol - 0.1089140000e-4 * percglyvol ** 2))