import numba as nb
from numba import jit, f8, int32,b1

# Fluid Specific Heat
percGly=20
@jit(f8 (f8 ,f8 ),nopython=True)
def fCp(Tref,percglyvol=percGly):
    if(Tref>=292):
        return 0.6502826505e5 - 0.55296090e8 / Tref + 0.8269255092e2 * percglyvol - 0.8130626690e-1 * percglyvol ** 2 + 0.167409225e11 / Tref ** 2 - 0.180431100e4 * (0.6336420000e-3 + 0.1107710000e-1 * percglyvol - 0.1089140000e-4 * percglyvol ** 2) ** 2 - 0.44220204e7 * (0.6336420000e-3 + 0.1107710000e-1 * percglyvol - 0.1089140000e-4 * percglyvol ** 2) / Tref - 0.16876350e13 / Tref ** 3 - 0.68413806e3 * (0.6336420000e-3 + 0.1107710000e-1 * percglyvol - 0.1089140000e-4 * percglyvol ** 2) ** 3 + 0.52095834e6 * (0.6336420000e-3 + 0.1107710000e-1 * percglyvol - 0.1089140000e-4 * percglyvol ** 2) ** 2 / Tref + 0.54229338e9 * (0.6336420000e-3 + 0.1107710000e-1 * percglyvol - 0.1089140000e-4 * percglyvol ** 2) / Tref ** 2
    else:
        return (0.2451e6 - 0.25033e4 * Tref + 0.867151e1 * Tref ** 2 - 0.100147e-1 * Tref ** 3) * (0.1560553517e2 - 0.13270e5 / Tref + 0.1984462465e-1 * percglyvol - 0.1951194310e-4 * percglyvol ** 2 + 0.40175e7 / Tref ** 2 - 0.43300e0 * (0.6336420000e-3 + 0.1107710000e-1 * percglyvol - 0.1089140000e-4 * percglyvol ** 2) ** 2 - 0.10612e4 * (0.6336420000e-3 + 0.1107710000e-1 * percglyvol - 0.1089140000e-4 * percglyvol ** 2) / Tref - 0.4050e9 / Tref ** 3 - 0.16418e0 * (0.6336420000e-3 + 0.1107710000e-1 * percglyvol - 0.1089140000e-4 * percglyvol ** 2) ** 3 + 0.12502e3 * (0.6336420000e-3 + 0.1107710000e-1 * percglyvol - 0.1089140000e-4 * percglyvol ** 2) ** 2 / Tref + 0.13014e6 * (0.6336420000e-3 + 0.1107710000e-1 * percglyvol - 0.1089140000e-4 * percglyvol ** 2) / Tref ** 2)