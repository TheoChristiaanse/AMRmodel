# Numpy library
import numpy as np
# Interpolation Functions
from scipy.interpolate import RectBivariateSpline, interp1d, interp2d


########################################## Applied Field Data ####################################
hapl = np.loadtxt('sourcefiles/device/muhapl_pm1.txt')
Rotation = hapl[0, 1:]
xPosition = hapl[1:, 0]
# Build interpolation function
appliedField = RectBivariateSpline(xPosition, Rotation, hapl[1:, 1:], ky=1, kx=1)