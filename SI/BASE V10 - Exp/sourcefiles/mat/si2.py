# Numpy library
import numpy as np
# Interpolation Functions
from scipy.interpolate import RectBivariateSpline, interp1d, interp2d


######################### Data loading of Material data Si1 Material #########################

# Specifc Heat
cpdat_c   = np.loadtxt('sourcefiles/mat/ssi2/M2_cp_Si_05_AT_G_Q20_c.txt')
cpdat_h   = np.loadtxt('sourcefiles/mat/ssi2/M2_cp_Si_05_AT_G_Q20_h.txt')
# Magnetization
magdata_c = np.loadtxt('sourcefiles/mat/ssi2/M2_Mag_Si_05_AT_G_Q20_c.txt')
magdata_h = np.loadtxt('sourcefiles/mat/ssi2/M2_Mag_Si_05_AT_G_Q20_h.txt')
# Entropy
datstot_c = np.loadtxt('sourcefiles/mat/ssi2/M2_S_Si_05_AT_G_Q20_c.txt')
datstot_h = np.loadtxt('sourcefiles/mat/ssi2/M2_S_Si_05_AT_G_Q20_h.txt')

######################### Make Material Data Functions Si1 Material ##########################

#########################  Specific Heat
# Specifc Heat cooling
HintCp_c = cpdat_c[0, 1:]
TempCp_c = cpdat_c[1:, 0]
# Build interpolation function
mCp_c = RectBivariateSpline(TempCp_c, HintCp_c, cpdat_c[1:, 1:], ky=1, kx=1)
# Specifc Heat heating
HintCp_h = cpdat_h[0, 1:]
TempCp_h = cpdat_h[1:, 0]
# Build interpolation function
mCp_h = RectBivariateSpline(TempCp_h, HintCp_h, cpdat_h[1:, 1:], ky=1, kx=1)

#########################  Magnetisation
# Magnetisation cooling
HintMag_c = magdata_c[0, 1:]
TempMag_c = magdata_c[1:, 0]
# Build interpolation function
mMag_c = RectBivariateSpline(TempMag_c, HintMag_c, magdata_c[1:, 1:], ky=1, kx=1)
# Magnetisation heating
HintMag_h = magdata_h[0, 1:]
TempMag_h = magdata_h[1:, 0]
# Build interpolation function
mMag_h = RectBivariateSpline(TempMag_h, HintMag_h, magdata_h[1:, 1:], ky=1, kx=1)
# Anhysteretic Magnetisation
minTempMag  = np.round(np.max([np.min(TempMag_h),np.min(TempMag_c)]))
maxTempMag  = np.round(np.min([np.max(TempMag_h),np.max(TempMag_c)]))
nPoint      = 4000
rangeTemp   = np.linspace(minTempMag, maxTempMag, num=nPoint + 1)
nMag_an     = np.zeros([np.size(rangeTemp), np.size(HintMag_h)])
for i in range(np.size(HintMag_h)):
    nMag_an[:, i] = np.ndarray.flatten(mMag_c(rangeTemp, HintMag_c[i])*.5+mMag_h(rangeTemp, HintMag_h[i])*.5)
mMag_an = RectBivariateSpline(rangeTemp, HintMag_h, nMag_an, kx=1, ky=1)

#########################  Entropy
# Entropy Cooling
HintStot_c = datstot_c[0, 1:]
TempStot_c = datstot_c[1:, 0]
# Build interpolation function
mS_c = RectBivariateSpline(TempStot_c, HintStot_c, datstot_c[1:, 1:], kx=1, ky=1)
# Entropy Heating
HintStot_h = datstot_h[0, 1:]
TempStot_h = datstot_h[1:, 0]
# Build interpolation function
mS_h = RectBivariateSpline(TempStot_h, HintStot_h, datstot_h[1:, 1:], kx=1, ky=1)
# Anhysteretic Entropy Cooling
minTempSan  = np.round(np.max([np.min(TempStot_h),np.min(TempStot_c)]))
maxTempSan  = np.round(np.min([np.max(TempStot_h),np.max(TempStot_c)]))
nPoint      = 4000
rangeTempSan   = np.linspace(minTempSan, maxTempSan, num=nPoint + 1)
datstot_an     = np.zeros([np.size(rangeTempSan), np.size(HintStot_h)])
for i in range(np.size(HintStot_h)):
    datstot_an[:, i] = np.ndarray.flatten(mS_c(rangeTempSan, HintStot_c[i])*.5+mS_h(rangeTempSan, HintStot_h[i])*.5)
mS_an = RectBivariateSpline(rangeTempSan, HintStot_h, datstot_an, kx=1, ky=1)

#########################  Temperature
# Temperature Entropy Field Cooling
minStot_c = np.round(np.min(datstot_c[1:, 1:]))
maxStot_c = np.round(np.max(datstot_c[1:, 1:]))
nPoint = 4000
rangeStot_c = np.linspace(minStot_c, maxStot_c, num=nPoint + 1)
nTemp_c = np.zeros([np.size(rangeStot_c), np.size(HintStot_c)])
for field in range(np.size(HintStot_c)):
    sSet_c = np.ndarray.flatten(mS_c(TempStot_c, HintStot_c[field]))
    TempField_c = interp1d(sSet_c, TempStot_c, kind='linear', bounds_error=False, fill_value='extrapolate')
    nTemp_c[:, field] = TempField_c(rangeStot_c)
mTemp_c = RectBivariateSpline(rangeStot_c, HintStot_c, nTemp_c, kx=1, ky=1)
# Temperature Entropy Field Heating
minStot_h = np.round(np.min(datstot_h[1:, 1:]))
maxStot_h = np.round(np.max(datstot_h[1:, 1:]))
nPoint = 4000
rangeStot_h = np.linspace(minStot_h, maxStot_h, num=nPoint + 1)
nTemp_h = np.zeros([np.size(rangeStot_h), np.size(HintStot_h)])
for field in range(np.size(HintStot_h)):
    sSet_h = np.ndarray.flatten(mS_h(TempStot_h, HintStot_h[field]))
    TempField_h = interp1d(sSet_h, TempStot_h, kind='linear', bounds_error=False, fill_value='extrapolate')
    nTemp_h[:, field] = TempField_h(rangeStot_h)
mTemp_h = RectBivariateSpline(rangeStot_h, HintStot_h, nTemp_h, kx=1, ky=1)
# Temperature Entropy Field  Anhysteretic
minStot_an = np.round(np.min(datstot_an))
maxStot_an = np.round(np.max(datstot_an))
nPoint = 4000
rangeStot_an = np.linspace(minStot_an, maxStot_an, num=nPoint + 1)
nTemp_an = np.zeros([np.size(rangeStot_an), np.size(HintStot_h)])
for field in range(np.size(HintStot_h)):
    sSet_an = np.ndarray.flatten(mS_an(rangeTemp, HintStot_h[field]))
    TempField_an = interp1d(sSet_an, rangeTemp, kind='linear', bounds_error=False, fill_value='extrapolate')
    nTemp_an[:, field] = TempField_an(rangeStot_an)
mTemp_an = RectBivariateSpline(rangeStot_an, HintStot_h, nTemp_an, kx=1, ky=1)