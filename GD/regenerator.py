# -*- coding: utf-8 -*-
"""
Created on Sat Mar 25 2017

@author: Theo Christiaanse

"""
# numba - this is for speed
import numba as nb
from numba import jit, f8, int32,b1
# Numpy library
import numpy as np
# Interpolation Functions
from scipy.interpolate import RectBivariateSpline, interp1d, interp2d
# Plotting
import matplotlib.pyplot as plt
# Importing Standard Libraries
import sys
import os
import time


t2 = time.time()
########################################## Applied Field Data ####################################
hapl = np.loadtxt('sourcefiles/muhapl.txt')
Rotation = hapl[0, 1:]
xPosition = hapl[1:, 0]
# Build interpolation function
appliedField = RectBivariateSpline(xPosition, Rotation, hapl[1:, 1:], ky=1, kx=1)

################################# Gd Material properties  ####################################

## Data loading of Material data Gd
# Specifc Heat
gdcp = np.loadtxt('sourcefiles/mft/gdcp_py.txt')
HintCp = gdcp[0, 1:]
TempCp = gdcp[1:, 0]
# Build interpolation function
mCp = RectBivariateSpline(TempCp, HintCp, gdcp[1:, 1:], ky=1, kx=1)

# Magnetization
gdmag = np.loadtxt('sourcefiles/mft/gdmag_py.txt')
HintMag = gdmag[0, 1:]
TempMag = gdmag[1:, 0]
# Build interpolation function
mMag = RectBivariateSpline(TempMag, HintMag, gdmag[1:, 1:], ky=1, kx=1)

# Temperature Field Entropy for Gd (look up table)
gdstot = np.loadtxt('sourcefiles/mft/gdstot_py.txt')
HintStot = gdstot[0, 1:]
TempStot = gdstot[1:, 0]
# Build interpolation function
mS = RectBivariateSpline(TempStot, HintStot, gdstot[1:, 1:], kx=1, ky=1)

# Entropy Field Temperature for Gd (build from GdStot)
minStot = np.round(np.min(gdstot[1:, 1:]))
maxStot = np.round(np.max(gdstot[1:, 1:]))
nPoint = 4000
rangeStot = np.linspace(minStot, maxStot, num=nPoint + 1)

nTemp = np.zeros([np.size(rangeStot), np.size(HintStot)])
for field in range(np.size(HintStot)):
    sSet = np.ndarray.flatten(mS(TempStot, HintStot[field]))
    TempField = interp1d(sSet, TempStot, kind='linear', bounds_error=False, fill_value='extrapolate')
    nTemp[:, field] = TempField(rangeStot)

mTemp = RectBivariateSpline(rangeStot, HintStot, nTemp, kx=1, ky=1)
t3 = time.time()


################################# Fluid properties  ####################################
# 80-20 water-glycol properties are calculated based on the following paper:
# D. Bohne, S. Fischer, and E. Obermeier, “Thermal Conductivity, Density, Viscosity, and
# Prandtl-Numbers of Ethylene Glycol-Water Mixtures.,” pp. 739–742.

# Fluid Specific Heat
cpfluid = np.loadtxt('sourcefiles/fluid/cp_fluid.txt')
fCp = interp1d(cpfluid[:,0],cpfluid[:,1])
# Fluid Conduction Coefficient
confluid = np.loadtxt('sourcefiles/fluid/cond_fluid.txt')
fK = interp1d(confluid[:,0],confluid[:,1])
# Fluid Density
rhofluid = np.loadtxt('sourcefiles/fluid/rho_fluid.txt')
fRho = interp1d(rhofluid[:,0],rhofluid[:,1])
# Fluid Dynamic Viscosity
visfluid = np.loadtxt('sourcefiles/fluid/vis_fluid.txt')
fMu = interp1d(visfluid[:,0],visfluid[:,1])

############################################## SOLVER ##############################################
############################################## TDMA   ##############################################
@jit(f8[:] (f8[:],f8[:],f8[:],f8[:] ),nopython=True)
def TDMAsolver(a, b, c, d):
    '''
    TDMA solver, a b c d can be NumPy array type or Python list type.
    refer to http://en.wikipedia.org/wiki/Tridiagonal_matrix_algorithm
    and to http://www.cfd-online.com/Wiki/Tridiagonal_matrix_algorithm_-_TDMA_(Thomas_algorithm)
    and to https://gist.github.com/TheoChristiaanse/d168b7e57dd30342a81aa1dc4eb3e469
    '''
    nf = len(d)  # number of equations
    ac = np.copy(a)
    bc = np.copy(b)
    cc = np.copy(c)
    dc = np.copy(d)
    for it in range(1, nf):
        mc = ac[it - 1] / bc[it - 1]
        bc[it] = bc[it] - mc * cc[it - 1]
        dc[it] = dc[it] - mc * dc[it - 1]
    xc = bc
    xc[-1] = dc[-1] / bc[-1]

    for il in range(nf - 2, -1, -1):
        xc[il] = (dc[il] - cc[il] * xc[il + 1]) / bc[il]

    return xc


######################################## UPWINDING SCHEME #########################################

@jit(f8(f8),nopython=True)
def alpha_pow(Pe):
    # Powerlaw
    val = (1-0.1*abs(Pe))**5
    return max(0,val)



######################################## SOLID SOLVER ##############################################
@jit(f8[:]     (f8[:],  f8[:], f8[:], f8[:], f8, f8[:], f8[:], f8[:],  f8[:], f8[:], f8,   f8, int32, f8, f8),nopython=True)
def SolveSolid(iynext, isnext, yprev, sprev, Vd,    Cs,   Kse,   Ksw, Omegas,  Smce, CS, CMCE,     N, dx, dt):
    '''
    Solve the NTU/U solid matrix.
    See notebook "Theo Christiaanse 2017" pg. 92-97

    pg. 109
    Cs dy/dt-d/dx(Ks*dy/dx)=Smce+Omegas*(y-s)
    '''
    # Prepare material properties
    a = np.zeros(N-1)
    b = np.zeros(N-1)
    c = np.zeros(N-1)
    d = np.zeros(N-1)
    snext = np.zeros(N+1)
    if Vd>=0:
        # Add a value at the start
        # Add bc, value to the start of the flow
        for j in range(N-1): # note range will give numbers from 0->N-2
            # Build tridagonal matrix coefficients
            # pg. 115 Theo Christiaanse 2017
            a[j] = CS*Ksw[j]/(dx*2)
            b[j] = Cs[j+1]/dt-CS*Ksw[j]/(dx*2)-CS*Kse[j]/(dx*2)+Omegas[j+1]/2
            c[j] = CS*Kse[j]/(dx*2)
            d[j] = (iynext[j]+iynext[j+1])*Omegas[j+1]/2+ sprev[j]*(-CS*Ksw[j]/(2*dx)) + sprev[j+1]*(Cs[j+1]/dt-Omegas[j+1]/2+CS*Ksw[j]/(2*dx)+CS*Kse[j]/(2*dx)) + sprev[j+2]*(-CS*Kse[j]/(2*dx)) + CMCE*Smce[j+1]
        # Add in BC
        # Neumann @ i=1
        b[0] = b[0]  + a[0]
        # Neumann @ i=-2
        b[-1]= b[-1] + c[-1]
        # Solve the unknown matrix 1-> N-1
        snext[1:-1] = TDMAsolver(a[1:], b, c[:-1], d)
        # Ghost node on either side of the Solid boundary
        snext[-1] = snext[-2]
        snext[0]  = snext[1]
        return snext
    else:
        # Add a value at the end
        for j in range(N-1):  # This will loop through 0 to N-1 which aligns with 1->N-1
            # Build tridagonal matrix coefficients
            # pg. 115 Theo Christiaanse 2017
            a[j] = CS*Ksw[j]/(dx*2)
            b[j] = Cs[j+1]/dt-CS*Ksw[j]/(dx*2)-CS*Kse[j]/(dx*2)+Omegas[j+1]/2
            c[j] = CS*Kse[j]/(dx*2)
            d[j] = (iynext[j+1]+iynext[j+2])*Omegas[j+1]/2+sprev[j]*(-CS*Ksw[j]/(2*dx)) + sprev[j+1]*(Cs[j+1]/dt+CS*Ksw[j]/(2*dx)+CS*Kse[j]/(2*dx)-Omegas[j+1]/2) + sprev[j+2]*(-CS*Kse[j]/(2*dx)) + CMCE*Smce[j+1]
        # Add in BC
        # Neumann @ i=0
        b[0] = b[0]  + a[0]
        # Neumann @ i=-1
        b[-1]= b[-1] + c[-1]
        # Solve the unknown matrix 1-> N-1
        snext[1:-1] = TDMAsolver(a[1:], b, c[:-1], d)
        # Ghost node on either side of the Solid boundary
        snext[-1] = snext[-2]
        snext[0]  = snext[1]
        return snext




######################################## FLUID SOLVER ##############################################
@jit(f8[:]     (f8[:], f8[:],f8[:],f8[:],f8, f8[:], f8[:], f8[:],  f8[:],  f8[:], f8[:], f8[:], f8, f8, f8,int32,f8,f8,f8[:]),nopython=True)
def SolveFluid(iynext,isnext,yprev,sprev,Vd,    Cf,   Kfe,   Kfw,     Ff, Omegaf,    Lf,    Sp, CF, CL,CVD,    N,dx,dt, yamb):
    '''
    Solve the NTU/U Fluid matrix.
    See notebook "Theo Christiaanse 2017" pg. 92-97

    pg. 109
    Cf dy/dt+d/dx(Ff*y)-d/dx(Kf*dy/dx)=Sp+Lf(yamb-y)+Omegaf*(s-y)
    '''
    # Prepare material properties
    a = np.zeros(N-1)
    b = np.zeros(N-1)
    c = np.zeros(N-1)
    d = np.zeros(N-1)
    ynext = np.zeros(N+1)
    if Vd>=0:
        # Add a value at the start
        # Add bc, value to the start of the flow
        # Dirichlet ghost node
        ynext[0]=0
        for j in range(N-1):  # This will loop through 1 to N+1 which aligns with 0->N
            # Build tridagonal matrix coefficients
            # pg 112-113 Theo Christiaanse 2017
            Aw=alpha_pow(2*Ff[j]/Kfw[j])
            Ae=alpha_pow(2*Ff[j+1]/Kfe[j])
            a[j] = -Ff[j]/(dx)+Ae*CF*Kfw[j]/(dx*2)+Omegaf[j+1]/2
            b[j] = Cf[j+1]/(dt)-Aw*CF*Kfw[j]/(2*dx)-Ae*CF*Kfe[j]/(2*dx)+CL*Lf[j+1]/2+Omegaf[j+1]/2+Ff[j+1]/(dx)
            c[j] = Ae*CF*Kfe[j]/(dx*2)
            d[j] = yprev[j]*(-Aw*CF*Kfw[j]/(2*dx)) + yprev[j+1]*(Cf[j+1]/dt+Aw*CF*Kfw[j]/(dx*2)+Ae*CF*Kfe[j]/(dx*2) - CL*Lf[j+1]/2) + yprev[j+2]*(-Ae*CF*Kfe[j]/(dx*2)) + yamb[j+1]*(CL*Lf[j+1])+isnext[j+1]*(Omegaf[j+1]/2)+sprev[j+1]*(Omegaf[j+1]/2)+CVD*Sp[j+1]
        # Add in bc
        # Dirichlet @ i=0
        d[0] = d[0]  - a[0]*ynext[0]
        # Neumann @ i=-1
        b[-1]= b[-1] + c[-1]
        # Solve the unknown matrix 1-> N-1
        ynext[1:-1] = TDMAsolver(a[1:], b, c[:-1], d)
        # d\dx=0 ghost node.
        ynext[-1] = ynext[-2]
        return ynext
    else:
        # Add a value at the end
        ynext[-1]=1
        for j in range(N-1):  # This will loop through 1 to N+1 which aligns with 0->N
            # Build tridagonal matrix coefficients
            # pg 112-113 Theo Christiaanse 2017
            Aw=alpha_pow(2*Ff[j+1]/Kfw[j])
            Ae=alpha_pow(2*Ff[j+2]/Kfe[j])
            a[j] = Aw*CF*Kfw[j]/(dx*2)
            b[j] = Cf[j+1]/dt-Aw*CF*Kfw[j]/(2*dx)-Ae*CF*Kfe[j]/(2*dx)+CL*Lf[j+1]/2+Omegaf[j+1]/2-Ff[j+1]/(dx)
            c[j] = Ff[j+2]/(dx)+Ae*CF*Kfe[j]/(dx*2)+Omegaf[j+1]/2
            d[j] = yprev[j]*(-Aw*CF*Kfw[j]/(dx*2)) + yprev[j+1]*(Cf[j+1]/dt+Aw*CF*Kfw[j]/(dx*2)+Ae*CF*Kfe[j]/(dx*2)-CL*Lf[j+1]/2) + yprev[j+2]*(-Ae*CF*Kfe[j]/(dx*2)) + yamb[j+1]*(CL*Lf[j+1]) + isnext[j+1]*(Omegaf[j+1]/2) + sprev[j+1]*(Omegaf[j+1]/2) + CVD*Sp[j+1]
        # Add in bc
        # Dirichlet @ i=0
        d[-1] = d[-1] - c[-1] * ynext[-1]
        # Neumann @ i=-1
        b[0]  = b[0]  + a[0]
        # Solve the unknown matrix 0->N-1
        ynext[1:-1] = TDMAsolver(a[1:], b, c[:-1], d)
        # d\dx=0 ghost node.
        ynext[0]=ynext[1]
        return ynext

#################################### CLOSURE RELATIONSHIPS ######################################

# Dynamic conduction in the fluid based on Paulo's work
@jit(f8 (   f8, f8,  f8, f8,   f8, f8),nopython=True)
def kDyn_P(Dsp, er, fCp, fK, fRho, Ud):
    PeNum = fCp / fK * np.abs(Ud) * Dsp * fRho
    if PeNum < 0.01 :
        kd = fK + fRho ** 2 * fCp ** 2 / fK * np.sqrt(0.2e1) * (np.abs(Ud)**2) / (er** 2) * Dsp**2 * ((1 - er) ** (-0.1e1 / 0.2e1)) / 0.240e3
    else:
        kd = fK + 0.375 * fRho * fCp * np.abs(Ud) * Dsp
    return kd

# Beta*Heff based on the work of Iman.
@jit(f8     (f8 , f8,  f8, f8,  f8,   f8,   f8,  f8, f8,   f8, f8),nopython=True)
def beHeff_I(Dsp, Ud, fCp, fK, fMu, fRho, freq, mCp, mK, mRho, er):
    # Iman uses the DF factor, and a wacou Nu.
    hefff = (2 + 0.11e1 * (fRho * Ud * Dsp / fMu) ** 0.6e0 * (fMu * fCp / fK) ** (0.1e1 / 0.3e1)) * fK / Dsp / (1 + (2 + 0.11e1 * (fRho * Ud * Dsp / fMu) ** 0.6e0 * (fMu * fCp / fK) ** (0.1e1 / 0.3e1)) * fK / mK * (1 - 0.1e1 / mK * mRho * mCp * freq * Dsp ** 2 / 35) / 10)
    beta  = 6 * (1 - er) / Dsp
    return hefff*beta

# Static conduction Kaviani
@jit(f8(  f8, f8, f8),nopython=True)
def kStat(er, fK, mK):
    return fK * ((1 - 10 ** (0.935844e0 - 0.6778e1 * er)) * (er * (0.8e0 + 0.1e0 * er) + (-er * (0.8e0 + 0.1e0 * er) + 1) * mK / fK) / (1 - er * (0.2e0 - 0.1e0 * er) + mK / fK * er * (0.2e0 - 0.1e0 * er)) + 10 ** (0.935844e0 - 0.6778e1 * er) * (2 * mK ** 2 / fK ** 2 * (1 - er) + (1 + 2 * er) * mK / fK) / ((2 + er) * mK / fK + 1 - er))

# This the pressure drop term Ergun relation
@jit(nb.types.Tuple((f8, f8))(f8, f8, f8, f8, f8,  f8,  f8),nopython=True)
def SPresM(Dsp, Ud, V, er, flMu, flRho, Af):
    #Nield
    dP = (1.75 * Ud ** 2 * (1 - er) / Dsp / er ** 3 * flRho + 150 * Ud * (1 - er) ** 2 / Dsp ** 2 / er ** 3 * flMu)
    Spress = dP * V
    return Spress, dP

# This calculates the thermal resistance for the regenerator.
@jit(f8(              f8,  f8,  f8,   f8,   f8, f8,   f8, f8, f8, f8),nopython=True)
def ThermalResistance(Dsp, Ud, fMu, fRho, kair, kf, kg10, r1, r2, r3):
    return 0.1e1 / (0.5882352941e1 * (fRho * np.abs(Ud) * Dsp / fMu) ** (-0.79e0) / kf * Dsp + 0.1e1 / kg10 * r1 * np.log(r2 / r1) + 0.1e1 / kair * r1 * np.log(r3 / r2))

# This calculates the thermal resistance for the void space.
@jit(f8(                  f8,  f8,  f8,   f8,   f8, f8, f8, f8),nopython=True)
def ThermalResistanceVoid(kair, kf, kg10, kult, r0, r1, r2, r3):
    return 0.1e1 / (0.4587155963e0 / kf * r0 + 0.1e1 / kult * r0 * np.log(r1 / r0) + 0.1e1 / kg10 * r0 * np.log(r2 / r1) + 0.1e1 / kair * r0 * np.log(r3 / r2))

####################################### LOOP FUNC ##############################################
@jit(nb.types.Tuple((b1,f8))(f8[:],f8[:],f8))
def AbsTolFunc(var1,var2,Tol):
    maximum_val=np.max(np.abs(var1-var2))
    return np.all(maximum_val<Tol),maximum_val

@jit(nb.types.Tuple((b1,f8))(f8[:,:],f8[:,:],f8))
def AbsTolFunc2d(var1,var2,Tol):
    maximum_val=np.max(np.abs(var1-var2))
    return np.all(maximum_val<Tol),maximum_val


######################################### RUN ACTIVE ############################################

def runActive(caseNum,Thot,Tcold,cen_loc,Tambset,dispV,ff,CF,CS,CL,CVD,CMCE,nodes,timesteps,Dsp):
    '''
    # runActive : Runs a AMR simulation of a pre-setup geometry
    # Arguments :
    # caseNum    <- caseNum number
    # Thot       <- Hot side heat exchanger
    # Tcold      <- Cold side heat exchanger
    # cen_loc    <- offset of the regenerator to the magnet
    # Tambset    <- Set ambient temperature
    # numid      <- Experimental id
    # startprev  <- Do you wish to reload the previous data to start the new cycle
    #               Only select when a previous file is available.
    #               You can do so by making sure to save the solid and fluid temperature matrix in the
    #               multi_sweep file
    # dispV      <- Displaced volume [m^3]
    # ff         <- frequency [Hz]
    # CF         <- Enable/Disable Conduction term in Fluid
    # CS         <- Enable/Disable Conduction term in Solid
    # CL         <- Enable/Disable Heat leaks term in the Fluid GE
    # CVD        <- Enable/Disable Viscous Dissipation Term in the Fluid GE
    # CMCE       <- Enable/Disable MCE
    # nodes      <- Number of Spacial nodes used
    # timesteps  <- Number of Timesteps per cycle
    # Dsp        <- Diameter of the Spheres

    '''
    print("Hot Side: {} Cold Side: {}".format(Thot,Tcold))


    # Start Timer
    t0 = time.time()

    ##### Most of the variable here till row nr 180 are global.
    ##### This means we can refer to them quite easily.


    # Number of spatial nodes
    N = nodes
    # Number of time steps
    nt = timesteps

    # Volume displacement in [m3]
    # Small displacer 2.53cm^2
    # Medium displacer 6.95cm^2
    # 1inch = 2.54cm
    Vd      = dispV
    freq    = ff
    tau_c   = 1/freq


    # Ambiant Temperature non-diamentionilized
    yamb = np.ones(N + 1) * ((Tambset - Tcold) / (Thot - Tcold))

    # information of the assembly

    r1      = 16e-3 / 2  # Inner Radius [m]
    r2      = 19.05e-3 / 2  # Outer Radius [m]
    r3      = 22e-3 / 2  # Bore Radius [m]

    Vvoid1  = 1.04e-6 # [m^3] 1 [cm^3]
    Vvoid2  = 6.55e-7 # [m^3]
    Lvoid1  = 0.139-0.110 #[m]
    Lvoid2  = 0.202-0.163 #[m]
    #
    L_add   = 18e-3
    Lvoid   = 15e-3 + L_add
    rv      = 1.9e-3  # Cold side void to the check valve radius [m]
    Vvoid   = 460e-9  +  rv**2*3.14*L_add # [m^3]
    rvs     = np.sqrt(Vvoid/(3.14*Lvoid))  # Cold side void radius [m]
    rvs1    = np.sqrt(Vvoid1/(3.14*Lvoid1))  # Hot  side void radius [m]
    rvs2    = np.sqrt(Vvoid2/(3.14*Lvoid2))  # Hot  side void radius [m]
    print(" R_void = {} \n R_void1={} \n R_void2={}".format(rvs,rvs1,rvs2))

    # You can build any geometry based on glass spheres, regenerator(in this case Gd), and void space
    species_discription = ['void','gs', 'reg', 'gs', 'void1', 'gs', 'void2']
    print("the geometry looks like: {}".format(species_discription))
    L_reg = 0.0229
    # Locations 0<- cold hex, hot hex -> end
    x_discription = [0, 0.015+L_add, 0.061+L_add, 0.061+L_add+L_reg, 0.110+L_add, 0.139+L_add, 0.163+L_add, 0.202+L_add] #[m]


    # Location of the regenerator with respect to the Halbach magnet center
    cen_loc = 0
    # Surface Area
    # Regenerator and glass spheres
    Ac      = np.pi * r1 ** 2
    # Regenerator and glass spheres
    Pc      = np.pi * r1 * 2
    # Molecule size
    dx = 1 / (N-1)
    # The time step is:
    dt = 1 / (nt+1)

    # To prevent the simulation running forever we limit the simulation to
    # find a tollerance setting per step and per cycle.
    maxStepTol  = [1e-2,1e-3,1e-4,1e-5,1e-6,1e-7]
    maxCycleTol = 1e-6
    # We also limit the number of steps and cycles the simulation can take. If
    # the simulation runs over these there might be an issue with the code.
    maxSteps  = 1000
    maxCycles = 2000

    # NTU and U analysis is based on NTU which flips halfway.
    # Cycle period

    t = np.linspace(0, tau_c, nt+1)

    # Build darcy velocity for all n steps
    # Sine wave (See notebook pg. 33 Theo Christiaanse Sept 2017)
    vf = lambda at, Ac, Vd, sf: (Vd) * sf * np.pi * np.sin(2 * np.pi * sf * at) + np.sign(np.sin(2 * np.pi * sf * at))*sys.float_info.epsilon*2
    # block wave (See notebook pg. 111 Theo Christiaanse Sept 2017)
    #uf = lambda at, Ac, Vd, sf: Vd*sf * np.sign(1/(sf*2)-sys.float_info.epsilon-at)


    # Build all volumetric fluid speeds V[n] (m^3/s) dt time unit
    V = vf(t, Ac, Vd, freq)

    # Total length of the domain
    L_tot   = np.max(x_discription)  # [m]

    # Real element size
    # Element size
    DX = L_tot/ (N-1)
    # Time step
    DT = tau_c/(nt+1)


    # Set Material & Fluid Properties
    # (BOROSILICATE) GLASS spheres
    gsCp  = 800.  # [J/(kg K)]
    gsRho = 2230.  # [kg/m^3]
    gsK   = 1.2  # [W/(m K)]
    # http://www.scientificglass.co.uk/contents/en-uk/d115_Physical_Properties_of_Borosilicate_Glass.html
    # http://www.schott.com/borofloat/english/attribute/thermic/index.html
    # http://www.schott.com/d/tubing/9a0f5126-6e35-43bd-bf2a-349912caf9f2/schott-algae-brochure-borosilicate.pdf

    # Ultem
    kult  =  0.122   # [W/(m K)]
    # https://www.plasticsintl.com/datasheets/ULTEM_GF30.pdf
    # g10 material
    kg10  =  0.608  # [W/(m K)]
    # http://cryogenics.nist.gov/MPropsMAY/G-10%20CR%20Fiberglass%20Epoxy/G10CRFiberglassEpoxy_rev.htm
    # air material
    kair  = 0.0255  # [W/(m K)]
    # Transport booklet
    ## Build material property functions
    mRho  = 7900.  # [kg/m^3]
    mK    = 10.5  # [W/(m K)]

    # Porosity of regenerator
    er      = 0.36  # [-]
    # Porosity of glass spheres (measured)
    egs     = 0.41  # [-]
    # Diameter of regenerator
    # Set as imput variable!
    #Dsp     = Dsp  # [m]
    # Diameter of glass spheres
    Dspgs   = 0.003125  # [m]
    # Demagnetization coefficient
    Nd      = 0.368  # [-]

    # This prints out the void space on the left and on the right.
    V_0c = Vvoid+egs*(0.061-0.015)*Ac
    V_0h = Vvoid2+Vvoid1+egs*(0.110-0.061-L_reg)*Ac+egs*(0.163-0.139)*Ac
    print(" V_0c/V_0 = {:2.2f} \n V_0c/Vd={:2.2f}".format(V_0c/(V_0c+V_0h),V_0c/dispV))

    # Calculate the utilization as defined by Armando's paper
    Uti     = (Vd * 1000 * 4200) / (350 * Ac * (1 - er) * 7900 * L_reg)
    print('Utilization: {0:1.3f} Frequency: {1:1.2f}'.format(Uti,freq))
    print('Urms: {0:3.3f}'.format((Vd / Ac*er) * freq * np.pi*1/np.sqrt(2)))


    ## Field settings for PM1
    nn = 0
    # We need to distribute the space identifiers along a matrix to use it later.
    # This function cycles through x_discription until it finds a new domain then sets according
    # to the int_discription
    int_discription = np.zeros(N+1,dtype=np.int)
    species_descriptor = []
    xloc            = np.zeros(N+1)
    # Set the rest of the nodes to id with geoDis(cription)
    for i in range(N+1): # sets 0->N
        xloc[i] = (DX * i - DX / 2)  #modify i so 0->N
        if (xloc[i] >= x_discription[nn + 1]):
            nn = nn + 1
        int_discription[i] = nn
        species_descriptor.append(species_discription[nn])

    ## Set Surface area and Porosity of solid and fluid
    A_c = np.ones(N + 1)
    e_r = np.ones(N + 1)
    P_c = np.ones(N + 1)
    for i in range(N+1):
        if species_descriptor[i]=='reg':
            A_c[i] = Ac
            e_r[i] = er
            P_c[i] = Pc
        elif species_descriptor[i]== 'gs':
            A_c[i] = Ac
            e_r[i] = egs
            P_c[i] = Pc
        else:
            if species_descriptor[i]=='void':
                A_c[i] = rvs**2 * np.pi
                e_r[i] = 1
                P_c[i] = 2*rvs*np.pi
            if species_descriptor[i]=='void1':
                A_c[i] = rvs1**2 * np.pi
                e_r[i] = 1
                P_c[i] = 2*rvs1*np.pi
            if species_descriptor[i]=='void2':
                A_c[i] = rvs2**2 *np.pi
                e_r[i] = 1
                P_c[i] = 2*rvs2 *np.pi


    # This is the domain fraction, determining the algebraic
    # split between domains. It will ensure the variation of
    # conduction between domains is taken into account.
    # By default it will be 0.5 if domain species does not change.
    # Please review  notebook pg. 116-117
    fr = np.ones(N) * 0.5
    nn = 1
    for i in range(N):
        # If we are between boundaries fr will be set to 0.5
        if (xloc[i] < x_discription[nn] and x_discription[nn] < xloc[i+1]):
            fr[i] = (xloc[i+1]-x_discription[nn])/DX
            nn    = nn + 1

    ############################# BUILD appliedField array ####################
    # Shift the rotation of the magnet so it aligns with the sin time
    RotMag = lambda t, f: 360 * t * f + 270 - 360 * np.floor(t * f + 270 / 360)
    # Build all rotMag[n]
    rotMag = np.copy(RotMag(t, freq))
    # Basically this array will contain all field values for each time and spatial node.
    appliedFieldm = np.ones((nt+1, N + 1))
    # Distance between base of the cold heat exchanger to magnet center
    magOffset = cen_loc - L_reg/2
    first_node_val_with_reg  = np.min(np.argwhere([val=='reg' for val in species_descriptor]))
    for i in range(N + 1):
        for n in range(0, nt+1):
            # Will only get the field if we find a regenerator
            if (species_descriptor[i] == 'reg'):
                x_pos_w_respect_to_magnet = xloc[i-first_node_val_with_reg] + magOffset
                appliedFieldm[n, i] = appliedField(x_pos_w_respect_to_magnet, rotMag[n])[0, 0]*CMCE
            else:
                appliedFieldm[n, i] = 0

    ########################## START THE AMR CYCLE #########################

    # Some housekeeping to make this looping work
    #
    # Initial temperature
    y1 = np.linspace(0,1, N + 1)
    y = np.ones((nt+1, N + 1))*y1
    s1 = np.linspace(0, 1, N + 1)
    s = np.ones((nt+1, N + 1))*s1
    # Magnetic Field Modifier (this is known as R(z) in the modelling paper)
    MFM = np.ones(N + 1)
    #
    cycleTol   = 0
    cycleCount = 1
    stepTolInt = 0
    #
    # Initial guess of the cycle values.
    iyCycle = np.copy(y)
    isCycle = np.copy(s)
    while (not cycleTol  and cycleCount <= maxCycles):
        # Account for pressure every time step (restart every cycle)
        pt = np.zeros(nt + 1)
        #
        minPrevHint = 0.5
        maxPrevHint = 0.5
        maxAplField = 0.5
        minAplField = 0.5
        maxMagTemp = Tcold
        minMagTemp = Thot
        maxCpPrev = 0
        minCpPrev = 3000
        maxSSprev = 0
        minSSprev = 3000
        #
        maxTemp = Tcold
        minTemp = Thot

        # Vacuum permeability constant
        mu0     = 4 * 3.14e-7  # [Hm^-1]

        for i in range(N+1):
            # Average Solid temperature
            Ts_ave=np.mean(s[:,i]* (Thot - Tcold) + Tcold)
            # Maximum Applied Field
            maxApliedField = np.amax(appliedFieldm[:,i])
            if maxApliedField==0:
                MFM[i] = 0
            else:
                # Maximum Magnetization at the maximum field
                maxMagLoc=mMag(Ts_ave,maxApliedField)[0, 0]
                # The resulting internal field
                Hint = maxApliedField - mRho * Nd * maxMagLoc * mu0
                # The decrease ratio of the applied field
                MFM[i] = Hint/maxApliedField
        for n in range(1, nt+1):  # 1->nt
            # Run every timestep
            # Initial
            stepTol = 0
            stepCount = 1
            # Initial guess of the current step values.
            iynext  = np.copy(y[n-1, :])
            isnext  = np.copy(s[n-1, :])
            # current and previous temperature in [K]
            pfT     = y[n-1, :]  * (Thot - Tcold) + Tcold
            psT     = s[n-1, :]  * (Thot - Tcold) + Tcold

            if max(pfT)>maxTemp:
                maxTemp = max(pfT)
            if min(pfT)<minTemp:
                minTemp = min(pfT)
            cpf_prev  = np.zeros(N+1)
            rhof_prev = np.zeros(N+1)
            muf_prev  = np.zeros(N+1)
            kf_prev   = np.zeros(N+1)
            cps_prev  = np.zeros(N+1)
            Ss_prev   = np.zeros(N+1)



            for i in range(N+1):
                if species_descriptor[i]=='reg':
                    # Internal field
                    prevHint = appliedFieldm[n-1,i]*MFM[i]
                    # previous specific heat
                    cps_prev[i]    = mCp(psT[i], prevHint)[0, 0]
                    # Entropy position of the previous value
                    Ss_prev[i]     = mS(psT[i], prevHint)[0, 0]
                    if prevHint > maxPrevHint:
                        maxPrevHint = prevHint
                        maxAplField = appliedFieldm[n-1,i]
                        maxMagTemp  = psT[i]
                        maxCpPrev   = cps_prev[i]
                        maxSSprev   = Ss_prev[i]
                    if prevHint < minPrevHint:
                        minPrevHint = prevHint
                        minAplField = appliedFieldm[n-1,i]
                        minMagTemp  = psT[i]
                        minCpPrev   = cps_prev[i]
                        minSSprev   = Ss_prev[i]
                elif species_descriptor[i]== 'gs':
                    cps_prev[i]    = gsCp
                    Ss_prev[i]     = 0
                    # This is where the gs stuff will go
                else:
                    cps_prev[i]    = 0
                    Ss_prev[i]     = 0
                    # This is where the void stuff will go
                # Calculate Specific heat
                cpf_prev[i]    = fCp(pfT[i],percGly)
                # Calculate Density
                rhof_prev[i]   = fRho(pfT[i],percGly)
                # Calculate Dynamic Viscosity
                muf_prev[i]    = fMu(pfT[i],percGly)
                # Calculate Conduction
                kf_prev[i]     = fK(pfT[i],percGly)

            # Loop untill stepTol is found or maxSteps is hit.
            while ( not stepTol and stepCount <= maxSteps):

                # iynext is the guess n Fluid
                # isnext is the guess n Solid
                # y[n-1,:] is the n-1 Fluid solution
                # s[n-1,:] is the n-1 Solid solution
                ################################################################
                # Grab Current State properties
                fT = iynext * (Thot - Tcold) + Tcold
                sT = isnext * (Thot - Tcold) + Tcold
                Cs = np.zeros(N + 1)
                ks = np.zeros(N + 1)
                Smce = np.zeros(N + 1)
                k = np.zeros(N + 1)
                Omegaf = np.zeros(N + 1)
                Spres = np.zeros(N + 1)
                Lf = np.zeros(N + 1)

                Cf = np.zeros(N + 1)
                rhof_cf_ave = np.zeros(N + 1)
                rhos_cs_ave = np.zeros(N + 1)
                Ff = np.zeros(N + 1)
                Sp = np.zeros(N + 1)
                Kfw = np.zeros(N - 1)
                Kfe = np.zeros(N - 1)
                Ksw = np.zeros(N - 1)
                Kse = np.zeros(N - 1)
                # Weighted guess value
                ww = 0.5
                # Int the pressure
                pt[n] = 0
                dP = 0
                for i in range(N + 1):
                    # Calculate Specific heat fluid
                    cpf_ave  = fCp(fT[i], percGly) * ww + cpf_prev[i] * (1 - ww)
                    # Calculate Density fluid
                    rhof_ave = fRho(fT[i], percGly) * ww + rhof_prev[i] * (1 - ww)
                    # Calculate Dynamic Viscosity fluid
                    muf_ave  = fMu(fT[i], percGly) * ww + muf_prev[i] * (1 - ww)
                    # Calculate Conduction fluid
                    kf_ave   = fK(fT[i], percGly) * ww + kf_prev[i] * (1 - ww)
                    # Combined rhof cf
                    rhof_cf_ave[i] = cpf_ave * rhof_ave
                    if species_descriptor[i] == 'reg':
                        # Field
                        Hint = appliedFieldm[n, i]*MFM[i]
                        # rho*cs fluid
                        cps_ave = (mCp(sT[i], Hint)[0, 0]) * ww + cps_prev[i] * (1 - ww)
                        rhos_cs_ave[i] = cps_ave * mRho
                        # Effective Conduction for fluid
                        k[i] = kDyn_P(Dsp, e_r[i], cpf_ave, kf_ave, rhof_ave, np.abs(V[n] / (A_c[i])))
                        # Forced convection term east of the P node
                        Omegaf[i] = A_c[i] * beHeff_I(Dsp, np.abs(V[n] / (A_c[i])), cpf_ave, kf_ave, muf_ave, rhof_ave,
                                                      freq, cps_ave, mK, mRho, e_r[i])  # Beta Times Heff
                        # Pressure drop
                        Spres[i], dP = SPresM(Dsp, np.abs(V[n] / (A_c[i])), np.abs(V[n]), e_r[i], muf_ave, rhof_ave,
                                              A_c[i] * e_r[i])
                        # Loss term
                        Lf[i] = P_c[i] * ThermalResistance(Dsp, np.abs(V[n] / (A_c[i])), muf_ave, rhof_ave, kair, kf_ave,
                                                       kg10, r1, r2, r3)
                        # Effective Conduction for solid
                        ks[i] = kStat(e_r[i], kf_ave, mK)
                        # Smce
                        Smce[i] = (A_c[i] * (1 - e_r[i]) * rhos_cs_ave[i] * (mTemp(Ss_prev[i], Hint)[0, 0] - psT[i])) / (DT * (Thot - Tcold))
                        ### Capacitance solid
                        Cs[i] = rhos_cs_ave[i] * A_c[i] * (1 - e_r[i]) * freq
                    elif species_descriptor[i] == 'gs':
                        # This is where the gs stuff will go
                        # Effective Conduction for solid
                        rhos_cs_ave[i] = gsCp * gsRho
                        # Effective Conduction for fluid
                        k[i] = kDyn_P(Dspgs, e_r[i], cpf_ave, kf_ave, rhof_ave, np.abs(V[n] / (A_c[i])))
                        # Forced convection term east of the P node
                        Omegaf[i] = A_c[i] * beHeff_I(Dspgs, np.abs(V[n] / (A_c[i])), cpf_ave, kf_ave, muf_ave, rhof_ave,
                                                      freq, gsCp, gsK, gsRho, e_r[i])  # Beta Times Heff
                        # Pressure drop
                        Spres[i], dP = SPresM(Dspgs, np.abs(V[n] / (A_c[i])), np.abs(V[n]), e_r[i], muf_ave, rhof_ave,
                                              A_c[i] * e_r[i])
                        # Loss term
                        Lf[i] = P_c[i] * ThermalResistance(Dspgs, np.abs(V[n] / (A_c[i])), muf_ave, rhof_ave, kair, kf_ave,
                                                       kg10, r1, r2, r3)
                        # Effective Conduction for solid
                        ks[i] = kStat(e_r[i], kf_ave, gsK)
                        #Smce
                        Smce[i] = 0
                        ### Capacitance solid
                        Cs[i] = rhos_cs_ave[i] * A_c[i] * (1 - e_r[i]) * freq
                    else:
                        k[i] = kf_ave
                        if species_descriptor[i] == 'void':
                            Lf[i] = P_c[i] * ThermalResistanceVoid(kair, kf_ave, kg10, kult, rvs, r1, r2, r3)
                        elif species_descriptor[i] == 'void1':
                            Lf[i] = P_c[i] * ThermalResistanceVoid(kair, kf_ave, kg10, kult, rvs1, r1, r2, r3)
                        elif species_descriptor[i] == 'void2':
                            Lf[i] = P_c[i] * ThermalResistanceVoid(kair, kf_ave, kg10, kult, rvs2, r1, r2, r3)
                        # No solid in the void
                        ks[i] = 0
                        # No interaction between solid and fluid since there is no solid.
                        Omegaf[i] = 0 #
                        # This will just make the plots nicer by having the temperature of the solid be the fluid temperature.
                        Cs[i] = 1
                        Smce[i] = (iynext[i]-s[n-1,i])/DT
                        #neglect pressure term.
                        Spres[i]= 0
                        dP = 0
                        # This is where the void stuff will go
                    pt[n] = dP * DX + pt[n]

                ### Capacitance fluid
                Cf = rhof_cf_ave * A_c * e_r * freq

                ### Fluid term
                Ff = (rhof_cf_ave * V[n]) / L_tot
                Sp = Spres / (Thot - Tcold)

                for i in range(N - 1):
                    # Fluid Conduction term west of the P node
                    Kfw[i] = ((1 - fr[i]) / (A_c[i] * e_r[i] * k[i])
                              + (fr[i]) / (A_c[i+1] * e_r[i+1] * k[i+1])) ** -1
                    # Fluid Conduction term east of the P node
                    Kfe[i] = ((1 - fr[i+1]) / (A_c[i+1] * e_r[i+1] * k[i+1])
                              + (fr[i+1]) / (A_c[i+2] * e_r[i+2] * k[i+2])) ** -1
                    # Solid Conduction term
                    if ks[i]==0 or ks[i+1]==0:
                        Ksw[i] =0
                    else:
                        # Conduction term west of the P node
                        Ksw[i] = ((1 - fr[i]) / (A_c[i] * e_r[i] * ks[i])
                                + (fr[i]) / (A_c[i+1] * e_r[i+1] * ks[i+1])) ** -1
                    if ks[i+1]==0 or ks[i+2]==0:
                        Kse[i] =0
                    else:
                        # Conduction term east of the P node
                        Kse[i] = ((1 - fr[i+1]) / (A_c[i+1] * e_r[i+1] * ks[i+1])
                                + (fr[i+1]) / (A_c[i+2] * e_r[i+2] * ks[i+2])) ** -1
                Omegas = np.copy(Omegaf)


                ################################################################
                ####################### SOLVE FLUID EQ      ####################
                # Fluid Equation
                ynext = SolveFluid(iynext, isnext, y[n-1,:], s[n-1,:],V[n],Cf,Kfe,Kfw,Ff,Omegaf,Lf,Sp,CF,CL,CVD,N,dx,dt,yamb)
                ################################################################
                ####################### SOLVE SOLID EQ      ####################
                # Solid Equation
                snext = SolveSolid(ynext, isnext, y[n-1,:], s[n-1,:],V[n],Cs,Kse,Ksw,Omegas,Smce,CS,CMCE,N,dx,dt)
                ################################################################
                ####################### CHECK TOLLERANCE    ####################
                # Check the tolerance of the current time step
                stepTol = AbsTolFunc(ynext,iynext,maxStepTol[stepTolInt])[0] and AbsTolFunc(snext,isnext,maxStepTol[stepTolInt])[0]
                ################################################################
                ####################### DO housekeeping     ####################
                # Copy current values to new guess and current step.
                s[n, :] = np.copy(snext)
                isnext  = np.copy(snext)
                y[n, :] = np.copy(ynext)
                iynext  = np.copy(ynext)
                # Add a new step to the step count
                stepCount = stepCount + 1
                # Check if we have hit the max steps
                if (stepCount == maxSteps):
                    print("Hit max step count\n")
                    print(AbsTolFunc(ynext,iynext,maxStepTol[stepTolInt])[1])
                ################################################################
                ####################### ELSE ITERATE AGAIN  ####################
        # Check if current cycle is close to previous cycle.
        [bool_y_check,max_val_y_diff]=AbsTolFunc2d(y,iyCycle,maxCycleTol)
        [bool_s_check,max_val_s_diff]=AbsTolFunc2d(s,isCycle,maxCycleTol)
        cycleTol = bool_y_check and bool_s_check
        if (max_val_y_diff/10)<maxStepTol[stepTolInt]:
            stepTolInt = stepTolInt + 1
            if stepTolInt == 6:
                stepTolInt=5
        if cycleCount%10==1:
            # This is a print that will show how far along we are finding steady state.
            print("Case num {0:d} CycleCount {1:d} y-tol {2:2.5e} s-tol {3:2.5e} run time {4:4.1f} [min]".format(caseNum,cycleCount,max_val_y_diff,max_val_s_diff,(time.time()-t0)/60 ))
        # Copy last value to the first of the next cycle.
        s[0, :] = np.copy(s[-1, :])
        y[0, :] = np.copy(y[-1, :])
        # Add Cycle
        cycleCount = cycleCount + 1
        # Did we hit the maximum number of cycles
        if (cycleCount == maxCycles):
            print("Hit max cycle count\n")
        # Copy current cycle to the stored value
        isCycle = np.copy(s)
        iyCycle = np.copy(y)
        if np.any(np.isnan(y)):
            # Sometimes the simulation hits a nan value. We can redo this point later.
            print(y)
            break
        # End Cycle
    t1 = time.time()
    ########################## END THE CYCLE #########################
    quart=int(nt/4)
    halft=int(nt/2)
    tquat=int(nt*3/4)
    # This is a modifier so we can modify the boundary at which we calculate the
    # effectiveness
    cold_end_node  = np.min(np.argwhere([val=='reg' for val in species_descriptor])) # 0
    hot_end_node   = np.max(np.argwhere([val=='reg' for val in species_descriptor])) # -1
    eff_HB_CE = np.trapz((1-y[halft:,  cold_end_node]),x=t[halft:]) /(tau_c/2)
    eff_CB_HE = np.trapz(y[:halft+1,  hot_end_node],x=t[:halft+1])/ (tau_c/2)

    tFce = np.zeros(nt+1)
    tFhe = np.zeros(nt+1)
    yEndBlow = np.zeros(N+1)
    yHalfBlow = np.zeros(N+1)
    sEndBlow = np.zeros(N+1)
    sHalfBlow = np.zeros(N+1)

    yMaxCBlow = np.zeros(N+1)
    yMaxHBlow = np.zeros(N+1)
    sMaxCBlow = np.zeros(N+1)
    sMaxHBlow = np.zeros(N+1)

    tFce = y[:,  cold_end_node] * (Thot - Tcold) + Tcold
    tFhe = y[:, hot_end_node] * (Thot - Tcold) + Tcold

    yMaxCBlow  = y[quart,  :] * (Thot - Tcold) + Tcold
    yMaxHBlow = y[tquat, :] * (Thot - Tcold) + Tcold

    yEndBlow  = y[-1,  :] * (Thot - Tcold) + Tcold
    yHalfBlow = y[halft, :] * (Thot - Tcold) + Tcold

    sMaxCBlow  = s[quart,  :] * (Thot - Tcold) + Tcold
    sMaxHBlow = s[tquat, :] * (Thot - Tcold) + Tcold

    sEndBlow  = s[-1,  :] * (Thot - Tcold) + Tcold
    sHalfBlow = s[halft, :] * (Thot - Tcold) + Tcold
    NAMR=2
    # Calculate Gross Cooling power
    coolingpowersum=0
    startint=0
    for n in range(startint, nt):
        tF = y[n, 1] * (Thot - Tcold) + Tcold
        tF1 = y[n+1, 1] * (Thot - Tcold) + Tcold
        coolPn =  freq * fCp((tF+tF1)/2,percGly) * fRho((tF+tF1)/2,percGly) * V[n] * DT * ((tF+tF1)/2 - Tcold)
        coolingpowersum = coolingpowersum + coolPn
    qc = coolingpowersum * NAMR
    # Heating power
    coolingpowersum=0
    startint=0
    for n in range(startint, nt):
        tF = y[n, -2] * (Thot - Tcold) + Tcold
        tF1 = y[n+1, -2] * (Thot - Tcold) + Tcold
        coolPn =  freq * fCp((tF+tF1)/2,percGly) * fRho((tF+tF1)/2,percGly) * V[n] * DT * (Thot-(tF+tF1)/2)
        coolingpowersum = coolingpowersum + coolPn
    qh = coolingpowersum * NAMR

    # Net cooling power
    Kamb = 0.28
    qccor = qc - Kamb * (Tambset-Tcold)
    pave = np.trapz(pt[halft:], x=t[halft:]) / (tau_c/2)
    # Report some key debugging information
    print("{0:3.1f} {1:3.1f} {2:1.2f} {3:1.2f} Cycle Count: {4:d} Tol-y: {5:1.4e} Tol-s {6:1.4e}".format(float(Thot),float(Tcold),float(Uti),float(freq),int(cycleCount),float(max_val_y_diff),float(max_val_s_diff)))
    print('Utilization: {0:1.3f} Frequency: {1:1.2f}'.format(Uti,freq))
    print("Run time: {0:3.2f} [min]".format((t1 - t0) / 60))
    print("Hot Side: {} Cold Side: {}".format(Thot,Tcold))
    print('Effectiveness HB-CE {} CB-HE {}'.format(eff_HB_CE,eff_CB_HE))
    print('Cooling power {}'.format(qc))
    print('Corrected Cooling Power {}'.format(qccor))
    print('Pressure drop {} (kPa)'.format(pave/1000))
    # Min field values
    print('Values found at minimal field')
    print('min Applied Field {}'.format(minAplField))
    print('min Internal Field {}'.format(minPrevHint))
    print('min Magnetic Temperature {}'.format(minMagTemp))
    print('min Cp: {}'.format(minCpPrev))
    print('min SS:{}'.format(minSSprev))
    print('Lowest Temperature found in the SS cycle: {}'.format(minTemp))
    # Max field values
    print('Values found at maximum field')
    print('max Applied Field {}'.format(maxAplField))
    print('max Internal Field {}'.format(maxPrevHint))
    print('max Magnetic Temperature {}'.format(maxMagTemp))
    print('max Cp: {}'.format(maxCpPrev))
    print('max SS:{}'.format(maxSSprev))
    print('highest Temperature found in the SS cycle: {}'.format(maxTemp))

    # plt.figure(num=None,figsize=(12,12))
    # plt.title("Pressure drop")
    # #plt.plot(t, pt, 'k--')
    # plt.plot(t, tFce, 'k--', label="Tcold")
    # plt.plot(t, tFhe, 'g--', label="Thot")
    # plt.legend(loc=2)
    # plt.grid(True)
    # plt.show()

    # Return all the key outputs.
    return Thot,Tcold,qc,qccor,(t1-t0)/60,pave,eff_HB_CE,eff_CB_HE,tFce,tFhe,yHalfBlow,yEndBlow,sHalfBlow,sEndBlow,y, s, pt, np.max(pt),Uti,freq,t,xloc,yMaxCBlow,yMaxHBlow,sMaxCBlow,sMaxHBlow,qh



# Run a debug senario
# input variables: #Thot,Tcold,cen_loc,Tambset,dispV,ff,CF,CS,CL,CVD,nodes,timesteps,Dsp
print(runActive( 308,301.9,0,294,5.15e-6,2,0,0,0,0,50,100,300e-6))
