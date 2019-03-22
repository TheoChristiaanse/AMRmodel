# -*- coding: utf-8 -*-
"""
Created on Sat Mar 25 16:50:31 2017

@author: Theo
"""
# mpi4py
# from mpi4py import MPI
# multiprocessing
#from multiprocessing import Pool
#from multifuntest import sqrt
#from patankarreggd_one_void import runActive
import numba as nb
from numba import jit, f8, int32,b1
# Numpy library
import numpy as np
# Pickle Data
import pickle
# Interpolation Functions
from scipy.interpolate import RectBivariateSpline, interp1d, interp2d
# Plotting
import matplotlib.pyplot as plt
# Debugging libraries
import sys
import os
# Import time to time the simulation
import time


################################# Si Material properties  ####################################
#
# Naming convention
#
# Specific heat function
#
# mCp_c/h
#
# Magnetization
#
# mMag_c/h
#
# Entropy
#
# mS_c/h
#
# Temp
#
# mTemp_c/h


from sourcefiles.mat import si1
from sourcefiles.mat import si2
from sourcefiles.mat import si3
from sourcefiles.mat import si4
from sourcefiles.mat import si5
from sourcefiles.device import hapl



######################################## Fluid properties  ##########################################

# Density
from sourcefiles.fluid.density import fRho
# Dynamic Viscosity
from sourcefiles.fluid.dynamic_viscosity import fMu
# Specific Heat
from sourcefiles.fluid.specific_heat import fCp
# Conduction
from sourcefiles.fluid.conduction import fK

######################################## CLOSURE RELATIONSHIPS ######################################


# Dynamic conduction
from closure.dynamic_conduction import kDyn_P
# Static conduction
from closure.static_conduction import kStat
# Internal Heat transfer coefficient * Specific surface area
from closure.inter_heat import beHeff_I, beHeff_E
# pressure Drop
from closure.pressure_drop import SPresM
# Resistance Term in the Regenerator and void
from closure.resistance import ThermalResistance,ThermalResistanceVoid


############################################## SOLVER ##############################################
############################################## TDMA   ##############################################
from core.tdma_solver import TDMAsolver


######################################## EXPONENTIAL SCHEME #########################################

@jit
def alpha_exp(Pe):
    # Exponential Scheme
    val = np.abs(Pe)/(np.expm1(np.abs(Pe)))
    return np.max([0,val])



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
    if Vd>0:
        # Add a value at the start
        # Add bc, value to the start of the flow
        for j in range(N-1): # note range will give numbers from 0->N-2
            # Build tridagonal matrix coefficients
            # pg. 115 Theo Christiaanse 2017
            a[j] = -CS*Ksw[j]/(dx*2)
            b[j] = Cs[j+1]/dt+CS*Ksw[j]/(dx*2)+CS*Kse[j]/(dx*2)+Omegas[j+1]/2
            c[j] = -CS*Kse[j]/(dx*2)
            d[j] = (iynext[j]+iynext[j+1])*Omegas[j+1]/2+ sprev[j]*(CS*Ksw[j]/(2*dx)) + sprev[j+1]*(Cs[j+1]/dt-Omegas[j+1]/2-CS*Ksw[j]/(2*dx)-CS*Kse[j]/(2*dx)) + sprev[j+2]*(CS*Kse[j]/(2*dx)) + CMCE*Smce[j+1]
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
    else:
        # Add a value at the end
        for j in range(N-1):  # This will loop through 0 to N-1 which aligns with 1->N-1
            # Build tridagonal matrix coefficients
            # pg. 115 Theo Christiaanse 2017
            a[j] = -CS*Ksw[j]/(dx*2)
            b[j] = Cs[j+1]/dt+CS*Ksw[j]/(dx*2)+CS*Kse[j]/(dx*2)+Omegas[j+1]/2
            c[j] = -CS*Kse[j]/(dx*2)
            d[j] = (iynext[j+1]+iynext[j+2])*Omegas[j+1]/2+sprev[j]*(CS*Ksw[j]/(2*dx)) + sprev[j+1]*(Cs[j+1]/dt-CS*Ksw[j]/(2*dx)-CS*Kse[j]/(2*dx)-Omegas[j+1]/2) + sprev[j+2]*(CS*Kse[j]/(2*dx)) + CMCE*Smce[j+1]
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

@jit(f8(f8),nopython=True)
def alpha_pow(Pe):
    # Powerlaw
    val = (1-0.1*abs(Pe))**5
    return max(0,val)


######################################## FLUID SOLVER ##############################################
@jit(f8[:]     (f8[:], f8[:],f8[:],f8[:],f8, f8[:], f8[:], f8[:],  f8[:],  f8[:], f8[:], f8[:], f8, f8[:], f8,int32,f8,f8,f8[:]),nopython=True)
def SolveFluid(iynext,isnext,yprev,sprev,Vd,    Cf,   Kfe,   Kfw,     Ff, Omegaf,    Lf,    Sp, CF, CL,   CVD,    N,dx,dt, yamb):
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
    if Vd>0:
        # Add a value at the start
        # Add bc, value to the start of the flow
        # Dirichlet ghost node
        ynext[0]=0
        for j in range(N-1):  # This will loop through 1 to N+1 which aligns with 0->N
            # Build tridagonal matrix coefficients
            # pg 112-113 Theo Christiaanse 2017
            Aw=alpha_pow(Ff[j]/Kfw[j])
            Ae=alpha_pow(Ff[j+1]/Kfe[j])
            a[j] = -Ff[j]/(dx)-Aw*CF*Kfw[j]/(dx*2)+Omegaf[j+1]/2
            b[j] = Cf[j+1]/(dt)+Aw*CF*Kfw[j]/(2*dx)+Ae*CF*Kfe[j]/(2*dx)+CL[j+1]*Lf[j+1]/2+Omegaf[j+1]/2+Ff[j+1]/(dx)
            c[j] = -Ae*CF*Kfe[j]/(dx*2)
            d[j] = yprev[j]*(Aw*CF*Kfw[j]/(2*dx)) + yprev[j+1]*(Cf[j+1]/dt-Aw*CF*Kfw[j]/(dx*2)-Ae*CF*Kfe[j]/(dx*2) - CL[j+1]*Lf[j+1]/2) + yprev[j+2]*(Ae*CF*Kfe[j]/(dx*2)) + yamb[j+1]*(CL[j+1]*Lf[j+1])+isnext[j+1]*(Omegaf[j+1]/2)+sprev[j+1]*(Omegaf[j+1]/2)+CVD*Sp[j+1]
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
            Aw=alpha_pow(Ff[j+1]/Kfw[j])
            Ae=alpha_pow(Ff[j+2]/Kfe[j])
            a[j] = -Aw*CF*Kfw[j]/(2*dx)
            b[j] = Cf[j+1]/dt+Aw*CF*Kfw[j]/(2*dx)+Ae*CF*Kfe[j]/(2*dx)+CL[j+1]*Lf[j+1]/2+Omegaf[j+1]/2-Ff[j+1]/(dx)
            c[j] = Ff[j+2]/(dx)-Ae*CF*Kfe[j]/(2*dx)+Omegaf[j+1]/2
            d[j] = yprev[j]*(Aw*CF*Kfw[j]/(2*dx)) + yprev[j+1]*(Cf[j+1]/dt-Aw*CF*Kfw[j]/(dx*2)-Ae*CF*Kfe[j]/(dx*2)-CL[j+1]*Lf[j+1]/2) + yprev[j+2]*(Ae*CF*Kfe[j]/(dx*2)) + yamb[j+1]*(CL[j+1]*Lf[j+1]) + isnext[j+1]*(Omegaf[j+1]/2) + sprev[j+1]*(Omegaf[j+1]/2) + CVD*Sp[j+1]
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


####################################### LOOP FUNC ##############################################
@jit(nb.types.Tuple((b1,f8))(f8[:],f8[:],f8))
def AbsTolFunc(var1,var2,Tol):
    maximum_val=np.max(np.abs(var1-var2))
    return maximum_val<=Tol,maximum_val

@jit(nb.types.Tuple((b1,f8))(f8[:,:],f8[:,:],f8))
def AbsTolFunc2d(var1,var2,Tol):
    maximum_val=np.max(np.abs(var1-var2))
    return maximum_val<=Tol,maximum_val

######################################### RUN ACTIVE ############################################

def runActive(caseNum,Thot,Tcold,cen_loc,Tambset,dispV,ff,CF,CS,CL,CVD,CMCE,nodes,timesteps,Dsp,ConfName,jobName):
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
    # ConfName   <- Load a certain configuration file
    # jobName    <- The name of the job

    ########### 14-9-2017 16:39
    code has been check on NTU and U correctness. Some interesting spike was found
    when fluid speed hit zero. This can be solved by using the eps method which worked
    in COMSOL as well.
        - Moving forward implementing fluid and solid properties.
    ########### 18-9-2017 08:43
    Fluid and Solid properties have been inplemented.
    Code has been change to a function.
    Pressure drop term has been implemented however, need to redo the math on
    term.
    ###########    ''     09:20
    Implemented pressure and leak terms. Math checks out. Should be good to go
    and do some spatial and temporal resolution tests.
         - Need to implement Glass spheres and Void space options. Should not
           be difficult as I've already implemented the distretization of the
           discription and build placer functions to implement the different
           closure functions.
    ###########     ''      13:33
    Cleaning up the functions so only what is changing per time step is taken
    as an imput to the function. This makes the code a lot more readable.
    ########### 14-11-2017 14:17
    This version of the code is ported from the V4 version before the gradient
    porosity was added. The code has modified to activate the field again.
    '''


    # Import the configuration
    if ConfName == "R1":
        from configurations.R1  import Ac,Dspgs,Dspls,L_add,L_reg1, L_reg2, Lvoid,Lvoid1, Lvoid2,MOD_CL,Nd,Pc,egs,els,er,gsCp,gsK,gsRho,kair,kg10,kult,lsCp,lsK,lsRho,mK,mRho,magOffset,percGly,r1,r2,r3,rv,rvs,rvs1,rvs2,species_discription,x_discription,CL_set,ch_fac
    if ConfName == "R2":
        from configurations.R2  import Ac,Dspgs,Dspls,L_add,L_reg1, L_reg2, Lvoid,Lvoid1, Lvoid2,MOD_CL,Nd,Pc,egs,els,er,gsCp,gsK,gsRho,kair,kg10,kult,lsCp,lsK,lsRho,mK,mRho,magOffset,percGly,r1,r2,r3,rv,rvs,rvs1,rvs2,species_discription,x_discription,CL_set,ch_fac
    if ConfName == "R3":
        from configurations.R3  import Ac,Dspgs,Dspls,L_add,L_reg1, L_reg2, Lvoid,Lvoid1, Lvoid2,MOD_CL,Nd,Pc,egs,els,er,gsCp,gsK,gsRho,kair,kg10,kult,lsCp,lsK,lsRho,mK,mRho,magOffset,percGly,r1,r2,r3,rv,rvs,rvs1,rvs2,species_discription,x_discription,CL_set,ch_fac
    if ConfName == "R4":
        from configurations.R4  import Ac,Dspgs,Dspls,L_add,L_reg1, L_reg2, Lvoid,Lvoid1, Lvoid2,MOD_CL,Nd,Pc,egs,els,er,gsCp,gsK,gsRho,kair,kg10,kult,lsCp,lsK,lsRho,mK,mRho,magOffset,percGly,r1,r2,r3,rv,rvs,rvs1,rvs2,species_discription,x_discription,CL_set,ch_fac
    if ConfName == "R5":
        from configurations.R5  import Ac,Dspgs,Dspls,L_add,L_reg1, L_reg2, Lvoid,Lvoid1, Lvoid2,MOD_CL,Nd,Pc,egs,els,er,gsCp,gsK,gsRho,kair,kg10,kult,lsCp,lsK,lsRho,mK,mRho,magOffset,percGly,r1,r2,r3,rv,rvs,rvs1,rvs2,species_discription,x_discription,CL_set,ch_fac
    if ConfName == "R6":
        from configurations.R6  import Ac,Dspgs,Dspls,L_add,L_reg1, L_reg2, Lvoid,Lvoid1, Lvoid2,MOD_CL,Nd,Pc,egs,els,er,gsCp,gsK,gsRho,kair,kg10,kult,lsCp,lsK,lsRho,mK,mRho,magOffset,percGly,r1,r2,r3,rv,rvs,rvs1,rvs2,species_discription,x_discription,CL_set,ch_fac

    print("Hot Side: {} Cold Side: {}".format(Thot,Tcold))
    # Start Timer
    t0 = time.time()
    # Volume displacement in [m3]
    # Small displacer 2.53cm^2
    # Medium displacer 6.95cm^2
    # 1inch = 2.54cm
    Vd      = dispV
    freq    = ff
    tau_c   = 1/freq

    # Number of spatial nodes
    N = nodes
    # Number of time steps
    nt = timesteps
    # Molecule size
    dx = 1 / (N+1)
    # The time step is:
    dt = 1 / (nt+1)

    # To prevent the simulation running forever we limit the simulation to
    # find a tollerance setting per step and per cycle.
    maxStepTol  = [1,1e-1,1e-2,1e-3,1e-4,1e-5,1e-6,1e-7]
    maxCycleTol = 1e-6
    # We also limit the number of steps and cycles the simulation can take. If
    # the simulation runs over these there might be an issue with the code.
    maxSteps  = 2000
    maxCycles = 2000
    # Cycle period
    t = np.linspace(0, tau_c, nt+1)
    # Total length of the domain
    L_tot   = np.max(x_discription)  # [m]
    # Real element size
    # Element size
    DX = L_tot/ (N+1)
    # Time step
    DT = tau_c/(nt+1)

    # Build darcy velocity for all n steps
    # Sine wave (See notebook pg. 33 Theo Christiaanse Sept 2017)
    vf = lambda at, Ac, Vd, sf: (Vd) * sf * np.pi * np.sin(2 * np.pi * sf * at) + np.sign(np.sin(2 * np.pi * sf * at))*sys.float_info.epsilon*2
    # block wave (See notebook pg. 111 Theo Christiaanse Sept 2017)
    #uf = lambda at, Ac, Vd, sf: (Vd*sf / (Ac)) * np.sign(1/(sf*2)-sys.float_info.epsilon-at)
    pdrop = lambda at, dP, sf: (dP) * sf * np.pi * np.sin(2 * np.pi * sf * at) + np.sign(np.sin(2 * np.pi * sf * at))*sys.float_info.epsilon*2

    # Build all volumetric fluid speeds V[n] (m^3/s) dt time unit
    V = vf(t, Ac, Vd, freq)
    dPreg    = 14.7 * 6894.7572931783/2
    Lreg_exp = 45e-3
    ddP      = pdrop(t, dPreg, freq)
    dPdz_exp = ddP/Lreg_exp
    #U = uf(t, 1, 1, freq)

    # Calculate the utilization as defined by Armando's paper
    Uti     = (Vd * 1000 * 4200) / (1000 * Ac * (1 - er) * 6100 * (L_reg1+L_reg2))
    print('Utiization: {0:1.3f} Frequency: {1:1.2f}'.format(Uti,freq))
    print('Urms: {0:3.3f}'.format((Vd / Ac*er) * freq * np.pi*1/np.sqrt(2)))

    # Initial ch-factor
    ch_factor = np.ones(N + 1)*ch_fac

    # This is modification of the casing BC
    if CL_set=="Tamb":
        # Ambiant Temperature non-diamentionilized
        yamb = np.ones(N + 1) * ((Tambset - Tcold) / (Thot - Tcold))
        CL = np.ones(N+1)
    if CL_set=="f292":
        yamb = np.ones(N + 1) * ((292 - Tcold) / (Thot - Tcold))
        CL = np.ones(N+1)
    if CL_set=="grad":
        yamb = (np.linspace(Tcold,Thot,num=N+1) - Tcold) / (Thot - Tcold)
        CL = np.ones(N+1)
    if CL_set=="adiabatic":
        yamb = (np.linspace(Tcold,Thot,num=N+1) - Tcold) / (Thot - Tcold)
        CL = np.zeros(N+1)
    if MOD_CL==1:
        # Outer component discription
        #
        # insul
        # insulator this will set casing losses to zero
        # condu
        # This indicates a conductor, temperature will be set based on the temperature assumptions
        # air
        # This part is set to the temperature of the ambient
        # hothex
        # This will be the temperature set by the hot hex
        # coldhex
        # This will be the temperature set by the cold hex
        outer_dis = ['coldhex','condu','condu','condu','air','hothex']
        # Length left flange
        L_lf= 0.019
        # Length magnet with plates
        L_mp= 0.112
        # Length right flange
        L_rf= 0.047
        # Length air gap
        L_ag = 0.0146
        outer_x   = [0,
                    L_add,
                    L_add+L_lf,
                    L_add+L_lf+L_mp,
                    L_add+L_lf+L_mp+L_rf,
                    L_add+L_lf+L_mp+L_rf+L_ag,
                    L_add+0.209]
        # Loop through discription to make array CL and Tamb
        mm=0
        # We need to distribute the space identifiers along a matrix to use it later.
        # This funcion cycles through x_discription until it finds a new domain then sets acording
        # to the int_discription
        int_disc_outer      = np.zeros(N+1,dtype=np.int)
        outer_descriptor    = []
        xloc_outer          = np.zeros(N+1)
        # Set the rest of the nodes to id with geoDis(cription)
        for i in range(N+1): # sets 0->N
            xloc_outer[i] = (DX * i + DX / 2)  #modify i so 0->N
            if (xloc_outer[i] >= outer_x[mm + 1]):
                mm = mm + 1
            int_disc_outer[i] = mm
            outer_descriptor.append(outer_dis[mm])
            if outer_descriptor[i] == 'coldhex':
                yamb[i] = 0
                CL[i]   = 1
            if outer_descriptor[i] == 'hothex':
                yamb[i] = 1
                CL[i]   = 1
            if outer_descriptor[i] == 'insul':
                CL[i]   = 0
            if outer_descriptor[i] == 'condu':
                CL[i]   = 1
                yamb[i] = ((273+OptVar - Tcold) / (Thot - Tcold))
            if outer_descriptor[i] == 'air':
                yamb[i] = ((Tambset - Tcold) / (Thot - Tcold))
                CL[i]   = 1



    ## Field settings for PM1
    nn = 0
    # We need to distribute the space identifiers along a matrix to use it later.
    # This funcion cycles through x_discription until it finds a new domain then sets acording
    # to the int_discription
    int_discription = np.zeros(N+1,dtype=np.int)
    species_descriptor = []
    xloc            = np.zeros(N+1)
    # Set the rest of the nodes to id with geoDis(cription)
    for i in range(N+1): # sets 0->N
        xloc[i] = (DX * i + DX / 2)  #modify i so 0->N
        if (xloc[i] >= x_discription[nn + 1]):
            nn = nn + 1
        int_discription[i] = nn
        species_descriptor.append(species_discription[nn])



    ## Set Surface area and Porosity of solid and fluid
    A_c = np.ones(N + 1)
    e_r = np.ones(N + 1)
    P_c = np.ones(N + 1)
    for i in range(N+1):
        if species_descriptor[i].startswith("reg"):
            A_c[i] = Ac
            e_r[i] = er
            P_c[i] = Pc
        elif species_descriptor[i]== 'gs':
            A_c[i] = Ac
            e_r[i] = egs
            P_c[i] = Pc
        elif species_descriptor[i]== 'ls':
            A_c[i] = Ac
            e_r[i] = els
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
    # This one exists everywhere.
    appliedFieldm = np.ones((nt+1, N + 1))
    for i in range(N + 1):
        for n in range(0, nt+1):
            # Will only get the field if we find a regenerator
            if (species_descriptor[i].startswith("reg")):
                x_pos_w_respect_to_magnet = xloc[i] - magOffset
                appliedFieldm[n, i] = hapl.appliedField(x_pos_w_respect_to_magnet, rotMag[n])[0, 0]*CMCE
            else:
                appliedFieldm[n, i] = 0
    ########################## START THE LOOP #########################

    # Some housekeeping to make this looping work

    #
    # Initial temperature
    y1 = np.linspace(0,1, N + 1)
    s1 = np.linspace(0, 1, N + 1)

    #
    # Check is there is some pickeled data
    PickleFileName = "./pickleddata/{0:}-{1:d}".format(jobName,int(caseNum))
    print("Pickle Data File: {}".format(PickleFileName))
    try:
        # we open the file for reading
        fileObject = open(PickleFileName,'rb')
        print("we are loading the pickle file!")
        # load the object from the file into var b
        bbb = pickle.load(fileObject)
        y   = bbb[0]
        s   = bbb[1]
        stepTolInt = bbb[2]
        iyCycle = bbb[3]
        isCycle = bbb[4]
    except FileNotFoundError:
        # Keep preset values
        print("started normal")
        y = np.ones((nt+1, N + 1))*y1
        s = np.ones((nt+1, N + 1))*s1
        stepTolInt = 0
        # Initial guess of the cycle values.
        iyCycle = np.copy(y)
        isCycle = np.copy(s)

    # Magnetic Field Modifier
    MFM = np.ones(N + 1)
    #
    cycleTol   = 0
    cycleCount = 1

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
                if   species_descriptor[i]== 'reg-si1': mag_c = si1.mMag_c; mag_h = si1.mMag_h
                elif species_descriptor[i]== 'reg-si2': mag_c = si2.mMag_c; mag_h = si2.mMag_h
                elif species_descriptor[i]== 'reg-si3': mag_c = si3.mMag_c; mag_h = si3.mMag_h
                elif species_descriptor[i]== 'reg-si4': mag_c = si4.mMag_c; mag_h = si4.mMag_h
                elif species_descriptor[i]== 'reg-si5': mag_c = si5.mMag_c; mag_h = si5.mMag_h
                maxMagLoc= mag_c(Ts_ave,maxApliedField)[0, 0]*(1-ch_factor[i])+mag_h(Ts_ave,maxApliedField)[0, 0]*ch_factor[i]
                # The resulting internal field
                Hint = maxApliedField - mRho * Nd * maxMagLoc * mu0
                # The decrease ratio of the applied field
                MFM[i] = Hint/maxApliedField
        for n in range(1, nt+1):  # 1->nt
            # Run every timestep
            # Initial
            stepTol = 0
            stepCount = 1
            # ch_factor[i]=0 coolingcurve selected
            # ch_factor[i]=1 heatingcurve seelected
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
            S_c_past  = np.zeros(N+1)
            S_h_past  = np.zeros(N+1)
            Sirr_prev = np.zeros(N+1)
            Sprev     = np.zeros(N+1)
            prevHintNew  = np.zeros(N+1)


            for i in range(N+1):
                if species_descriptor[i].startswith("reg"):
                    # Internal field
                    prevHint        = appliedFieldm[n-1,i]*MFM[i]
                    prevHintNew[i]  = appliedFieldm[n-1,i]*MFM[i]
                    if   species_descriptor[i]== 'reg-si1': cp_c = si1.mCp_c; cp_h = si1.mCp_h; ms_c = si1.mS_c; ms_h = si1.mS_h
                    elif species_descriptor[i]== 'reg-si2': cp_c = si2.mCp_c; cp_h = si2.mCp_h; ms_c = si2.mS_c; ms_h = si2.mS_h
                    elif species_descriptor[i]== 'reg-si3': cp_c = si3.mCp_c; cp_h = si3.mCp_h; ms_c = si3.mS_c; ms_h = si3.mS_h
                    elif species_descriptor[i]== 'reg-si4': cp_c = si4.mCp_c; cp_h = si4.mCp_h; ms_c = si4.mS_c; ms_h = si4.mS_h
                    elif species_descriptor[i]== 'reg-si5': cp_c = si5.mCp_c; cp_h = si5.mCp_h; ms_c = si5.mS_c; ms_h = si5.mS_h
                    # Previous specific heat
                    Tr=psT[i]
                    dT=.5
                    dsdT = (ms_c(Tr+dT, prevHint)[0, 0]*(.5)  +  ms_h(Tr+dT, prevHint)[0, 0]*(.5)) - (ms_c(Tr-dT, prevHint)[0, 0]*(.5)  +  ms_h(Tr-dT, prevHint)[0, 0]*(.5))
                    cps_prev[i]  = psT[i]*(np.abs(dsdT)/(dT*2))
                    # Entropy position of the previous value
                    S_c_past[i]   = ms_c(Tr, prevHint)[0, 0]
                    S_h_past[i]   = ms_h(Tr, prevHint)[0, 0]
                    Sirr_prev[i]  = S_c_past[i] *(1-ch_factor[i])  -  S_h_past[i] *ch_factor[i]
                    Sprev[i]      = S_c_past[i] *(1-ch_factor[i])  +  S_h_past[i] *ch_factor[i]
                    # old code
                    Ss_prev[i]     = ms_c(psT[i], prevHint)[0, 0]*(1-ch_factor[i]) + ms_h(psT[i], prevHint)[0, 0]*ch_factor[i]
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
                elif species_descriptor[i]== 'ls':
                    cps_prev[i]    = lsCp
                    Ss_prev[i]     = 0
                    # This is where the gs stuff will go
                else:
                    cps_prev[i]    = 0
                    Ss_prev[i]     = 0
                    # This is where the void stuff will go
                # liquid calculations
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
                    if species_descriptor[i].startswith("reg"):
                        if   species_descriptor[i]== 'reg-si1': cp_c = si1.mCp_c; cp_h = si1.mCp_h; ms_c = si1.mS_c; ms_h = si1.mS_h; T_h=si1.mTemp_h; T_c = si1.mTemp_c; Reduct     = 0.55;
                        elif species_descriptor[i]== 'reg-si2': cp_c = si2.mCp_c; cp_h = si2.mCp_h; ms_c = si2.mS_c; ms_h = si2.mS_h; T_h=si1.mTemp_h; T_c = si1.mTemp_c; Reduct     = 0.77;
                        elif species_descriptor[i]== 'reg-si3': cp_c = si3.mCp_c; cp_h = si3.mCp_h; ms_c = si3.mS_c; ms_h = si3.mS_h; T_h=si1.mTemp_h; T_c = si1.mTemp_c; Reduct     = 0.73;
                        elif species_descriptor[i]== 'reg-si4': cp_c = si4.mCp_c; cp_h = si4.mCp_h; ms_c = si4.mS_c; ms_h = si4.mS_h; T_h=si1.mTemp_h; T_c = si1.mTemp_c; Reduct     = 0.75;
                        elif species_descriptor[i]== 'reg-si5': cp_c = si5.mCp_c; cp_h = si5.mCp_h; ms_c = si5.mS_c; ms_h = si5.mS_h; T_h=si1.mTemp_h; T_c = si1.mTemp_c; Reduct     = 0.72;
                        # Field
                        Hint = appliedFieldm[n, i]*MFM[i]
                        # rho*cs fluid
                        dT=1
                        Tr         = psT[i]
                        aveField = (Hint+prevHintNew[i])/2
                        dsdT = (ms_c(sT[i]+dT, aveField)[0, 0]*(.5)  +  ms_h(sT[i]+dT, aveField)[0, 0]*(.5)) - (ms_c(sT[i]-dT, aveField)[0, 0]*(.5)  +  ms_h(sT[i]-dT, aveField)[0, 0]*(.5))
                        cps_curr   = Tr*(np.abs(dsdT)/(dT*2))
                        cps_ave = cps_curr  * ww + cps_prev[i] * (1 - ww)
                        rhos_cs_ave[i] = cps_ave * mRho
                        # Smce
                        S_c_curr   = ms_c(Tr, Hint)[0, 0]
                        S_h_curr   = ms_h(Tr, Hint)[0, 0]
                        Sirr_cur   = S_c_curr *(1-ch_factor[i])  -  S_h_curr *ch_factor[i]
                        Scur       = S_c_curr *(1-ch_factor[i])  +  S_h_curr *ch_factor[i]
                        #Mod        = 0.5*(Sirr_cur+Sirr_prev[i])*np.abs((2*dT)/dsdT)
                        Smce[i]    = (Reduct*A_c[i] * (1 - e_r[i]) * mRho * Tr * (Sprev[i]-Scur))/ (DT* (Thot - Tcold))
                        # Effective Conduction for fluid
                        k[i] = kDyn_P(Dsp, e_r[i], cpf_ave, kf_ave, rhof_ave, np.abs(V[n] / (A_c[i])))
                        # Forced convection term east of the P node
                        Omegaf[i] = A_c[i] * beHeff_E(Dsp, np.abs(V[n] / (A_c[i])), cpf_ave, kf_ave, muf_ave, rhof_ave, freq, cps_ave, mK, mRho, e_r[i])  # Beta Times Heff
                        # Pressure drop
                        dP = ddP[n]
                        Spres[i] = dPdz_exp[n]*V[n]
                        # Loss term
                        Lf[i] = P_c[i] * ThermalResistance(Dsp, np.abs(V[n] / (A_c[i])), muf_ave, rhof_ave, kair, kf_ave, kg10, r1, r2, r3)
                        # Effective Conduction for solid
                        ks[i] = kStat(e_r[i], kf_ave, mK)
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
                    elif species_descriptor[i] == 'ls':
                        # This is where the gs stuff will go
                        # Effective Conduction for solid
                        rhos_cs_ave[i] = lsCp * lsRho
                        # Effective Conduction for fluid
                        k[i] = kDyn_P(Dspls, e_r[i], cpf_ave, kf_ave, rhof_ave, np.abs(V[n] / (A_c[i])))
                        # Forced convection term east of the P node
                        Omegaf[i] = A_c[i] * beHeff_I(Dspls, np.abs(V[n] / (A_c[i])), cpf_ave, kf_ave, muf_ave, rhof_ave,
                                                      freq, lsCp, lsK, lsRho, e_r[i])  # Beta Times Heff
                        # Pressure drop
                        Spres[i], dP = SPresM(Dspls, np.abs(V[n] / (A_c[i])), np.abs(V[n]), e_r[i], muf_ave, rhof_ave,
                                              A_c[i] * e_r[i])
                        # Loss term
                        Lf[i] = P_c[i] * ThermalResistance(Dspls, np.abs(V[n] / (A_c[i])), muf_ave, rhof_ave, kair, kf_ave,
                                                       kg10, r1, r2, r3)
                        # Effective Conduction for solid
                        ks[i] = kStat(e_r[i], kf_ave, lsK)
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
                Kfw = Kfw /(DX*L_tot)
                Kfe = Kfe /(DX*L_tot)
                Ksw = Ksw /(DX*L_tot)
                Kse = Kse /(DX*L_tot)


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
                # Add a new step to the step count
                stepCount = stepCount + 1
                # Check if we have hit the max steps
                if (stepCount == maxSteps):
                    print("Hit max step count")
                    print(AbsTolFunc(ynext,iynext,maxStepTol[stepTolInt]))
                    print(AbsTolFunc(snext,isnext,maxStepTol[stepTolInt]))
                    print(stepTol)
                # Copy current values to new guess and current step.
                s[n, :] = np.copy(snext)
                isnext  = np.copy(snext)
                y[n, :] = np.copy(ynext)
                iynext  = np.copy(ynext)
                if (np.any(np.isnan(y)) or np.any(np.isnan(s))):
                    # Sometimes the simulation hits a nan value. We can redo this point later.
                    print(y)
                    print(s)
                    break
                # Break the step calculation
                if ((time.time()-t0)/60)>110:
                    break
                ################################################################
                ####################### ELSE ITERATE AGAIN  ####################
            # break the cycle calculation
            if ((time.time()-t0)/60)>110:
                break
        # Check if current cycle is close to previous cycle.
        [bool_y_check,max_val_y_diff]=AbsTolFunc2d(y,iyCycle,maxCycleTol)
        [bool_s_check,max_val_s_diff]=AbsTolFunc2d(s,isCycle,maxCycleTol)
        cycleTol = bool_y_check and bool_s_check
        if (max_val_y_diff/10)<maxStepTol[stepTolInt]:
            stepTolInt = stepTolInt + 1
            if stepTolInt == len(maxStepTol):
                stepTolInt=len(maxStepTol)-1
        if cycleCount%10==1:
            coolingpowersum=0
            startint=0
            for n in range(startint, nt):
                tF = y[n, 0] * (Thot - Tcold) + Tcold
                tF1 = y[n+1, 0] * (Thot - Tcold) + Tcold
                coolPn =  freq * fCp((tF+tF1)/2,percGly) * fRho((tF+tF1)/2,percGly) * V[n] * DT * ((tF+tF1)/2 - Tcold)
                coolingpowersum = coolingpowersum + coolPn
            qc = coolingpowersum * 2
            print("Case num {0:d} CycleCount {1:d} Cooling Power {2:2.5e} y-tol {3:2.5e} s-tol {4:2.5e} run time {5:4.1f} [min]".format(int(caseNum),cycleCount,qc,max_val_y_diff,max_val_s_diff,(time.time()-t0)/60 ))
        # Break the cycle while
        if ((time.time()-t0)/60)>110:
            coolingpowersum=0
            startint=0
            for n in range(startint, nt):
                tF = y[n, 0] * (Thot - Tcold) + Tcold
                tF1 = y[n+1, 0] * (Thot - Tcold) + Tcold
                coolPn =  freq * fCp((tF+tF1)/2,percGly) * fRho((tF+tF1)/2,percGly) * V[n] * DT * ((tF+tF1)/2 - Tcold)
                coolingpowersum = coolingpowersum + coolPn
            qc = coolingpowersum * 2
            print("Case num {0:d} CycleCount {1:d} Cooling Power {2:2.5e} y-tol {3:2.5e} s-tol {4:2.5e} run time {5:4.1f} [min]".format(int(caseNum),cycleCount,qc,max_val_y_diff,max_val_s_diff,(time.time()-t0)/60 ))
            # Pickle data
            aaa = ((y,s,stepTolInt,iyCycle,isCycle))
            # open the file for writing
            fileObject = open(PickleFileName,'wb')
            # this writes the object a to the
            # file named 'testfile'
            pickle.dump(aaa,fileObject)
            # here we close the fileObject
            fileObject.close()
            print("saving pickle file")
            # Quit Program
            sys.exit()
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
    ########################## END THE LOOP #########################


    quart=int(nt/4)
    halft=int(nt/2)
    tquat=int(nt*3/4)
    # This is a modifier so we can modify the boundary at which we calculate the
    # effectiveness
    cold_end_node  = np.min(np.argwhere([val.startswith("reg") for val in species_descriptor])) # 0
    hot_end_node   = np.max(np.argwhere([val.startswith("reg") for val in species_descriptor])) # -1
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

    coolingpowersum=0
    startint=0
    for n in range(startint, nt):
        tF = y[n, 0] * (Thot - Tcold) + Tcold
        tF1 = y[n+1, 0] * (Thot - Tcold) + Tcold
        coolPn =  freq * fCp((tF+tF1)/2,percGly) * fRho((tF+tF1)/2,percGly) * V[n] * DT * ((tF+tF1)/2 - Tcold)
        coolingpowersum = coolingpowersum + coolPn
    qc = coolingpowersum * 2

    coolingpowersum=0
    startint=0
    for n in range(startint, nt):
        tF = y[n, -1] * (Thot - Tcold) + Tcold
        tF1 = y[n+1, -1] * (Thot - Tcold) + Tcold
        coolPn =  freq * fCp((tF+tF1)/2,percGly) * fRho((tF+tF1)/2,percGly) * V[n] * DT * (Thot-(tF+tF1)/2)
        coolingpowersum = coolingpowersum + coolPn
    qh = coolingpowersum * 2

    Kamb = 0.28
    qccor = qc - Kamb * (Tambset-Tcold)
    pave = np.trapz(pt[halft:], x=t[halft:]) / (tau_c/2)
    print("{0:3.1f} {1:3.1f} {2:1.2f} {3:1.2f} Cycle Count: {4:d} Tol-y: {5:1.4e} Tol-s {6:1.4e}".format(float(Thot),float(Tcold),float(Uti),float(freq),int(cycleCount),float(max_val_y_diff),float(max_val_s_diff)))
    print('Utilization: {0:1.3f} Frequency: {1:1.2f}'.format(Uti,freq))
    print("Run time: {0:3.2f} [min]".format((t1 - t0) / 60))
    print("Hot Side: {} Cold Side: {}".format(Thot,Tcold))
    print('Effectiveness HB-CE {} CB-HE {}'.format(eff_HB_CE,eff_CB_HE))
    print('Cooling power {}'.format(qc))
    print('Corrected Cooling Power {}'.format(qccor))
    print('Pressure drop {} (kPa)'.format(pave/1000))

    print('Values found at minimal field')
    print('min Applied Field {}'.format(minAplField))
    print('min Internal Field {}'.format(minPrevHint))
    print('min Magnetic Temperature {}'.format(minMagTemp))
    print('min Cp: {}'.format(minCpPrev))
    print('min SS:{}'.format(minSSprev))
    print('Lowest Temperature found in the SS cycle: {}'.format(minTemp))


    print('Values found at maximum field')
    print('max Applied Field {}'.format(maxAplField))
    print('max Internal Field {}'.format(maxPrevHint))
    print('max Magnetic Temperature {}'.format(maxMagTemp))
    print('max Cp: {}'.format(maxCpPrev))
    print('max SS:{}'.format(maxSSprev))
    print('highest Temperature found in the SS cycle: {}'.format(maxTemp))

    # Remove Pickle
    try:
        os.remove(PickleFileName)
        print("We removed the pickle file")
    except FileNotFoundError:
        print('Hey! It was done very fast.')

    return Thot,Tcold,qc,qccor,(t1-t0)/60,pave,eff_HB_CE,eff_CB_HE,tFce,tFhe,yHalfBlow,yEndBlow,sHalfBlow,sEndBlow,y, s, pt, np.max(pt),Uti,freq,t,xloc,yMaxCBlow,yMaxHBlow,sMaxCBlow,sMaxHBlow,qh


if __name__ == '__main__':

    # Some useful functions for storing data.
    def FileSave(filename, content):
        with open(filename, "a") as myfile:
            myfile.write(content)

    def FileSaveMatrix(filename, content):
        with open(filename, "a") as f:
            for line in content:
                f.write(" ".join("{:9.6f}\t".format(x) for x in line))
                f.write("\n")

    def RunCaseThotTcold(case,jobName):
        numCases       = 1
        hotResolution  = 20
        coldResolution = 20

        maxcase =  numCases * hotResolution * coldResolution
        Thotarr = np.linspace(273+33, 273+15, hotResolution)

        casenum=int(np.floor(case/(hotResolution*coldResolution)))

        if (casenum==0):
            #RunTest("test_128_20_ALL.txt", 6.4e-6, 2, CF, CS, CL, CVD,CMCE, Thot, 35, num_processors, 200, 400, [0,20,40],300e-6)
            fileName      = "{}.txt".format(jobName)
            MaxTSpan      = 17
            cen_loc       = 0
            Tambset       = 294
            dispV         = 3.91e-6
            ff            = 1
            Dsp           = 425e-6
            CF            = 1
            CS            = 1
            CL            = 0
            CVD           = 1
            CMCE          = 1
            nodes         = 800
            timesteps     = 800
            cName         = "R2"
        if (casenum==1):
            #RunTest("test_128_20_ALL.txt", 6.4e-6, 2, CF, CS, CL, CVD,CMCE, Thot, 35, num_processors, 200, 400, [0,20,40],300e-6)
            fileName      = "test_078_10_ALL_s_4_dv.txt"
            MaxTSpan      = 20
            cen_loc       = 0
            Tambset       = 294
            dispV         = 3.91e-6
            ff            = 1
            Dsp           = 425e-6
            CF            = 1
            CS            = 1
            CL            = 1
            CVD           = 1
            CMCE          = 1
            nodes         = 400
            timesteps     = 400
        if (casenum==2):
            #RunTest("test_128_20_ALL.txt", 6.4e-6, 2, CF, CS, CL, CVD,CMCE, Thot, 35, num_processors, 200, 400, [0,20,40],300e-6)
            fileName      = "test_ch_conf1.txt"
            MaxTSpan      = 15
            cen_loc       = 0
            Tambset       = 294
            dispV         = 3.91e-6
            ff            = 1
            Dsp           = 425e-6
            CF            = 1
            CS            = 1
            CL            = 1
            CVD           = 1
            CMCE          = 1
            nodes         = 800
            timesteps     = 800
        if (casenum==3):
            #RunTest("test_128_20_ALL.txt", 6.4e-6, 2, CF, CS, CL, CVD,CMCE, Thot, 35, num_processors, 200, 400, [0,20,40],300e-6)
            fileName      = "test_078_10_ALL_s_16_dv.txt"
            MaxTSpan      = 20
            cen_loc       = 0
            Tambset       = 294
            dispV         = 3.91e-6
            ff            = 1
            Dsp           = 425e-6
            CF            = 1
            CS            = 1
            CL            = 1
            CVD           = 1
            CMCE          = 1
            nodes         = 1600
            timesteps     = 400

        Thot = Thotarr[int(np.floor(case/coldResolution)%hotResolution)]
        Tcold = Thot - MaxTSpan*(case%(coldResolution))/(coldResolution)-0.1

        print("iteration: {}/{} Case number: {} Thot: {} Tcold: {}".format(case,maxcase,casenum,Thot,Tcold))

        results = runActive(case,Thot,Tcold,cen_loc,Tambset,dispV,ff,CF,CS,CL,CVD,CMCE,nodes,timesteps,Dsp,cName,jobName)
        # Get result roots variable is broken down in:
        #  0     1    2   3      4        5
        # Thot,Tcold,qc,qccor,(t1-t0)/60,pave,
        #           6               7
        # integral_eff_HB_CE,integral_eff_CB_HE,
        #  8    9      10        11       12       13
        # tFce,tFhe,yHalfBlow,yEndBlow,sHalfBlow,sEndBlow,
        # 14 15 16      17     18   19  20 21
        # y, s, pt, np.max(pt),Uti,freq,t,xloc
        # 22           23       24         25
        #,yMaxCBlow,yMaxHBlow,sMaxCBlow,sMaxHBlow
        fileNameSave        = './' + fileName
        fileNameEndTemp     = './Ends/{:3.0f}-{:3.0f}-PysicalEnd'.format(Thot,Tcold)+fileName
        fileNameSliceTemp   = './Blow/{:3.0f}-{:3.0f}-BlowSlice'.format(Thot,Tcold)+fileName
        FileSave(fileNameSave,"{},{},{},{},{},{},{} \n".format(results[0],results[1],results[2],results[3],results[4],results[5],results[26]) )
        #FileSave(fileNameEndTemp,"{},{},{},{},{} \n".format('Thot [K]', 'Tcold [K]','Uti [-]', 'freq [Hz]', 'run time [min]','Eff CE-HB [-]', 'Eff HE-CB [-]') )
        #FileSave(fileNameEndTemp,"{},{},{},{},{} \n".format(results[0],results[1],results[18],results[19], results[4],results[6],results[7]) )
        #EndTemperatures = np.stack((results[20], results[8],results[9]), axis=-1)
        #FileSaveMatrix(fileNameEndTemp,EndTemperatures)
        #FileSave(fileNameSliceTemp,"{},{},{},{},{} \n".format('Thot [K]', 'Tcold [K]','Uti [-]', 'freq [Hz]', 'run time [min]') )
        #FileSave(fileNameSliceTemp,"{},{},{},{},{} \n".format(results[0],results[1],results[18],results[19], results[4]) )
        #BlowSliceTemperatures = np.stack((results[21],results[10],results[11],results[12],results[13],results[22],results[23],results[24],results[25]), axis=-1)
        #FileSaveMatrix(fileNameSliceTemp,BlowSliceTemperatures)

    RunCaseThotTcold(float(sys.argv[1]),sys.argv[2])
    #RunCaseThotTcold(1)
