

# Numba gives speed.
from numba import jit
# Numpy library
import numpy as np
# Interpolation Functions
from scipy.interpolate import RectBivariateSpline, interp1d
# Plotting
import matplotlib.pyplot as plt

# Importing standard python libraries. 
# This is useful for storing documents
import os
# This is for timing the program. 
import time

########################## Notes 2017-09-12 ####################################
# Please review NTU regenerator example from Heat Transfer – Gregory Nellis and Sanford Klein.
# This code is made to replicate the results presented by an ideal balanced regenerator.
# There is also a power point that has some explanation.

# Start Timer
t0 = time.time()

# Number of spatial nodes
N = 100
# Number of time steps
nt = 300
def FileSave(filename, content):
    with open(filename, "a") as myfile:
        myfile.write(content)

tau_c   = 1
U_val_loop   = np.array([0.01,0.5,1,1.4,2])
NTU_val_loop = np.array([1,2,4,8,10,20,40,80,100,200,400,800])
fileName1 = 'results.txt'
eff_vals=np.zeros((U_val_loop.size,NTU_val_loop.size))

# Molecule size
dx = 1 / (N+1)
# The time step is:
dt = 1 / (nt+1)

# Smoothing factor
# This has shown to be not useful and slows the simulation down so, I've set it
# to w=1 for now.
w=1

# To prevent the simulation running forever we limit the simulation to
# find a tollerance setting per step and per cycle.
maxStepTol  = 1e-6
maxCycleTol = 1e-6
# We also limit the number of steps and cycles the simulation can take. If
# the simulation runs over these there might be an issue with the code.
maxSteps  = 1000
maxCycles = 4000

# NTU and U analysis is based on NTU which flips halfway.
# Cycle period

t       = np.linspace(0, tau_c, nt+1)

m=np.zeros(nt+1)
for n in range(nt+1):
    m[n] = np.sign(tau_c/2-t[n])



@jit
def TDMAsolver(a, b, c, d):
    '''
    TDMA solver, a b c d can be NumPy array type or Python list type.
    refer to http://en.wikipedia.org/wiki/Tridiagonal_matrix_algorithm
    and to http://www.cfd-online.com/Wiki/Tridiagonal_matrix_algorithm_-_TDMA_(Thomas_algorithm)
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

@jit
def SolveSolid(N,iyn,isn,yp,sp,NTU,U,dt,m):
    '''
    Solve the NTU/U solid matrix.
    See notebook "Theo Christiaanse 2017" pg. 92-97
    '''
    # Prepare material properties
    a = np.zeros(N-1)
    b = np.zeros(N-1)
    c = np.zeros(N-1)
    d = np.zeros(N-1)
    snext = np.zeros(N+1)
    if m>0:
        # Add a value at the start
        # Add bc, value to the start of the flow
        for j in range(N-1): # note range will give numbers from 0->N-2
            # Build tridagonal matrix coefficients 1->N-1
            a[j] = 0
            b[j] = NTU/2 + 1/(2*dt*U)
            c[j] = 0
            d[j] = (1/(2*dt*U)-NTU/2)*sp[j+1] + NTU/2 * (yp[j+1] + iyn[j+1])
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
            # Build tridagonal matrix coefficients 0->N
            a[j] = 0
            b[j] = NTU/2 + 1/(2*dt*U)
            c[j] = 0
            d[j] = (1/(2*dt*U)-NTU/2 )*sp[j+1] + NTU/2 *( iyn[j+1] + yp[j+1])
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



@jit
def SolveFluid(N,iyn,isn,yp,sp,NTU,dx,m):
    '''
    Solve the NTU/U Fluid matrix.
    See notebook "Theo Christiaanse 2017" pg. 92-97
    '''
    # Prepare material properties
    a = np.zeros(N-1)
    b = np.zeros(N-1)
    c = np.zeros(N-1)
    d = np.zeros(N-1)
    ynext = np.zeros(N+1)
    if m>0:
        # Add a value at the start
        # Add bc, value to the start of the flow
        # Dirichlet ghost node
        ynext[0]=1
        for j in range(N-1):  # This will loop through 1 to N+1 which aligns with 0->N
            # Build tridagonal matrix coefficients 0->N
            a[j] =  - 1/(dx)
            b[j] = NTU/2 + 1/(dx)
            c[j] = 0
            d[j] = NTU/2*(isn[j+1] + sp[j+1])- NTU/2 *yp[j+1]
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
        ynext[-1]=0
        for j in range(N-1):  # This will loop through 1 to N+1 which aligns with 0->N
            # Build tridagonal matrix coefficients 0->N
            a[j] = 0
            b[j] = NTU/2 + 1/(dx)
            c[j] =  - 1/(dx)
            d[j] = NTU/2*(isn[j+1] + sp[j+1]) - NTU/2 *yp[j+1]
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

@jit
def AbsTolFunc(var1,var2,Tol):
    return np.all(np.abs(var1-var2)<Tol)
# Some housekeeping to make this looping work




for i, U_val in enumerate(U_val_loop):
    for j, NTU_val in enumerate(NTU_val_loop):

        # Initial temperature
        y1 = np.linspace(1,0, N + 1)
        y = np.ones((nt+1, N + 1))*y1
        s1 = np.linspace(1, 0, N + 1)
        s = np.ones((nt+1, N + 1))*s1

        #
        cycleTol   = 0
        cycleCount = 1
        # Initial guess of the cycle values.
        iyCycle = np.copy(y)
        isCycle = np.copy(s)
        while (not cycleTol  and cycleCount <= maxCycles):

            for n in range(1, nt+1):  # 1->nt
                # Run every timestep

                # Initial
                stepTol = 0
                stepCount = 1

                # Initial guess of the current step values.
                iynext  = np.copy(y[n-1, :])
                isnext  = np.copy(s[n-1, :])

                # Loop untill stepTol is found or maxSteps is hit.
                while ( not stepTol and stepCount <= maxSteps):


                    # iynext is the guess n Fluid
                    # isnext is the guess n Solid
                    # y[n-1,:] is the n-1 Fluid solution
                    # s[n-1,:] is the n-1 Solid solution

                    # Fluid Equation
                    ynext = SolveFluid(N,iynext, isnext, y[n-1,:], s[n-1,:],NTU_val,dx,m[n])

                    # Solid Equation
                    snext = SolveSolid(N,ynext, isnext, y[n-1,:], s[n-1,:],NTU_val,U_val,dt,m[n])


                    stepTol = AbsTolFunc(ynext,iynext,maxStepTol) and AbsTolFunc(snext,isnext,maxStepTol)

                    s[n, :] = np.copy(snext)
                    isnext  = np.copy(snext)
                    y[n, :] = np.copy(ynext)
                    iynext  = np.copy(ynext)

                    stepCount = stepCount + 1
                    if (stepCount == maxSteps):
                        print("Hit max step count")

            # Check if current cycle is close to previous cycle.
            cycleTol = AbsTolFunc(y,iyCycle,maxCycleTol) and AbsTolFunc(s,isCycle,maxCycleTol)

            # Copy last value to the first of the next cycle.
            s[0, :] = np.copy(s[-1, :])
            y[0, :] = np.copy(y[-1, :])
            # Report current cycle count
            #print("Cycle Count: {0:d}".format(cycleCount))
            # Add Cycle
            cycleCount = cycleCount + 1
            # Did we hit the maximum number of cycles
            if (cycleCount == maxCycles):
                print("Hit max cycle count")
                break
            # Copy current cycle to the stored value
            isCycle = np.copy(s)
            iyCycle = np.copy(y)
            # Go do a another cycle
        # End Cycle
        t1 = time.time()
        #print("Run time: ", (t1 - t0) / 60, "[min]")


        # Fluid plot
        # plt.figure(num=None,figsize=(12,12))
        # fig = plt.subplot(221)
        #
        # plt.title("0-1")
        # plt.plot(s1, s[0, :], 'g--', label="0&1 Solid")
        # plt.plot(y1, y[0, :], 'b-', label="0&1 Fluid")
        # plt.legend(loc=2)
        # # Solid Plot
        # fig = plt.subplot(222)
        # plt.title("0.25")
        # plt.plot(s1, s[int(round((nt+1) / 4)), :], 'g--', label="0.25 Solid")
        # plt.plot(y1, y[int(round((nt+1) / 4)), :], 'b--', label="0.25 Fluid")
        # plt.legend(loc=2)
        # fig = plt.subplot(223)
        # plt.title("0.5")
        # plt.plot(s1, s[int(round((nt+1) / 2)), :], 'g--', label="0.5 Solid")
        # plt.plot(y1, y[int(round((nt+1) / 2)), :], 'b--', label="0.5 Fluid")
        # plt.legend(loc=2)
        # # Solid Plot
        # fig = plt.subplot(224)
        # plt.title("0.75")
        # plt.plot(s1, s[int(round((nt+1) * 3 / 4)), :], 'g--', label="0.75 Solid")
        # plt.plot(y1, y[int(round((nt+1) * 3 / 4)), :], 'b--', label="0.75 Fluid")
        # plt.legend(loc=2)
        #plt.show()

        qc = np.sum( dt*(y[:int(nt/2)-1, -1]+y[1:int(nt/2), -1] )*0.5) / (np.max(t[:int(nt/2)])-np.min(t[:int(nt/2)]))

        qcmax=np.sum(dt*(np.ones(int(nt/2))))/(np.max(t[:int(nt/2)+1])-np.min(t[:int(nt/2)]))
        print(qcmax)
        integral_eff1 = 1 - qc#/qcmax
        #print(np.max(t[:int(nt/2)+1]))
        integral_eff=np.trapz((1-y[:int(nt/2)+1, -1]),x=t[:int(nt/2)+1])/(np.max(t[:int(nt/2)+1])-np.min(t[:int(nt/2)+1]))

        integral_eff2=np.trapz((y[int(nt/2):, 0]),x=t[int(nt/2):])/(np.max(t[int(nt/2):])-np.min(t[int(nt/2):]))
        print('Effectiveness {} {} {}'.format(integral_eff1,integral_eff,integral_eff2))
        eff_vals[i,j]=integral_eff1

        halft=int(nt/2)
        integral_eff_HB_CE = np.sum( dt*((y[halft+1:,  0] + y[halft:-1 ,  0]   )*0.5)) / (np.max(t[halft:]) - np.min(t[halft:]))
        integral_eff_CB_HE = np.sum( dt*(   y[:halft  , -1] + y[1:halft+1 ,-1]   )*0.5)  / (np.max(t[:halft+1]) - np.min(t[:halft]))
        print('Effectiveness HB-CE {} CB-HE {}'.format(integral_eff_HB_CE,integral_eff_CB_HE))
        # plt.figure(num=None,figsize=(6,4.5))
        # plt.title("Dimentionless Temperature v Dimensionless time")
        # plt.plot(t, s[:,0], 'k--', label="Solid x=0")
        # plt.plot(t, y[:,0], 'b', label="Fluid x=0")
        # plt.plot(t, s[:,int(N/2)], 'r--', label="Solid x=L")
        # plt.plot(t, y[:,int(N/2)], 'c', label="Fluid x=L")
        # plt.plot(t, s[:,-1], 'r--', label="Solid x=L")
        # plt.plot(t, y[:,-1], 'c', label="Fluid x=L")
        # plt.legend(loc=0)
        # plt.grid(True)
        #plt.show()
print(eff_vals)
FileSave(fileName1, np.array_str(eff_vals))
