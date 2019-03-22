
# Numpy library
import numpy as np



# Fluid
percGly = 20


# Radius information of the assembly

r1      = 16e-3 / 2  # Inner Radius [m]
r2      = 19.05e-3 / 2  # Outer Radius [m]
r3      = 22e-3 / 2  # Bore Radius [m]

Vvoid1  = 1.598e-6 # [m^3] 1 [cm^3]
Vvoid2  = 1169.86e-9 # [m^3]
Lvoid1  = 29.27e-3 #[m]
Lvoid2  = 51.435e-3 #[m]

L_add   = 18e-3
Lvoid   = 14.6e-3 + L_add
rv      = 1.9e-3  # Cold side void to the check valve radius [m]
Vvoid   = 458.24e-9+rv**2*3.14*L_add # [m^3]
rvs     = np.sqrt(Vvoid/(3.14*Lvoid))  # Cold side void radius [m]
rvs1    = np.sqrt(Vvoid1/(3.14*Lvoid1))  # Hot  side void radius [m]
rvs2    = np.sqrt(Vvoid2/(3.14*Lvoid2))  # Hot  side void radius [m]

################### WITH DV
# You can build any geometry based on glass spheres, regenerator(in this case Gd), and void space
species_discription = ['void','gs','reg-si2','void1','reg-si1','gs','void2']
print("the geometry looks like: {}".format(species_discription))

L_reg1 = 0.0225
L_reg2 = 0.0225

mRho  = 6100.  # [kg/m^3]
mK    = 6  # [W/(m K)]

# Locations 0<- cold hex, hot hex -> end
x_discription = [0,
                    L_add+14.6e-3,
                    L_add+76.1e-3+1.27e-3/2-Lvoid1/2-L_reg2,
                    L_add+76.1e-3+1.27e-3/2-Lvoid1/2,
                    L_add+76.1e-3+1.27e-3/2+Lvoid1/2,
                    L_add+76.1e-3+1.27e-3/2+Lvoid1/2+L_reg1,
                    L_add+76.1e-3+76.1e-3+1.27e-3-14.605e-3,
                    L_add+76.1e-3+76.1e-3+1.27e-3-14.605e-3+51.435e-3] #[m]

# Distance between base of the cold heat exchanger to magnet center
magOffset = L_add+ 76.1e-3+1.27e-3/2


# Surface Area
# Regenerator and glass spheres
Ac      = np.pi * r1 ** 2
# Regenerator and glass spheres
Pc      = 2 * np.pi * r1

# Set Material & Fluid Properties
# (BOROSILICATE) GLASS spheres
gsCp  = 800.  # [J/(kg K)]
gsRho = 2230.  # [kg/m^3]
gsK   = 1.2  # [W/(m K)]
# http://www.scientificglass.co.uk/contents/en-uk/d115_Physical_Properties_of_Borosilicate_Glass.html
# http://www.schott.com/borofloat/english/attribute/thermic/index.html
# http://www.schott.com/d/tubing/9a0f5126-6e35-43bd-bf2a-349912caf9f2/schott-algae-brochure-borosilicate.pdf


# Lead spheres
lsCp  = 830.  # [J/(kg K)]
lsRho = 2230.  # [kg/m^3]
lsK   = 1.005  # [W/(m K)]


# Ultem
kult  =  0.122   # [W/(m K)]
# https://www.plasticsintl.com/datasheets/ULTEM_GF30.pdf


# g10 material
kg10  =  0.608  # [W/(m K)]
# http://cryogenics.nist.gov/MPropsMAY/G-10%20CR%20Fiberglass%20Epoxy/G10CRFiberglassEpoxy_rev.htm


# air material
kair  = 0.0255  # [W/(m K)]
# Transport booklet


# Porosity of regenerator
er      = 0.53 # [-]
# Porosity of glass spheres
els     = 0.36  # [-]
# Porosity of glass spheres
egs     = 0.43  # [-]
# Diameter of regenerator
# Set as imput variable!
#Dsp     = Dsp  # [m]
# Diameter of glass spheres
Dspgs   = 0.003175  # [m]
# Diameter of glass spheres
Dspls   = 300e-6  # [m]
# Demagnetization coefficient
Nd      = 0.36  # [-]


# Casing BC assumption
CL_set="grad"
# Do we wish to use experimental modication of the BC
MOD_CL=0

ch_fac=0.5



