import numpy as np

def Deuterium3DPotential(x, y, z):
    # Returns 3D Deuterium potential at location (x, y, z) from the origin (assumed potential center at x,y,z = 0)
    # All distances are in fm, and all energies in MeV.
    eWells = 65.4823128982115
    eWell = 54.531
    eCores = 40.0*eWell
    rCore = 1.0/4
    rWell = 17.0/10
    fPow = 4.0
    r = np.sqrt(x**2 + y**2 + z**2)
    return eCores*np.exp(-(r/rCore)**fPow) - eWells*np.exp(-(r/rWell)**fPow)
