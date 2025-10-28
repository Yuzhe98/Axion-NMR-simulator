##################################################
# for NMR data processing
##################################################


import numpy as np

# plotting

# curve fitting (including calculating uncertainties)

# for interpolation


# importing and processing hdf5 files

# monitor run time


np.random.seed(None)  # WARNING!
# BaselineRemoval will effect the randomness of the script.
# Better to set the random seed to None so as to restore the randomness


from Envelope import *


class SQUID:
    def __init__(self, name=None, Mf=None, Rf=None, attenuation=None):  # in Ohm
        """
        name : str
            name of the SQUID. default to 'PhiC6L1W'. 'PhiC73L1' is the other option
        Mf : float
            feedback sensitivity which can be found in the SQUID specifications.
            For the SQUID on channel 3 we usually use before 2024-05, M_f = 1 / (44.12e-6) \Phi_0 / A
            For the new SQUID on channel 2 we installed on 2024-05 (Sensor ID: C649_O12), M_f = 1 / (44.16e-6) \Phi_0 / A
        Rf : float
            feedback resistance in Ohm
        attenuation : float
            in dB
        """
        self.name = name
        self.Mf = Mf
        self.Rf = Rf
        self.attenuation = attenuation


class Pickup:
    def __init__(self, name=None, Mf=None, Rf=None, attenuation=None):  # in Ohm
        """
        name : str
            name of the SQUID. default to 'PhiC6L1W'. 'PhiC73L1' is the other option
        Mf : float
            feedback sensitivity which can be found in the SQUID specifications.
            For the SQUID on channel 3 we usually use before 2024-05, M_f = 1 / (44.12e-6) \Phi_0 / A
            For the new SQUID on channel 2 we installed on 2024-05 (Sensor ID: C649_O12), M_f = 1 / (44.16e-6) \Phi_0 / A
        Rf : float
            feedback resistance in Ohm

        attenuation : float
            in dB
        """
        self.name = name
        self.Mf = Mf
        self.Rf = Rf
        self.attenuation = attenuation
