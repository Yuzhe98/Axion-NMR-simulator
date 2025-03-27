import numpy as np
import time

# sqdsensor = SQUID(name="Virtual SQUID", Mf=1.0, Rf=1.0)  # in Ohm
from SimuTools import Sample, MagField, Simulation, TTL

# from SimuTools import *
# from DataAnalysis import *
from functioncache import check

# B1_amp = np.array([1,2,4,5,6,7])
# # B1_direction = np.array([1,0,0])
# C = np.multiply(B1_amp, B1_amp)
# check(C)
excField_B_vec = np.array(([[-0.00000000e+00, -0.00000000e+00, -0.00000000e+00],
       [-3.28981986e-05, -0.00000000e+00, -0.00000000e+00],
       [-3.28981986e-05, -0.00000000e+00, -0.00000000e+00],
       [-6.57935009e-05, -0.00000000e+00, -0.00000000e+00],
       [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
       [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
       [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00]]))
B0z_rot_amp = 121
B0z_rot = B0z_rot_amp * np.ones(len(excField_B_vec))
B0_rot = np.outer(B0z_rot, np.array([0,0,1]))

check(B0_rot)
