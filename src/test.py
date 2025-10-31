import os
import sys
print("1")
import numpy as np
# from SimuTools import MagField, Simulation
# from Sample import Sample
print("2")
# from Apparatus import SQUID, Magnet, CASPEr_LF, LockinAmplifier
print("3")
# from DataAnalysis import DualChanSig
from functioncache import GiveDateandTime
print("4")
# from Envelope import ureg
from Envelope import PhysicalQuantity as PQ

print("5")
# import matplotlib.pyplot as plt
# import matplotlib.gridspec as gridspec

from tqdm import tqdm
print("6")

# os.chdir("src")  # go to parent folder
# print(os.path.abspath(os.curdir))
# sys.path.insert(0, os.path.abspath(os.curdir))


# num_runs = 1
# simuRate = 500 / ureg.second  #
print("7")

B_values = np.array([PQ(v, "T") for v in np.linspace(0.1, 2.0, 10)])

print(B_values)
