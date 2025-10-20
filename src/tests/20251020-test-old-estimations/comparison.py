# compare estimations of axion signals with simulation results

import os
import sys
os.chdir("src")  # go to parent folder
# print(os.path.abspath(os.curdir))
sys.path.insert(0, os.path.abspath(os.curdir))

import numpy as np
import time
from SimuTools import Sample, MagField, Simulation, gate
# from DataAnalysis import DualChanSig
from functioncache import check, GiveDateandTime

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# what was in the 
# The average flux power of an ALP-inducedspin-precession signal is determined by ALP induced transverse magnetization that is described in Equations(8), (9), and(10): 
#  ⟨Φ^2_a⟩= 1/2 (𝛼 un 𝜇0 M0)^2 ⟨𝜉^2⟩= 1/2 (𝛼 un 𝜇0 M0)^2 Ωa^2 ⟨tau^2⟩ 
