"""
Low-field sensitivity
"""

# import os
# import sys
# os.chdir("..")  # if you want to go to parent folder
# os.chdir("..")  # if you want to go to parent folder
# print(os.path.abspath(os.curdir))
# sys.path.insert(0, os.path.abspath(os.curdir))
import numpy as np
from functioncache import LF_2025, PhysicalQuantity, check
from math import pi

lf2025 = LF_2025()
lf2025.getOmega_a()

lf2025.sampleMethanol()
# lf2025.sampleLXe129()
# lf2025.T2 = 1
# check(lf2025.T2)
# check((lf2025.B0_max))
Tmeas_list = []
freq_list = []
# for freq_val in 10 ** (np.arange(3, 7, 0.5)):
#     freq = PhysicalQuantity(freq_val, "Hz")
#     freq_list.append(freq)
#     Tmeas_list.append(100 * lf2025.Q_a / freq)

# freqmax = (lf2025.B0_max * lf2025.gamma / (2 * pi)).convert_to("Hz")
# for freq_val in [freqmax.value]:
#     freq = PhysicalQuantity(freq_val, "Hz")
#     freq_list.append(freq)
#     Tmeas_list.append(100 * lf2025.Q_a / freq)
for freq_val in [1.348e6]:
    freq = PhysicalQuantity(freq_val, "Hz")
    freq_list.append(freq)
    Tmeas_list.append(100 * lf2025.Q_a / freq)

lf2025.getThermMethanol_Sensi(
    [PhysicalQuantity(1.348, "MHz")], [PhysicalQuantity(100, "s")], verbose=False
)
# lf2025.getThermMethanol_Sensi(freq_list, Tmeas_list)
# lf2025.plotThermMethanol_Sensi(freq_list, Tmeas_list, verbose=False)
