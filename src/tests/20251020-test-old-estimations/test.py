"""
High-field sensitivity
"""

# import os
# import sys
# os.chdir("..")  # if you want to go to parent folder
# os.chdir("..")  # if you want to go to parent folder
# print(os.path.abspath(os.curdir))
# sys.path.insert(0, os.path.abspath(os.curdir))


from functioncache import LF_2025, PhysicalQuantity
from math import pi

hf2025_SQD = LF_2025(
    B0_max=PhysicalQuantity(14.1, "T"),
    rho_E_DM=PhysicalQuantity(0.45, "GeV / cm**3"),
    va=PhysicalQuantity(270, "km / s"),
)

hf2025_SQD.getOmega_a()

hf2025_SQD.sampleLXe129_approx()

hf2025_SQD.getEfficPow(
    RBW_Hz=PhysicalQuantity(1, "Hz"), ALP_lw_Hz=PhysicalQuantity(10, "MHz")
)

Tmeas_list = []
freq_list = []
for freq_val in [150e6]:
    freq = PhysicalQuantity(freq_val, "Hz")
    freq_list.append(freq)
    Tmeas_list.append(pi * hf2025_SQD.Q_a / freq)

glim_list = hf2025_SQD.getXe129_Sensi_Phase2(freq_list, Tmeas_list, verbose=True)

# report = ''
# report += f'Transver magnetization:{}'
"""
Transver magnetization: 2.38e-07 A/m
Transver magnetization: 4.01e-04 A/m
F0: 150.00 MHz
T2:1000.00 s,
T2*:1.06e-03 s
T2/T2* ratio:9.42e+05, Reduced Q factor ratio (Qmu0/Qmu*): 3.00
M0 1.97e+00 A/m
Volume: 10.00 mL
Coupling gaNN: 1.10e-13 eV-1
DM density: 4.50e+14 eV/m^3
DM rms velocity: 2.70e+05 m/s
Rabi frequency: 1.98e-04 Hz
Axion Q factor:1.00e+06, axion coherence time: 2.12e-03 s

Transver magnetization (T2*) = -6.01e-10 ampere / meter
Transver magnetization (T2) = -5.66e-04 ampere / meter
F0 = 1.50e+08 hertz
T2 = 1000 second
T2* = 1.06e-03 second
T2/T2* ratio = 9.42e+05 dimensionless
M0 = -1.96e+00 ampere / meter
Volume = 1.00e+01 centimeter ** 3
Coupling gaNN = 1.10e-13 1 / electron_volt
DM density = 4.50e-01 gigaelectron_volt / centimeter ** 3
DM rms velocity = 2.70e+02 kilometer / second
Rabi frequency = 1.98e-04 hertz
Axion Q factor = 1.00e+06 dimensionless
axion coherence time = 2.12e-03 1 / hertz
"""
