import os
import sys

print(os.path.abspath(os.curdir))
# os.chdir("..")  # go to parent folder
# os.chdir("..")  # go to parent folder

print(os.path.abspath(os.curdir))
sys.path.insert(0, os.path.abspath(os.curdir))

import numpy as np
import time

# sqdsensor = SQUID(name="Virtual SQUID", Mf=1.0, Rf=1.0)  # in Ohm
from SimuTools import Sample, MagField, Simulation, TTL

# from SimuTools import *
# from DataAnalysis import *
from functioncache import check

ExampleSample10MHzT = Sample(
    name="TestSample",  # name of the atom/molecule
    gyroratio=2
    * np.pi
    * (10)
    * 10**6,  # [Hz/T]. Remember input it like 2 * np.pi * 11.777*10**6
    numofnuclei=1,  #
    tempunit="K",  # temperature scale
    boilpt=165.051,  # [K]
    meltpt=161.40,  # [K]
    density_liquid=2.942,  # [g/cm^3] at boiling point
    density_gas=5.894 * 10**3,  # [g/cm^3] at STP
    density_solid=None,  # [g/cm^3]
    molarmass=131.2930,  # [g/mol]
    spindenisty_liquid=None,  # [mol/cm^3]
    spindenisty_gas=None,  # [g/cm^3] at STP
    spindenisty_solid=None,  # [mol/cm^3]
    shareofpeaks=[1.0],  # array or list.
    T2=1,  # [s]
    T1=1000,  # [s]
    pol=1,
    verbose=False,
)


excField = MagField(name="excitation field")  # excitation field in the rotating frame
excField.nu = 1e6-10  # [Hz]


simu = Simulation(
    name="TestSample 10MHzT",
    sample=ExampleSample10MHzT,  # class Sample
    # gyroratio=(2*np.pi)*11.777*10**6,  # [Hz/T]
    init_time=0.0,  # [s]
    station=None,
    init_mag_amp=1.0,
    init_M_theta=0.0,  # [rad]
    init_M_phi=0.0,  # [rad]
    demodfreq=1e6,
    B0z=(1e6 - 10) / (ExampleSample10MHzT.gyroratio / (2 * np.pi)),  # [T]
    simuRate=(6696.42871094),  # 
    duration=5,
    excField=excField,
    verbose=False,
)

simu.generatePulseExcitation(
    pulseDur=5.*simu.timeStep,
    tipAngle=np.pi / 2,
    direction=np.array([1, 0, 0]),
    showplt=False,  # whether to plot B_ALP
    plotrate=None,
    verbose=False,
)
check(simu.excField.B_vec)
check(simu.excField.dBdt_vec)

tic = time.perf_counter()
simu.GenerateTrajectory(verbose=False)
toc = time.perf_counter()
print(f'GenerateTrajectory time consumption = {toc-tic:.3f} s')

simu.MonitorTrajectory(plotrate=133, verbose=True)
simu.VisualizeTrajectory3D(
    plotrate=1e3,  # [Hz]
    # rotframe=True,
    verbose=False,
)
