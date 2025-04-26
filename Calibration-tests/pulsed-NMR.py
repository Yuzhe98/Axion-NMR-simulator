import os
import sys

# print(os.path.abspath(os.curdir))
# os.chdir("..")  # go to parent folder
# os.chdir("..")  # go to parent folder
# print(os.path.abspath(os.curdir))
sys.path.insert(0, os.path.abspath(os.curdir))

import numpy as np
import time
from SimuTools import Sample, MagField, Simulation, gate
from DataAnalysis import DualChanSig
from functioncache import check

ExampleSample10MHzT = Sample(
    name="TestSample",  # name of the atom/molecule
    gyroratio=2
    * np.pi
    * (10)
    * 1e6,  # [Hz/T]. Remember input it like 2 * np.pi * 11.777*10**6
    numofnuclei=1,  #
    tempunit="K",  # temperature scale
    T2=1 / np.pi,  # [s]
    T1=1000,  # [s]
    pol=1,
    verbose=False,
)


excField = MagField(name="excitation field")  # excitation field in the rotating frame
excField.nu = 1e6 - 10  # [Hz]

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
    simuRate=(1e3),  #
    duration=2,
    excField=excField,
    verbose=False,
)

simu.generatePulseExcitation(
    pulseDur=5.0 * simu.timeStep,
    tipAngle=np.pi / 2,
    direction=np.array([1, 0, 0]),  # along x-axis
    showplt=False,  # whether to plot B_ALP
    plotrate=None,
    verbose=False,
)
# check(simu.excField.B_vec)
# check(simu.excField.dBdt_vec)

tic = time.perf_counter()
simu.GenerateTrajectory(verbose=False)
toc = time.perf_counter()
print(f"GenerateTrajectory time consumption = {toc-tic:.3f} s")

# simu.MonitorTrajectory(plotrate=None, verbose=True)
# simu.VisualizeTrajectory3D(
#     plotrate=None,  # [Hz]
#     # rotframe=True,
#     verbose=False,
# )

simu.analyzeTrajectory()

specxaxis, spectrum, specxunit, specyunit = simu.trjryStream.GetSpectrum(
    showtimedomain=True,
    showfit=True,
    showresidual=False,
    showlegend=True,  # !!!!!show or not to show legend
    spectype="PSD",  # in 'PSD', 'ASD'
    ampunit="V",
    specxunit="Hz",  # 'Hz' 'kHz' 'MHz' 'GHz' 'ppm' 'ppb'
    specxlim=[simu.demodfreq - 0, simu.demodfreq + 20],
    # specylim=[0, 4e-23],
    specyscale="linear",  # 'log', 'linear'
    showstd=False,
    showplt_opt=True,
    return_opt=True,
)
