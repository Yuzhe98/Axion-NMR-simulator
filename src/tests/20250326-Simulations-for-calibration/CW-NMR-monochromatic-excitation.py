import os
import sys

print(os.path.abspath(os.curdir))
# os.chdir("..")  # go to parent folder
# os.chdir("..")  # go to parent folder
print(os.path.abspath(os.curdir))
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
    * 10**6,  # [Hz/T]. Remember input it like 2 * np.pi * 11.777*10**6
    numofnuclei=1,  #
    tempunit="K",  # temperature scale
    T2=1 / (10 * np.pi),  # [s]
    T1=2 / (10 * np.pi),  # [s]
    pol=1,
    verbose=False,
)


excField = MagField(name="excitation field")  # excitation field in the rotating frame
excField.nu = 1e6  # [Hz]


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
    B0z=(1e6 - 0) / (ExampleSample10MHzT.gyroratio / (2 * np.pi)),  # [T]
    simuRate=(6696.42871094),  #
    duration=5,
    excField=excField,
    verbose=False,
)

simu.generatePulseExcitation(
    pulseDur=1.0 * simu.duration,
    tipAngle=np.pi,
    direction=np.array([1, 0, 0]),
    showplt=False,  # whether to plot B_ALP
    plotrate=None,
    verbose=False,
)

tic = time.perf_counter()
simu.GenerateTrajectory(verbose=False)
toc = time.perf_counter()
print(f"GenerateTrajectory time consumption = {toc-tic:.3f} s")

simu.MonitorTrajectory(plotrate=1000, verbose=True)
simu.VisualizeTrajectory3D(
    plotrate=1e3,  # [Hz]
    # rotframe=True,
    verbose=False,
)

processdata = True
if processdata:
    simu.analyzeTrajectory()
    specxaxis, spectrum, specxunit, specyunit = simu.trjryStream.GetSpectrum(
        # showtimedomain=True,
        showfit=True,
        spectype="PSD",  # in 'PSD', 'ASD', 'FLuxPSD', 'FluxASD'
        ampunit="V",
        specxlim=[simu.demodfreq - 20, simu.demodfreq + 12],
        return_opt=True,
    )
    simu.trjryStream.GetNoPulseFFT()
    simu.trjryStream.plotFFT()
