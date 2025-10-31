import os
import sys

# print(os.path.abspath(os.curdir))
# os.chdir("..")  # go to parent folder
# os.chdir("..")  # go to parent folder
# print(os.path.abspath(os.curdir))
sys.path.insert(0, os.path.abspath(os.curdir))

import numpy as np
import time
from SimuTools import Sample, MagField, Simulation
from functioncache import check

ExampleSample10MHzT = Sample(
    name="TestSample",  # name of the atom/molecule
    gamma=2
    * np.pi
    * (10)
    * 1e6,  # [Hz/T]. Remember input it like 2 * np.pi * 11.777*10**6
    numofnuclei=1,  #
    tempunit="K",  # temperature scale
    T2=1000,  # [s]
    T1=10000,  # [s]
    pol=1,
    verbose=False,
)

ALP_Field_grad = MagField(
    name="ALP field gradient"
)  # excitation field in the rotating frame

simu = Simulation(
    name="TestSample 10MHzT",
    sample=ExampleSample10MHzT,  # class Sample
    # gyroratio=(2*np.pi)*11.777*10**6,  # [Hz/T]
    init_time=0.0,  # [s]
    station=None,
    init_mag_amp=1.0,
    init_M_theta=0.0,  # [rad]
    init_M_phi=0.0,  # [rad]
    demodFreq=1e6,
    B0z=(1e6) / (ExampleSample10MHzT.gamma / (2 * np.pi)),  # [T]
    simuRate=(6696.42871094),  #
    duration=10,
    excField=ALP_Field_grad,
    verbose=False,
)

tic = time.perf_counter()
simu.excField.setALP_Field(
    method="time-interfer",
    timeStamp=simu.timeStamp,
    Bamp=1e-15,  # RMS amplitude of the pseudo-magnetic field in [T]
    nu_a=(-0.5),  # frequency in the rotating frame
    # direction: np.ndarray,  #  = np.array([1, 0, 0])
    use_stoch=False,
    demodFreq=simu.demodFreq_Hz,
    makeplot=False,
)
simu.excType = "ALP"
toc = time.perf_counter()
print(f"setALP_Field() time consumption = {toc-tic:.3f} s")

tic = time.perf_counter()
simu.generateTrajectory(verbose=False)
toc = time.perf_counter()
print(f"GenerateTrajectory time consumption = {toc-tic:.3f} s")

simu.monitorTrajectory(plotrate=133, verbose=True)
simu.visualizeTrajectory3D(
    plotrate=1e3,  # [Hz]
    # rotframe=True,
    verbose=False,
)


tau_a = 1e6 / np.abs(simu.excField.nu + simu.demodFreq_Hz)
check(tau_a)
check(1 / (np.pi * np.sqrt(simu.sample.T2 * tau_a)))
simu.analyzeTrajectory()
specxaxis, spectrum, specxunit, specyunit = simu.trjryStream.GetSpectrum(
    showtimedomain=True,
    showfit=True,
    showresidual=False,
    showlegend=True,  # !!!!!show or not to show legend
    spectype="PSD",  # in 'PSD', 'ASD', 'FLuxPSD', 'FluxASD'
    ampunit="V",
    specxunit="Hz",  # 'Hz' 'kHz' 'MHz' 'GHz' 'ppm' 'ppb'
    specxlim=[simu.demodFreq_Hz - 5, simu.demodFreq_Hz + 12],
    # specylim=[0, 4e-23],
    specyscale="linear",  # 'log', 'linear'
    showstd=False,
    showplt_opt=True,
    return_opt=True,
)
