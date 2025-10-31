import os
import sys


import numpy as np
import time
from SimuTools import Sample, MagField, Simulation
from functioncache import check

os.chdir("src")  # go to /src folder
print(os.path.abspath(os.curdir))
sys.path.insert(0, os.path.abspath(os.curdir))
os.chdir("..")  # go back to parent folder

ExampleSample10MHzT = Sample(
    name="TestSample",  # name of the atom/molecule
    gamma=2
    * np.pi
    * (10)
    * 1e6,  # [Hz/T]. Remember input it like 2 * np.pi * 11.777*10**6
    numofnuclei=1,  #
    tempunit="K",  # temperature scale
    T2=100 / np.pi,  # [s]
    T1=1e9,  # [s]
    pol=1,
    verbose=False,
)

ALP_Field_grad = MagField(
    name="ALP field gradient"
)  # excitation field in the rotating frame
# excField.nu = 1e6 - 10  # [Hz]

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
    simuRate=(500),  #
    duration=300,
    excField=ALP_Field_grad,
    verbose=False,
)

tic = time.perf_counter()
# check(simu.demodfreq)
simu.excField.setALP_Field(
    method="inverse-FFT",
    timeStamp=simu.timeStamp,
    Bamp=1e-10,  # RMS amplitude of the pseudo-magnetic field in [T]
    nu_a=(-0.7),  # frequency in the rotating frame
    # direction: np.ndarray,  #  = np.array([1, 0, 0])
    use_stoch=True,
    demodfreq=simu.demodFreq_Hz,
    makeplot=True,
)
simu.excType = "ALP"
toc = time.perf_counter()
# print(f"setALP_Field() time consumption = {toc-tic:.3f} s")

tic = time.perf_counter()
simu.generateTrajectory(verbose=False)
toc = time.perf_counter()
# print(f"GenerateTrajectory time consumption = {toc-tic:.3f} s")

simu.monitorTrajectory(verbose=True)
# simu.VisualizeTrajectory3D(
#     plotrate=1e3,  # [Hz]
#     # rotframe=True,
#     verbose=False,
# )
Delta_nu_a = 1.2  # Hz
tau_a = 1 / (np.pi * Delta_nu_a)
T2 = simu.sample.T2
tau = np.sqrt(tau_a * T2)
# check(tau)
check(1 / (np.pi * T2))
check(1 / (np.pi * tau))
check(1 / (tau))
simu.compareBandSig()
