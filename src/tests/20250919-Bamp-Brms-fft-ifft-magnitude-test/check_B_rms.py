import os
import sys


import numpy as np
from SimuTools import Sample, MagField, Simulation

# from DataAnalysis import DualChanSig
from functioncache import check, GiveDateandTime

os.chdir("src")  # go to parent folder
# print(os.path.abspath(os.curdir))
sys.path.insert(0, os.path.abspath(os.curdir))


num_runs = 100
simuRate = 100  #
duration = 500
timeLen = int(simuRate * duration)

results = np.empty(
    (num_runs, timeLen), dtype=np.float64
)  # or float, depending on your data


demodfreq = 1e6
T2 = 30
T1 = 1e9
Bamp = 1e-10
nu_a = -0.7
use_stoch = True

savedir = ""
timestr = GiveDateandTime()

ExampleSample10MHzT = Sample(
    name="TestSample",  # name of the atom/molecule
    gamma=2
    * np.pi
    * (10)
    * 1e6,  # [Hz/T]. Remember input it like 2 * np.pi * 11.777*10**6
    T2=T2,  # [s]
    T1=T1,  # [s]
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
    init_mag_amp=1.0,
    init_M_theta=0.0,  # [rad]
    init_M_phi=0.0,  # [rad]
    demodFreq=demodfreq,
    B0z=(1e6) / (ExampleSample10MHzT.gamma / (2 * np.pi)),  # [T]
    simuRate=simuRate,  #
    duration=duration,
    excField=ALP_Field_grad,
    verbose=False,
)

B_rms_from_simu_arr = []

for i in range(num_runs):
    # rand_seed = i

    # tic = time.perf_counter()
    # check(simu.demodfreq)
    simu.excField.setALP_Field(
        method="inverse-FFT",
        timeStamp=simu.timeStamp,
        simuRate=simuRate,
        duration=duration,
        Bamp=Bamp,  # RMS amplitude of the pseudo-magnetic field in [T]
        nu_a=nu_a,  # frequency in the rotating frame
        # direction: np.ndarray,  #  = np.array([1, 0, 0])
        use_stoch=use_stoch,
        demodFreq=simu.demodFreq_Hz,
        # rand_seed=rand_seed,
        makeplot=False,
    )
    simu.excType = "ALP"
    # B_rms_from_simu = (
    #     np.mean(simu.excField.B_vec[:, 0] ** 2 + simu.excField.B_vec[:, 1] ** 2) ** 0.5
    # )
    B_rms_from_simu = np.mean(np.abs(simu.excField.B_vec[:, 1]) ** 2) ** 0.5
    # check(B_rms_from_simu)
    B_rms_from_simu_arr.append(B_rms_from_simu)

B_rms_from_simu_arr = np.array(B_rms_from_simu_arr)
check(np.mean(B_rms_from_simu_arr))
check(np.std(B_rms_from_simu_arr))
