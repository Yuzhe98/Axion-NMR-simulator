import os
import sys


os.chdir("src")  # go to parent folder
print(os.path.abspath(os.curdir))
sys.path.insert(0, os.path.abspath(os.curdir))


import numpy as np
from SimuTools import Sample, MagField, Simulation
from functioncache import GiveDateandTime


num_runs = 1
simuRate = 500  #
duration = 100
timeLen = int(simuRate * duration)

results = np.empty(
    (num_runs, timeLen), dtype=np.float64
)  # or float, depending on your data


demodfreq = 1e6
T2 = 30
T1 = 1e9
Brms = 1e-10
nu_a = -0.7
use_stoch = True

savedir = r"C:\Users\zhenf\D\Mainz\CASPEr\20250520-tau_a_《_T2/"
timestr = GiveDateandTime()
# # Create DataFrame with time and data columns
# df = pd.DataFrame({"name": ["ALP field simulation"]})

# # Store metadata in DataFrame attributes (won't save in CSV, but can save in pickle)
# df.attrs["timeStamp"] = simu.timeStamp
# df.attrs["m_transverse"] = [results]
# df.attrs["simuRate"] = simuRate
# df.attrs["duration"] = duration

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
    station=None,
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


for i in range(num_runs):
    rand_seed = i

    # tic = time.perf_counter()
    # check(simu.demodfreq)
    simu.excField.setALP_Field(
        method="inverse-FFT",
        timeStamp=simu.timeStamp,
        Bamp=Brms,  # RMS amplitude of the pseudo-magnetic field in [T]
        nu_a=nu_a,  # frequency in the rotating frame
        # direction: np.ndarray,  #  = np.array([1, 0, 0])
        use_stoch=use_stoch,
        demodFreq=simu.demodFreq_Hz,
        # rand_seed=rand_seed,
        makeplot=False,
    )
    simu.excType = "ALP"
    # toc = time.perf_counter()
    # print(f"setALP_Field() time consumption = {toc-tic:.3f} s")

    # tic = time.perf_counter()
    simu.generateTrajectory(verbose=False)
    # toc = time.perf_counter()
    # print(f"GenerateTrajectory time consumption = {toc-tic:.3f} s")

    simu.monitorTrajectory(verbose=True)
    # simu.VisualizeTrajectory3D(
    #     plotrate=1e3,  # [Hz]
    #     # rotframe=True,
    #     verbose=False,
    # )
    # Delta_nu_a = 1.2  # Hz
    # tau_a = 1 / (np.pi * Delta_nu_a)
    # T2 = simu.sample.T2
    # tau = np.sqrt(tau_a * T2)
    # # check(tau)
    # check(1 / (np.pi * T2))
    # check(1 / (np.pi * tau))
    # check(1 / (tau))
    # simu.compareBandSig()

    # normalized magnetization
    m_t = np.sqrt(simu.trjry[0:-1, 0] ** 2 + simu.trjry[0:-1, 1] ** 2)

    # m_transverse = run_simulation(rand_seed)
    # results[i] = m_t
    # np.save(f"m_transverse_run_{i}.npy", m_transverse)
