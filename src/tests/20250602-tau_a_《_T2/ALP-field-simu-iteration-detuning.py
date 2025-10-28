import os
import sys


os.chdir("src")  # go to parent folder
print(os.path.abspath(os.curdir))
sys.path.insert(0, os.path.abspath(os.curdir))


import numpy as np
from SimuTools import Sample, MagField, Simulation

# from DataAnalysis import DualChanSig
from functioncache import GiveDateandTime


# import matplotlib.pyplot as plt
# import matplotlib.gridspec as gridspec

from tqdm import tqdm


num_runs = 1000
simuRate = 500  #
duration = 100
timeLen = int(simuRate * duration)
nu_a_offsets = np.arange(-10, 10, 0.5)

results = np.empty(
    (num_runs, timeLen), dtype=np.float64
)  # or float, depending on your data


demodfreq = 1e6
T2 = 30
T1 = 1e9
Brms = 1e-10
nu_a = -0.7
use_stoch = True

savedir = (
    r"C:\Users\zhenf\D\Yu0702\Axion-NMR-simulator\Tests\20250602-tau_a_《_T2\data_0/"
)
timestr = GiveDateandTime()
# # Create DataFrame with time and data columns
# df = pd.DataFrame({"name": ["ALP field simulation"]})

# # Store metadata in DataFrame attributes (won't save in CSV, but can save in pickle)
# df.attrs["timeStamp"] = simu.timeStamp
# df.attrs["theta"] = [results]
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
    demodfreq=demodfreq,
    B0z=(1e6) / (ExampleSample10MHzT.gamma / (2 * np.pi)),  # [T]
    simuRate=simuRate,  #
    duration=duration,
    excField=ALP_Field_grad,
    verbose=False,
)

for j, nu_a_offset in enumerate((nu_a_offsets)):
    for i in tqdm(range(num_runs)):
        rand_seed = i

        # tic = time.perf_counter()
        # check(simu.demodfreq)
        simu.excField.setALP_Field(
            method="inverse-FFT",
            timeStamp=simu.timeStamp,
            Bamp=Brms,  # RMS amplitude of the pseudo-magnetic field in [T]
            nu_a=nu_a_offset,  # frequency in the rotating frame
            # direction: np.ndarray,  #  = np.array([1, 0, 0])
            use_stoch=use_stoch,
            demodfreq=simu.demodfreq,
            # rand_seed=rand_seed,
            makeplot=False,
        )
        simu.excType = "ALP"
        # toc = time.perf_counter()
        # print(f"setALP_Field() time consumption = {toc-tic:.3f} s")

        # tic = time.perf_counter()
        simu.GenerateTrajectory(verbose=False)
        # toc = time.perf_counter()
        # print(f"GenerateTrajectory time consumption = {toc-tic:.3f} s")

        # simu.MonitorTrajectory(verbose=True)
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
        sin_theta = np.sqrt(simu.trjry[0:-1, 0] ** 2 + simu.trjry[0:-1, 1] ** 2)

        # theta = run_simulation(rand_seed)
        results[i] = sin_theta
        # np.save(f"theta_run_{i}.npy", theta)

    #
    data_file_name = savedir + "theta_all_runs_" + timestr + f"_{j}.npz"
    np.savez(
        data_file_name,
        timeStamp=simu.timeStamp,
        sin_theta=results,
        simuRate=simuRate,
        duration=duration,
        demodfreq=demodfreq,
        T2=T2,
        T1=T1,
        Brms=Brms,
        nu_a=nu_a,
        use_stoch=True,
        gyroratio=ExampleSample10MHzT.gamma,
    )

    # np.save(savedir + f"theta_all_runs_" + timestr + ".npy", results)

    # # Sample data
    # theta = np.random.rand(1000)
    # simuRate = 1000
    # duration = 1.0

    # # Save DataFrame as pickle (preserves attrs)

    # data_file_name = savedir + f"theta_all_runs_" + timestr + ".pkl"
    # df.to_pickle(data_file_name)

    # with open(data_file_name, "wb") as f:
    #     pickle.dump({"df": df, "theta": results}, f)
