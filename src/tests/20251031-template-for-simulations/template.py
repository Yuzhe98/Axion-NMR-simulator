import os
import sys

print(os.path.abspath(os.curdir))
import numpy as np
from src.SimuTools import MagField, Simulation
from src.Sample import Sample

from src.Apparatus import SQUID, Magnet, CASPEr_LF, LockinAmplifier

# from DataAnalysis import DualChanSig
from src.functioncache import GiveDateandTime

# from Envelope import ureg
from src.Envelope import PhysicalQuantity, gamma_p, gamma_Xe129, mu_p, mu_Xe129

# import matplotlib.pyplot as plt
# import matplotlib.gridspec as gridspec

from tqdm import tqdm


# os.chdir("src")  # go to parent folder
# print(os.path.abspath(os.curdir))
# sys.path.insert(0, os.path.abspath(os.curdir))


num_runs = 1
simuRate = PhysicalQuantity(500, "Hz")  #
duration = PhysicalQuantity(100, "s")
timeLen = int((simuRate * duration).convert_to("").value)
# nu_a_offsets = np.array(
#     [PhysicalQuantity(Delta_nu, "Hz") for Delta_nu in np.arange(-10, 10, 0.5)]
# )

nu_a_offsets = np.array([PhysicalQuantity(Delta_nu, "Hz") for Delta_nu in [0.0]])

results = np.empty(
    (num_runs, timeLen), dtype=np.float64
)  # or float, depending on your data


demodFreq = 1e6
LIA = LockinAmplifier(
    name="virtual LIA",
    demodFreq=PhysicalQuantity(1.0, "MHz"),
    sampRate=None,
    DTRC_Tc=None,
    DTRC_order=None,
    verbose=False,
)


# T2 = 30
# T1 = 1e9

# CH3CH2OH
sample = Sample(
    name="Ethanol",  # name of the sample
    gamma=gamma_p,  # [Hz/T]. Remember input it with 2 * np.pi
    massDensity=PhysicalQuantity(0.78945, "g / cm**3 "),
    molarMass=PhysicalQuantity(46.069, "g / mol"),  # molar mass
    numOfSpinsPerMolecule=6,  # number of spins per molecule
    T2=PhysicalQuantity(1, "s"),  #
    T1=PhysicalQuantity(5, "s"),  #
    vol=PhysicalQuantity(1, "cm**3"),
    mu=mu_p,  # magnetic dipole moment
    # boilpt=351.38,  # [K]
    # meltpt=159.01,  # [K]
    verbose=False,
)

magnet_det = Magnet(
    name=None,
    B0=LIA.demodFreq / (sample.gamma / (2 * np.pi)),
    lw_ppm=PhysicalQuantity(10, "ppm"),
)

Brms = 1e-10
nu_a = -0.7
use_stoch = True

savedir = r"src\tests\20251031-template-for-simulations\data/"
timestr = GiveDateandTime()


ALP_Field_grad = MagField(
    name="ALP field gradient"
)  # excitation field in the rotating frame
# excField.nu = 1e6 - 10  # [Hz]

simu = Simulation(
    name="simulation template",
    sample=sample,  # class Sample
    pickup=None,
    SQUID=None,
    magnet_pol=None,
    magnet_det=magnet_det,
    LIA=LIA,
    init_time=0.0,  # [s]
    station=None,
    init_mag_amp=1.0,
    init_M_theta=0.0,  # [rad]
    init_M_phi=0.0,  # [rad]
    simuRate=simuRate,  #
    duration=duration,
    excField=ALP_Field_grad,
    verbose=False,
)

for j, nu_a_offset in enumerate((nu_a_offsets)):
    for i in tqdm(range(num_runs)):
        rand_seed = i

        # tic = time.perf_counter()
        simu.excField.setALP_Field(
            method="inverse-FFT",
            timeStamp=simu.timeStamp,
            simuRate=simu.simuRate_Hz,
            duration=simu.duration_s,
            Bamp=Brms,  # RMS amplitude of the pseudo-magnetic field in [T]
            nu_a=nu_a_offset.value_in("Hz"),  # frequency in the rotating frame
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
        sin_theta = np.sqrt(simu.trjry[0:-1, 0] ** 2 + simu.trjry[0:-1, 1] ** 2)

        # theta = run_simulation(rand_seed)
        results[i] = sin_theta
        # np.save(f"theta_run_{i}.npy", theta)

    #
    data_file_name = savedir + "theta_all_runs_" + timestr + f"_{j}.npz"
    # np.savez(
    #     data_file_name,
    #     timeStamp=simu.timeStamp,
    #     sin_theta=results,
    #     simuRate=simuRate,
    #     duration=duration,
    #     demodfreq=demodFreq,
    #     T2=T2,
    #     T1=T1,
    #     Brms=Brms,
    #     nu_a=nu_a,
    #     use_stoch=True,
    #     gyroratio=ExampleSample10MHzT.gamma,
    # )

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
