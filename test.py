import numpy as np
import time
from SimuTools import Sample, MagField, Simulation, TTL
from DataAnalysis import LIASignal
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

ALP_Field_grad = MagField(name="ALP field gradient")  # excitation field in the rotating frame
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
    demodfreq=1e6,
    B0z=(1e6) / (ExampleSample10MHzT.gyroratio / (2 * np.pi)),  # [T]
    simuRate=(6696.42871094),  #
    duration=10,
    excField=ALP_Field_grad,
    verbose=False,
)

tic = time.perf_counter()
check(simu.demodfreq)
simu.excField.setALP_Field(
    method='inverse-FFT',
        timeStamp=simu.timeStamp,
        Brms=1e-8,  # RMS amplitude of the pseudo-magnetic field in [T]
        nu_a=(5),  # frequency in the rotating frame
        # direction: np.ndarray,  #  = np.array([1, 0, 0])
        use_stoch=False,
        demodfreq=simu.demodfreq,
        makeplot=True)
simu.excType = "ALP"
toc = time.perf_counter()
print(f"setALP_Field() time consumption = {toc-tic:.3f} s")

# tic = time.perf_counter()
# simu.GenerateTrajectory(verbose=False)
# toc = time.perf_counter()
# print(f"GenerateTrajectory time consumption = {toc-tic:.3f} s")

# simu.MonitorTrajectory(plotrate=133, verbose=True)
# simu.VisualizeTrajectory3D(
#     plotrate=1e3,  # [Hz]
#     # rotframe=True,
#     verbose=False,
# )

# simu.analyzeTrajectory()

# specxaxis, spectrum, specxunit, specyunit = simu.trjryStream.GetSpectrum(
#     showtimedomain=True,
#     showfit=True,
#     showresidual=False,
#     showlegend=True,  # !!!!!show or not to show legend
#     spectype="PSD",  # in 'PSD', 'ASD', 'FLuxPSD', 'FluxASD'
#     ampunit="V",
#     specxunit="Hz",  # 'Hz' 'kHz' 'MHz' 'GHz' 'ppm' 'ppb'
#     specxlim=[simu.demodfreq - 0 , simu.demodfreq + 20],
#     # specylim=[0, 4e-23],
#     specyscale="linear",  # 'log', 'linear'
#     showstd=False,
#     showplt_opt=True,
#     return_opt=True,
# )