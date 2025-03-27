import os
import sys
print(os.path.abspath(os.curdir))
# os.chdir("..")  # go to parent folder
# os.chdir("..")  # go to parent folder
print(os.path.abspath(os.curdir))
sys.path.insert(0, os.path.abspath(os.curdir))

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
    simuRate=(6696.42871094),  #
    duration=5,
    excField=excField,
    verbose=False,
)

simu.generatePulseExcitation(
    pulseDur=5.0 * simu.timeStep,
    tipAngle=np.pi / 2,
    direction=np.array([1, 0, 0]),
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

simu.MonitorTrajectory(plotrate=133, verbose=True)
simu.VisualizeTrajectory3D(
    plotrate=1e3,  # [Hz]
    # rotframe=True,
    verbose=False,
)

processdata = True
if processdata:
    liastream = LIASignal(
        name="Simulation data",
        device="Simulation",
        device_id="Simulation",
        file=f"Simulation",
        verbose=True,
    )
    liastream.attenuation = 0
    liastream.filterstatus = "off"
    liastream.filter_TC = 0.0
    liastream.filter_order = 0
    liastream.dmodfreq = simu.demodfreq
    saveintv = 1
    liastream.samprate = simu.simuRate / saveintv
    # check(simu.timestamp.shape)
    # check(simu.trjry[0:-1:saveintv, 0].shape)

    liastream.dataX = 1 * simu.trjry[int(0 * simu.simuRate) : -1 : saveintv, 0]  # * \
    # np.cos(2 * np.pi * simu.nu_rot * simu.timestamp[0:-1:saveintv])
    liastream.dataY = 1 * simu.trjry[int(0 * simu.simuRate) : -1 : saveintv, 1]  # * \
    # np.sin(2 * np.pi * simu.nu_rot * simu.timestamp[0:-1:saveintv])

    # liastream.dataX = 0.5 * 1 * \
    # 	np.cos(2 * np.pi * simu.nu_rot * simu.timestamp[0:-1:saveintv])
    # liastream.dataY = 0.5 * 1 * \
    # 	np.sin(2 * np.pi * simu.nu_rot * simu.timestamp[0:-1:saveintv])

    liastream.GetNoPulsePSD(
        windowfunction='rectangle',
        # decayfactor=-10,
        chunksize=None,  # sec
        analysisrange = [0,-1],
        getstd=False,
        stddev_range=None,
        # polycorrparas=[],
        # interestingfreq_list=[],
        selectshots=[],
        verbose=False
    )
    liastream.FitPSD(
        fitfunction="Lorentzian",  # 'Lorentzian' 'dualLorentzian' 'tribLorentzian' 'Gaussian 'dualGaussian' 'auto' 'Polyeven'
        inputfitparas=["auto", "auto", "auto", "auto"],
        smooth=False,
        smoothlevel=1,
        fitrange=["auto", "auto"],
        alpha=0.05,
        getresidual=False,
        getchisq=False,
        verbose=False,
    )
    specxaxis, spectrum, specxunit, specyunit = liastream.GetSpectrum(
        showtimedomain=False,
        showacqdata=True,
        showfreqdomain=True,
        showfit=True,
        showresidual=False,
        showlegend=True,  # !!!!!show or not to show legend
        spectype="PSD",  # in 'PSD', 'ASD', 'FLuxPSD', 'FluxASD'
        ampunit="V",
        # Mf=sqdsensor.Mf,
        # Rf=sqdsensor.Rf,
        specxunit="Hz",  # 'Hz' 'kHz' 'MHz' 'GHz' 'ppm' 'ppb'
        # specxlim=[axionwind.nu - 10, axionwind.nu + 10],
        # specylim=[0, 4e-23],
        # specxunit2 = 'ppm',
        # referfreq=liastream.dmodfreq,
        # specx2format = '{:.0f}',
        specyscale="linear",  # 'log', 'linear'
        showstd=False,
        figsize=(10, 6),
        top=0.85,
        bottom=0.03,
        left=0.11,
        right=0.98,
        hspace=0.73,
        wspace=0.2,
        showplt_opt=True,
        return_opt=True,
        verbose=False,
    )
    # print(
    #     f"linewidth = {1.0 / (np.pi * T2star):g} Hz, np.amax(spectrum) = {np.amax(spectrum):.2e}"
    # )
    # listofGammaandSAmp.append([T2star, np.amax(spectrum)])
    # listofT2andavgMtsq.append(
    #     [simu.T2, simu.avgMxsq + simu.avgMysq]
    # )  # , np.sum(spectrum), np.amax(spectrum)
    # print(
    #     f"T2star = {T2star:g} , avg Mt sq = {simu.avgMxsq + simu.avgMysq:.2e}"
    # )
