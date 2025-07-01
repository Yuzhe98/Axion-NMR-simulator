import os
from random import sample
import sys

print(os.path.abspath(os.curdir))
# os.chdir("..")  # go to parent folder
# os.chdir("..")  # go to parent folder

print(os.path.abspath(os.curdir))
sys.path.insert(0, os.path.abspath(os.curdir))
from SimuTools import liquid_Xe129, Methanol, Mainz, TestSample10MHzT, TestStation
from SimuTools import *
from DataAnalysis import *

sqdsensor = SQUID(name="Virtual SQUID", Mf=1.0, Rf=1.0)  # in Ohm
listofGammaandSAmp = []
listofT2andavgMtsq = []
listofspectrum = []
gaNNorder_arr = np.arange(-2.0, -11.0, -0.5)
ALPnuorder_arr = np.arange(3.0, 7.0, 1)
Mt_list = []
T2star_list = [
    0.01,
    0.1,
    1,
    10,
]  #  30.0, 40.0, 50.0, 100.0 0.1,  [2,10,50,100], 10**(1.5), 100, 10**(2.5), 1000
TestSample = TestSample10MHzT
for ALPnuorder in [6]:
    check(ALPnuorder)
    for T2star in T2star_list:
        # [0.1, 1, 10**(0.5), 10, 10**(1.5), 100, 10**(2.5), 1000]
        #
        # check(T2star)
        for gaNNorder in [-7]:
            # check(gaNNorder)
            axionwind = AxionWind(
                name="ALP",
                nu=10**ALPnuorder,  # compton frequency in [Hz]
                Gamma=1.0 * 10.0 ** (-6),  # spectrum linewidth.
                gaNN=10**gaNNorder,  # in Gev^-1
                direction_solar=np.array([1, 90 * np.pi / 180.0, -90.0 * np.pi / 180]),
                # Sun is moveing towards phi=90 deg, so axion direction phi = -90 deg
                verbose=False,
            )
            axionwind.BALP = 100 * axionwind.gaNN / abs(TestSample.gyroratio)
            magnetization = Simulation(
                name="TestSample10MHzT",
                sample=TestSample,  # class Sample
                gyroratio=TestSample.gyroratio,  # [Hz/T]
                init_time=0.0,  # [s]
                station=TestStation,
                init_mag_amp=1.0,
                init_M_theta=0.0,  # [rad]
                init_M_phi=0.0,  # [rad]
                B0z=(10**ALPnuorder + 0 * 1.34)
                / TestSample.gyroratio
                * 2
                * np.pi,  # [T]
                simuRate=(6696.42871094),  # max(10000, a
                excField=axionwind,
                T2=1.0 * T2star,  # 1.0/(np.pi*samplelinewidth)
                T1=100000.0,
                verbose=False,
            )
            # tic = time.perf_counter()
            magnetization.GenerateParam(
                numofcohT=5,  # max(10 * T2star, 1)
                excType="ThermalLight",  #'ThermalLight' 'RandomJump' 'InfCoherence'
                showplt=False,  # whether to plot B_ALP
                plotrate=0.1,
                verbose=False,
            )
            # toc = time.perf_counter()
            # print(f'GenerateParam time consumption = {toc-tic:.3f} s')
            # simurate 1000, numofcohT=100/axionwind.cohT, not usejit. time = 2.3 s
            # simurate 1000, numofcohT=100/axionwind.cohT, usejit @jit. time = 1.8 s
            # simurate 1000, numofcohT=100/axionwind.cohT, usejit @jit(types blabla). time = 0.1 s
            # simurate 1000, numofcohT=1000/axionwind.cohT, usejit @jit(types blabla). time = 1.1 s
            # simurate 1000, numofcohT=100000/axionwind.cohT, usejit @jit(types blabla). time = 107.1 s

            tic = time.perf_counter()
            magnetization.GenerateTrajectory(verbose=False)
            toc = time.perf_counter()
            # print(f'GenerateTrajectory time consumption = {toc-tic:.3f} s')
            # simurate 1000, numofcohT=100/axionwind.cohT, not usejit. time = 4.2 s
            # simurate 1000, numofcohT=100/axionwind.cohT, usejit @jit. time = 1.6 s
            # simurate 1000, numofcohT=100/axionwind.cohT, usejit @jit(types blabla). time = 0.208 s
            # simurate 1000, numofcohT=1000/axionwind.cohT, usejit @jit(types blabla). time = 2.069 s

            # magnetization.MonitorTrajectory(plotrate=10**3,verbose=True)
            # magnetization.VisualizeTrajectory3D(
            #         plotrate=10**3,  # [Hz]
            #         # rotframe=True,
            #         verbose=False,
            #     )
            # magnetization.SaveTrajectory(
            # 		h5fpathandname=f'K:/CASPEr data/20220522_NMRKineticSimu_data_test2/sample_IDEN/'+\
            # 						f'simudata_test2_ALPwind.nu_1e{ALPnuorder:.1f}_gaNN_1e{gaNNorder:.1f}_samplelinewidth_{samplelinewidth:g}_'+\
            # 							f'T1_{magnetization.T1:g}_T2_{magnetization.T2:.3g}',
            # 		saveintv=1,  # int
            # 		verbose=False
            # )
            print(
                f"***************************************************************************"
            )
            print(f"T2* = {magnetization.T2:e}")
            magnetization.StatTrajectory(verbose=True)
            print(
                f"**************************************************************************"
            )
            Mt_list.append(np.sqrt(magnetization.avgMxsq + magnetization.avgMysq))
            processdata = False
            if processdata:
                liastream = DualChanSig(
                    name="LIA data",
                    device="LIA",
                    device_id="dev4434",
                    file=f"T2star {T2star:g}",
                    verbose=True,
                )
                liastream.attenuation = 0
                liastream.filterstatus = "off"
                liastream.filter_TC = 0.0
                liastream.filter_order = 0
                liastream.dmodfreq = (
                    magnetization.B0z * TestSample.gyroratio / (2 * np.pi)
                )
                saveintv = 1
                liastream.samprate = magnetization.simuRate / saveintv
                # check(magnetization.timestamp.shape)
                # check(magnetization.trjry[0:-1:saveintv, 0].shape)

                liastream.dataX = (
                    0.5
                    * magnetization.trjry[
                        int(0 * magnetization.simuRate) : -1 : saveintv, 0
                    ]
                )  # * \
                # np.cos(2 * np.pi * magnetization.nu_rot * magnetization.timestamp[0:-1:saveintv])
                liastream.dataY = (
                    0.5
                    * magnetization.trjry[
                        int(0 * magnetization.simuRate) : -1 : saveintv, 1
                    ]
                )  # * \
                # np.sin(2 * np.pi * magnetization.nu_rot * magnetization.timestamp[0:-1:saveintv])

                # liastream.dataX = 0.5 * 1 * \
                # 	np.cos(2 * np.pi * magnetization.nu_rot * magnetization.timestamp[0:-1:saveintv])
                # liastream.dataY = 0.5 * 1 * \
                # 	np.sin(2 * np.pi * magnetization.nu_rot * magnetization.timestamp[0:-1:saveintv])

                liastream.GetSpinNoisePSD(
                    chunksize=2,  # magnetization.T2
                    analysisrange=[
                        0,
                        -1,
                    ],  # [0, int(9*samplelinewidth*liastream.samprate)]
                    interestingfreq_list=[],
                    # ploycorrparas=ployparas,
                    ploycorrparas=[],
                    showstd=False,
                    # stddev_range=[1.349150e6,1.349750e6],
                    verbose=False,
                )
                liastream.FitPSD(
                    fitfunction="dualLorentzian",  # 'Lorentzian' 'dualLorentzian' 'tribLorentzian' 'Gaussian 'dualGaussian' 'auto' 'Polyeven'
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
                    Mf=sqdsensor.Mf,
                    Rf=sqdsensor.Rf,
                    specxunit="Hz",  # 'Hz' 'kHz' 'MHz' 'GHz' 'ppm' 'ppb'
                    specxlim=[axionwind.nu - 10, axionwind.nu + 10],
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
                print(
                    f"linewidth = {1.0 / (np.pi * T2star):g} Hz, np.amax(spectrum) = {np.amax(spectrum):.2e}"
                )
                listofGammaandSAmp.append([T2star, np.amax(spectrum)])
                listofT2andavgMtsq.append(
                    [magnetization.T2, magnetization.avgMxsq + magnetization.avgMysq]
                )  # , np.sum(spectrum), np.amax(spectrum)
                print(
                    f"T2star = {T2star:g} , avg Mt sq = {magnetization.avgMxsq + magnetization.avgMysq:.2e}"
                )
