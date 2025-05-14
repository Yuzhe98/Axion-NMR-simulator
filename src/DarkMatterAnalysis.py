################################################

# import packages
import sys, os, glob
import time

# importing and processing hdf5 files
import h5py

# plotting
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec  # for creating subplots
from matplotlib.ticker import ScalarFormatter, FormatStrFormatter, MultipleLocator

# basic computations
import numpy as np
import math

from scipy.stats import norm, chi2, gamma, rayleigh, uniform, expon
from scipy.optimize import curve_fit, minimize

# curve fitting (including calculating uncertainties)
from scipy.signal import find_peaks
from scipy.io import loadmat, savemat
from sympy import true

from DataAnalysis import SQUID, DualChanSig
from functioncache import (
    Lorentzian,
    MovAvgByStep,
    ReturnMeasureTimFromFile,
    ReturnNMRdata,
    ReturnT2time,
    axion_lineshape,
    PlotGaussianLine,
    angle_between,
    record_runtime_YorN,
    GetDateTimeSimple,
    merge_close_entries2,
    find_matching_tuples,
    read_double_precision_floats,
    check,
    extract_number,
    clear_lines,
    print_progress_bar,
    get_random_color,
)

# from KeaControl import Kea
import pickle

import pandas as pd
from astropy.time import Time
import TASSLE.tassle.axion_wind as wind

from multiprocessing import Pool

# Set RECORD_RUNTIME to False when you don't want to record runtime
RECORD_RUNTIME = True


class dmScanStep:
    def __init__(
        self,
        name,
        NoPulse_dataFile_list: list = None,
        pNMR_dataFile_list=None,
        CPMG_dataFile_list=None,
        station=None,
        verbose=False,
    ):
        """
        Initialize singleDMmeas class for a single dark-matter measurement by CASPEr-gradient-lowfield
        The design of this object should be and is capable of dealing with large data file,
        which is important to the earth- / solar- dark-matter halo data analysis.

        """
        self.name = name
        self.NoPulse_datafile_list = NoPulse_dataFile_list
        self.pNMR_dataFile_list = pNMR_dataFile_list
        self.CPMG_dataFile_list = CPMG_dataFile_list
        self.LoadMeasInfo()

        self.station = station
        self.DS_func = None
        self.T2 = None
        self.axion = None

    def LoadMeasInfo(self, device_id="dev4434", demod_index=0, verbose=False):
        """
        Load measurement information, including
        No-pulse stream, pulsed-NMR stream, and CPMG stream.
        'Stream' here means data stream.
        TODO add Get measurement duration self.MeasDur

        info: dict
        level-1
        ['NoPulseStream']: dict
        ['pNMR_Stream']: dict
        ['CPMG_Stream']: dict

        level-2
        ['NoPulseStream']['filelist']: list
            NoPulse_dataFile_list
        ['NoPulseStream']['MeasTime']: dict
            ['period']: list, optional
                start and end of the the scan step in str
            ['duration']: float, optional
                duration of the the scan step in [s]
        ['NoPulseStream']['dmodfreq']: float, optional
            demodulator frequency of the scan step in [Hz]

        ['pNMR_Stream']['filelist']: list
            pNMR_dataFile_list
        ['NoPulseStream']['MeasTime']: dict
            ['period']: list, optional
                start and end of the p-NMR in str
            ['duration']: float, optional
                duration of the p-NMR in [s]
        ['NoPulseStream']['FreqRange']: list, optional
            frequency range of the scannings in [Hz]

                =None,
        CPMG_dataFile_list=None,


        """
        self.info = []
        self.nps = DualChanSig(
            name="No-Pulse Stream",
            device="LIA",
            device_id=device_id,
            filelist=self.NoPulse_datafile_list,
            verbose=verbose,
        )
        self.nps.SortFiles(verbose=False)

        self.pNMRs = DualChanSig(
            name="Pulsed-NMR Stream",
            device="LIA",
            device_id=device_id,
            filelist=self.pNMR_dataFile_list,
            verbose=verbose,
        )
        self.pNMRs.SortFiles(verbose=False)

        self.CPMGs = DualChanSig(
            name="Pulsed-NMR Stream",
            device="LIA",
            device_id=device_id,
            filelist=self.CPMG_dataFile_list,
            verbose=verbose,
        )
        self.CPMGs.SortFiles(verbose=False)

        def LoadStreamInfo(Stream: DualChanSig, file_list: list):
            if file_list is None:
                file_list = []
            for singlefile in file_list:
                if verbose:
                    print("Loading data from " + singlefile)
                if singlefile[-3:] == ".h5":
                    with h5py.File(
                        singlefile, "r", driver="core"
                    ) as dataFile:  # h5py loading method
                        # check recording method
                        if verbose:
                            check(dataFile.keys())
                        if (
                            f"{device_id:s}/demods/{demod_index:d}/samplex"
                            in dataFile.keys()
                        ):
                            recordmethod = "DAQ_record"
                        elif (
                            f"000/{device_id:s}/demods/{demod_index:d}/sample/x"
                            in dataFile.keys()
                        ):
                            recordmethod = "UI"
                        elif (
                            f"000/{device_id:s}/demods/{demod_index:d}/sample.x/value"
                            in dataFile.keys()
                        ):
                            recordmethod = "DAQ_continuous"
                        elif "NMRKineticSimu/demods/0" in dataFile.keys():
                            recordmethod = "NMRKineticSimu"
                        else:
                            raise ValueError(
                                "LoadMeasInfo() cannot figure out the recording method"
                            )
                        if verbose:
                            print("recordmethod: ", recordmethod)

                        if recordmethod == "DAQ_record":
                            Stream.dmodfreq = dataFile[
                                f"{device_id:s}/demods/{demod_index:d}/dmodfreq"
                            ][0]
                            Stream.samprate = dataFile[
                                f"{device_id:s}/demods/{demod_index:d}/samprate"
                            ][0]
                            Stream.data_len += len(
                                list(
                                    dataFile[
                                        f"{device_id:s}/demods/{demod_index:d}/samplex"
                                    ]
                                )
                            )
                        elif recordmethod == "UI":
                            Stream.dmodfreq = dataFile[
                                f"000/{device_id:s}/demods/{demod_index:d}/sample/frequency"
                            ][0]
                            Stream.samprate = dataFile[
                                f"000/{device_id:s}/demods/{demod_index:d}/rate/value"
                            ][0]
                        elif recordmethod == "DAQ_continuous":
                            pass
                        elif recordmethod == "NMRKineticSimu":
                            Stream.dmodfreq = dataFile[
                                "NMRKineticSimu/demods/0/dmodfreq"
                            ][0]
                            Stream.samprate = dataFile[
                                "NMRKineticSimu/demods/0/samprate"
                            ][0]
                        else:
                            raise ValueError("cannot find recording method")
                        if verbose:
                            print("Loading Finished")
                elif singlefile[-4:] == ".csv":
                    recordmethod = "Kea"
                    Stream.exptype = "Kea Pulsed-NMR"
                    if verbose:
                        print("recordmethod: ", recordmethod)
                    dataFile = np.loadtxt(singlefile, delimiter=",")
                    Stream.timestamp = np.array(dataFile[:, 0], dtype=np.float64)

                    Stream.dataX += list(1e-6 * dataFile[:, 1])
                    Stream.dataY += list(1e-6 * dataFile[:, 2])

                    Stream.dmodfreq = 0.0
                    Stream.filterstatus = "off"
                    Stream.filter_TC = 0.0
                    Stream.filter_order = 0.0
                    Stream.attenuation = 0.0
                    Stream.acq_arr = np.array(
                        [[0, len(Stream.timestamp)], [0, len(Stream.timestamp)]],
                        dtype=np.int64,
                    )
                else:
                    raise ValueError("file type not in .h5 nor .csv")

        # LoadStreamInfo(self.nps, self.NoPulse_datafile_list)
        # LoadStreamInfo(self.pNMR_Stream, self.pNMR_dataFile_list)
        # LoadStreamInfo(self.CPMG_Stream, self.CPMG_dataFile_list)
        # self.GetMeasTimes(verbose)  # TODO uncomment it, and run it without any nps data files
        # self.NoPulseStream.freq_resol = 1.0 * self.NoPulseStream.samprate / self.NoPulseStream.data_len

    def GetMeasTimes(self, verbose: bool = False):
        """
        Get the start, end, and the duration of the measurement.
        """
        self.nps.GetStreamTimes(verbose)
        self.pNMRs.GetStreamTimes(verbose)
        self.CPMGs.GetStreamTimes(verbose)

    def GetAxionWind(
        self,
        nu_a: float = None,
        Qa=None,
        year: int = None,
        month: int = None,
        day: int = None,
        time_hms: int = None,
        verbose: bool = False,
    ):
        self.axion = AxionWind(
            name="axion",
            nu_a=nu_a or self.nps.dmodfreq,  # Compton frequency in [Hz]
            Qa=None,
            year=year,
            month=month,
            day=day,
            time_hms=time_hms,
            timeastro=self.nps.timeastro,
            station=self.station,
            verbose=False,
        )
        self.axion.lw_p = int(np.ceil(self.axion.lw_Hz / self.nps.freq_resol))
        assert self.axion.lw_p >= 1

    def InsertFakeAxion(
        self,
        nu_a: float = None,
        Qa=None,
        g_a=1.0,
        year=None,
        month=None,
        day=None,
        time_hms=None,
        rand_amp: bool = True,
        rand_phase: bool = True,
        verbose: bool = False,
    ):
        """
        Insert fake axion into the power spectrum with a give coupling
        """
        self.GetAxionWind(
            nu_a=nu_a, Qa=Qa, year=year, month=month, day=day, time_hms=time_hms
        )

        if not hasattr(self.nps, "avgFFT"):
            self.nps.GetNoPulseFFT()

        nu_a_index = self.nps.Hz2Index(self.axion.nu_a)
        f0 = nu_a_index
        f1 = min(len(self.nps.frequencies) - 1, f0 + int(self.axion.lw_p * 10.0))
        freq = self.nps.frequencies[f0:f1]
        # check(freq.shape)
        # check((self.nps.avgFFT[f0:f1]).shape)
        self.nps.FFT[f0:f1] += g_a * self.axion.GetFFTsignal(
            freq=freq, rand_amp=rand_amp, rand_phase=rand_phase, verbose=verbose
        )
        return

    def MovAvgByStep(self, step_len: int = 10, verbose: bool = False):
        """'
        regarding the standard deviation after the moving average, check Ref.[1].

        Returns
        -------
        weights

        References
        ----------
        [1] https://www.overleaf.com/read/mqtjnfqzjgkf#383444
        [2] Mathematica code for checking chi-sq distribution related.
        k=2
        data = RandomVariate[ChiSquareDistribution[k], 10000];
        data1 = RandomVariate[ChiSquareDistribution[k], 10000];
        Show[Histogram[data, 40, "ProbabilityDensity"],
        Plot[PDF[ChiSquareDistribution[2], x], {x, 0.1, 9},
        PlotStyle -> Thick]]
        Mean[data * data1] (*The value should be k^2*)
        k^2

        """
        weights = self.axion.lineshape_t(
            self.axion.nu_a
            + np.arange(0.0, 10 * self.axion.lw_Hz, step=self.nps.freq_resol)
        )
        self.nps.psdMovAvgByStep(weights=weights, step_len=step_len, verbose=False)
        self.axion.lw_p = int(np.ceil(self.axion.lw_Hz / self.nps.freq_resol))
        assert self.axion.lw_p >= 1
        return weights

    def OptimalFilter(self, step_len: int = 10, verbose: bool = False):
        """'

        regarding the standard deviation after the moving average, check Ref.[1].

        References
        ----------
        [1] https://www.overleaf.com/read/mqtjnfqzjgkf#383444
        [2] Mathematica code for checking chi-sq distribution related.
        k=2
        data = RandomVariate[ChiSquareDistribution[k], 10000];
        data1 = RandomVariate[ChiSquareDistribution[k], 10000];
        Show[Histogram[data, 40, "ProbabilityDensity"],
        Plot[PDF[ChiSquareDistribution[2], x], {x, 0.1, 9},
        PlotStyle -> Thick]]
        Mean[data * data1] (*The value should be k^2*)
        k^2



        """
        weights: np.ndarray = self.axion.lineshape_t(
            self.axion.nu_a
            + np.arange(0.0, 10 * self.axion.lw_Hz, step=self.nps.freq_resol)
        )
        self.nps.psdMovAvgByStep(weights=weights, step_len=step_len, verbose=False)
        self.nps.PSD *= np.sum(weights) / np.sum(weights**2)

        self.axion.lw_p = int(np.ceil(self.axion.lw_Hz / self.nps.freq_resol))
        assert self.axion.lw_p >= 1
        return weights

    def TestRecovery(
        self,
    ):
        """
        test the analysis with inserting axions of different Compton frequencies
        and coupling strengths.
        """
        return 0

    def GetDecaySpectrum(
        self,
    ):
        """
        Generate the decay spectrum from the pulsed-NMR data file (pNMR_datafile).
        The decay spectrum is corrected for the acquisition delay and time.
        """

        self.DS_func = None  # store the function of the decayspectrum
        # DS_func returns 0 when the input frequency is ? times NMR linewidth
        # from the Larmor frequency.
        self.DS_lw_Hz = 0
        self.DS_lw_p = 0
        return 0

    def GetT2(
        self,
    ):
        """
        Get T2 time from the CPMG data file (CPMG_datafile).

        """
        self.T2 = None
        return 0

    def Hz2Index(self, freq_Hz):
        print("Yuzhe: I need to rewrite it... -- 2024-09-23")
        lenofdata = self.nps.data_len
        center_freq = self.nps.dmodfreq
        return lenofdata // 2 + int(
            np.ceil((freq_Hz - center_freq) / self.nps.freq_resol)
        )

    def Index2Hz(self, freq_index):
        print("Yuzhe: I need to rewrite it... -- 2024-09-23")
        lenofdata = self.nps.data_len
        center_freq = self.nps.dmodfreq
        return center_freq + self.nps.freq_resol * (freq_index - lenofdata // 2)  # type: ignore

    def MakeAnalysisWindow(
        self,
        awin_width_in_DS_lw,  # analysis window width, in the unit of [axion linewidth]
    ):
        """
        Determine the analysis windows and the buffer regions
        By default, it makes only one analysis window.
        """
        print("right now it only supports one window")
        self.spec_center_Hz = self.nps.dmodfreq
        self.spec_center_index = self.nps.data_len // 2

        halfwidth_p = int(np.ceil(self.DS_lw_p * awin_width_in_DS_lw / 2.0))

        # self.resnt_region_in_Hz = [self.spec_center_index - halfwidth_in_pixel, \
        #     self.spec_center_index + halfwidth_in_pixel]
        self.resnt_region_index = [
            self.spec_center_index - halfwidth_p,
            self.spec_center_index + halfwidth_p,
        ]
        self.r0 = self.resnt_region_index[0]
        self.r1 = self.resnt_region_index[1]

        self.bkg_region = None
        self.bkg_region_index = None
        # self.b0 = self.bkg_region_index[0]
        # self.b1 = self.bkg_region_index[1]

        self.buff_region = None
        self.buff_region_index = None

        self.fullwindow = None
        self.fullwindow_index = [0, -1]

        return 0

    @record_runtime_YorN(RECORD_RUNTIME)
    def GetNoPulsePSD(self, writeInHDF5=False, verbose=False):
        """
        Get spectrum from the NoPulse data file (NoPulse_datafile).
        This function firstly generates the FFT, and then generate PSD based on that.
        """
        # Keadevice = Kea(name='blank')
        # SQDsensor = SQUID(name='blank')
        if not hasattr(self, "nps"):
            self.nps = DualChanSig(
                name="LIA data",
                device="LIA",
                device_id="dev4434",
                filelist=self.NoPulse_datafile_list,
                verbose=False,
            )
        self.nps.LoadStream(
            # Keadevice=Keadevice,
            # SQDsensor=SQDsensor,
            verbose=False
        )
        self.nps.GetNoPulseFFT(chunksize=None, getstd=False, verbose=False)

        self.freq_resol = np.abs(self.nps.frequencies[1] - self.nps.frequencies[0])
        self.nps.PSD = np.abs(self.nps.FFT) ** 2.0
        # del self.nps.frequencies

        return 0

    def RemoveSpikes(
        self,
    ):
        """
        remove the spikes in the power spectrum
        """
        return 0

    def amp_ga_1iter(
        self,
        g_a,
        ifstochastic=False,
        use_sg=False,
        # use_sg = True
        sg_axlw_frac=10,
        sg_order=2,
        dmodfreq=1e6,
        samprate=230.0,
        total_dur=10.0,
        year=2022,
        month=12,
        day=23,
        time_hms="00:00:00.0",  # Use UTC time!
        nu_a=1e6 + 1,
        Qa=None,
    ):

        step = dmScanStep(
            name="test data",
            NoPulse_dataFile_list=[],
            pNMR_dataFile_list=None,
            CPMG_dataFile_list=None,
            station=self.station,
            verbose=False,
        )
        # '20221223_154712', '20221223_175006', '20221223_194651'
        step.nps.CreateArtificialStream(
            dmodfreq=dmodfreq,
            samprate=samprate,
            total_dur=total_dur,
            year=year,
            month=month,
            day=day,
            time_hms=time_hms,  # Use UTC time!
        )
        step.nps.GetNoPulseFFT()
        #
        step.GetAxionWind(nu_a=nu_a, Qa=Qa)
        axlin = step.axion.lineshape_t(nu=step.nps.frequencies)
        axlin2sum = np.sum(axlin**2) / np.sum(axlin) ** 2
        check(np.sum(axlin**2))
        check(np.sum(axlin) ** 2)
        check(1 / np.sqrt(axlin2sum))
        std = 1.0 / samprate
        step.InsertFakeAxion(
            nu_a=nu_a,
            g_a=g_a,
            Qa=Qa,
            rand_amp=ifstochastic,
            rand_phase=ifstochastic,
            # verbose=True
        )
        step.nps.PSD = np.abs(step.nps.FFT) ** 2.0
        # step.nps.GetSpectrum(showtimedomain=False, showfreqdomain=True, spectype='PSD')
        if use_sg:
            step.nps.PSD = step.nps.sgFilterPSD(
                window_length=step.axion.lw_p // sg_axlw_frac,
                polyorder=sg_order,
                makeplot=False,
            )
        # step.nps.GetSpectrum(showtimedomain=False, showfreqdomain=True, spectype='PSD')
        step.MovAvgByStep(step_len=max(1, step.axion.lw_p // 5), verbose=False)
        # step.nps.GetSpectrum(showtimedomain=False, showfreqdomain=True, spectype='PSD')
        r1 = step.nps.Hz2Index(step.axion.nu_a - 5.0 * step.axion.lw_Hz)
        # check(r1)
        noisePSDmean = np.mean(step.nps.PSD[0:r1])
        noisePSDstd = np.std(step.nps.PSD[0:r1])
        check(noisePSDmean)
        check(noisePSDstd)
        check(std / noisePSDstd)
        step.nps.PSD -= noisePSDmean
        # step.nps.GetSpectrum(showtimedomain=False, showfreqdomain=True, spectype='PSD')

        nu_a_index = step.nps.Hz2Index(step.axion.nu_a)
        # check(step.nps.Index2Hz(nu_a_index))

        ax_amp = step.nps.PSD[nu_a_index] / std
        # check(nu_a_index)
        # check(step.nps.avgPSD[nu_a_index])
        # check(np.amax(step.nps.avgPSD))
        del step
        return ax_amp

    def amp_ga(
        self,
        # numofIter = 100
        storedata=False,
        ifstochastic=False,
        use_sg=False,
        # use_sg = True
        sg_axlw_frac=10,
        sg_order=2,
        dmodfreq=1e6,
        samprate=230.0,
        total_dur=10.0,
        year=2022,
        month=12,
        day=23,
        time_hms="00:00:00.0",  # Use UTC time!
        nu_a=1e6 + 1,
        Qa=None,
        g_a_arr=np.linspace(start=0.001, stop=0.819, num=50),
    ):
        def quadratic(x, a):
            return a * x**2

        axion_signal_amps = []
        list_of_steps = []

        tic = time.time()

        # Loop style
        # for i, g_a in enumerate(g_a_arr):
        #     axion_signal_amps_for1g_a = []
        #     for j in range(10):
        #         axion_signal_amps_for1g_a.append(amp_ga_1iter(g_a))
        #     axion_signal_amps.append(axion_signal_amps_for1g_a)

        # Multi-task style
        with Pool() as pool:
            axion_signal_amps = pool.map(amp_ga_10iter, g_a_arr)
        toc = time.time()
        print(f"time consumption: {toc-tic:.1f} [s]")
        # print_progress_bar(i, total=len(g_a_arr), prefix='Progress', suffix='Complete', length=50)
        check(axion_signal_amps)

        #
        fig = plt.figure(figsize=(5, 4), dpi=150)  #
        gs = gridspec.GridSpec(nrows=1, ncols=1)  #
        # fig = plt.figure(figsize=(10, 4), dpi=150)  #
        # gs = gridspec.GridSpec(nrows=1, ncols=2)  #
        ax00 = fig.add_subplot(gs[0, 0])
        ax00.errorbar(
            x=g_a_arr,
            y=np.mean(axion_signal_amps, axis=1),
            yerr=np.std(axion_signal_amps, axis=1),
            label="MC simulation",
            fmt="o",
            markerfacecolor="none",
            markeredgecolor="black",
            ecolor="black",
            capsize=5,
            capthick=2,
        )
        # check(np.mean(axion_signal_amps, axis=1))

        popt, pcov = curve_fit(
            quadratic, g_a_arr, np.mean(axion_signal_amps, axis=1), [155.0]
        )
        check(popt)

        ax00.plot(g_a_arr, quadratic(g_a_arr, popt[0]), label="fit curve")
        print(f"5 sigma coupling strength is {np.sqrt(5. / popt[0])}. ")

        ax00.set_xlabel("input g_a")
        ax00.set_ylabel("axion signal [$\\sigma$]")
        ax00.set_title("")

        title = f"Stochastic (spiky)? {str(ifstochastic)}\n"
        title += f"SG filter? {str(use_sg)}"
        title += (
            f". SG window length = axion linewidth // {sg_axlw_frac}" if use_sg else ""
        )

        fname = f"Stochastic (spiky) {str(ifstochastic)}_"
        fname += f"SG filter {str(use_sg)}"
        fname += (
            f"_SG window length = axion linewidth by {sg_axlw_frac}" if use_sg else ""
        )
        print(fname)

        fig.suptitle(title)
        fig.tight_layout()

        plt.show()
        return

    def FindSharpSpike(
        self,
    ):
        """
        Find the sharp spikes in the power spectrum
        and (optional) record them in the NoPulse HDF5 data file
        """
        return 0

    def AxionPSD_g1(
        self,
    ):
        """
        return a function that returns axion signal with g_aNN = 1 [eV^-1]
        """
        # frequencies = np.arange(start=self.fullwindow[0], \
        #     stop=self.fullwindow[1], step=self.NMR_lw / 10.)
        # AxionPSD_g_1_map = np.ones(len(frequencies))
        # def func(nu_a):
        #     return np.vdot(Lorentzian(frequencies, c=nu_a, FWHM=self.axlw_Hz,area=1, offset=0), \
        #                    self.DS_func(frequencies))
        # AxionPSD_g_1_map *= func(frequencies)
        # AxionPSD_g_1_map *= self.axion.lineshape_t()
        # AxionPSD_g_1_map *= 1#xi_rms

        # c = 2.99792458e8 # [m/s]
        # v_0 = 220e3 #[m/s]
        # hbar_Joule = 1.05457182e-34  # [J/Hz]
        # hbar_eV = 6.582119569e-16  # [eV/Hz]
        # h_eV = 4.135667696e-15  # [eV/Hz]

        # # dark matter axion density
        # rho_DM = (0.35*1e9)*(1e6)  # eV/m^3
        # # SQUID_Mf = 22665.457842248416
        # # SQUID_Rf = 3000.0
        # # phi90 *= SQUID_Mf / SQUID_Rf # decay signal correction
        # rabi_g1 = 1./2. * np.sqrt(2. * hbar_eV * c * rho_DM)
        # AxionPSD_g_1_map *= self.axion.lineshape_t()
        # self.AxionPSD_g_1 = 1
        return 0

    def FindCandidate(
        self,
        # showchanXhist=True,
        # showchanYhist=True,
        # showFFTrealhist=True,
        # showFFTimghist=True,
        # showPSDhist=True,
        # freqRangeforHist=None,
        scale="linear",  # or 'log'
        normalizebysigma=False,
        makeplot=False,
        verbose=False,
    ):
        """
        Find the axion candidates in the moving-averaged spectrum.
        List their information, including axion Compton frequency,
        coupling strength g_aNN, linewidth.
        """

        std = np.std(self.nps.PSD[self.r0 : self.r1])
        threshold_std = 3.3
        threshold = threshold_std * std
        # Convert to non-negative values
        NonNegative_arr = np.where(
            self.nps.PSD[self.r0 : self.r1] - threshold < 0,
            0,
            self.nps.PSD[self.r0 : self.r1] - threshold,
        )

        peaks, properties = find_peaks(
            self.nps.PSD[self.r0 : self.r1],
            height=threshold,
            distance=1,
            prominence=threshold,
            width=1,
        )

        if makeplot:
            fontsize = 8
            plt.rc("font", size=fontsize)  # font size for all figures
            plt.style.use("seaborn-v0_8-deep")  # to specify different styles
            fig = plt.figure(figsize=(11, 7), dpi=150)
            width_ratios = [1, 1]
            height_ratios = [1, 1, 1]
            gs = gridspec.GridSpec(
                nrows=3, ncols=2, width_ratios=width_ratios, height_ratios=height_ratios
            )
            NoPulsePSD_ax = fig.add_subplot(gs[0, 0])  #
            NonNegative_ax = fig.add_subplot(gs[1, 0])  #
            CND_ax = fig.add_subplot(gs[2, 0])  #
            hist_ax = fig.add_subplot(gs[0:2, 1])

            NoPulsePSD_ax.plot(
                self.Index2Hz(np.arange(self.r0, self.r1)),
                self.nps.PSD[self.r0 : self.r1],
                color="tab:blue",
                label="NoPulse PSD after MovAvg",
            )
            NoPulsePSD_ax.set_ylabel("PSD")
            NonNegative_ax.plot(
                NonNegative_arr, color="tab:green", label="PSD > threshold * std"
            )
            NoPulsePSD_ax.set_ylabel("PSD")

            def index2hz_r0offset(arg):
                return self.Index2Hz(self.r0 + arg)

            CND_ax.plot(
                self.Index2Hz(np.arange(self.r0, self.r1)),
                self.nps.PSD[self.r0 : self.r1],
                color="tab:blue",
                label="NoPulse PSD after MovAvg",
            )
            CND_ax.plot(
                index2hz_r0offset(peaks),
                self.nps.PSD[self.r0 + peaks],
                "x",
                color="tab:orange",
                label="Candidate(s)",
            )

            # draw candidate(s)' height and width
            CND_ax.vlines(
                x=index2hz_r0offset(peaks),
                ymin=self.nps.PSD[self.r0 + peaks] - properties["prominences"],
                ymax=self.nps.PSD[self.r0 + peaks],
                color="tab:orange",
            )
            CND_ax.hlines(
                y=properties["width_heights"],
                xmin=index2hz_r0offset(properties["left_ips"]),
                xmax=index2hz_r0offset(properties["right_ips"]),
                color="tab:orange",
            )

            # draw the threshold
            CND_ax.hlines(
                y=threshold,
                xmin=NoPulsePSD_ax.get_xlim()[0],
                xmax=NoPulsePSD_ax.get_xlim()[1],
                color="tab:grey",
                linestyles="--",
                label="threshold",
            )

            numofbin_input = int(
                3 * np.ceil(np.ptp(self.nps.PSD[self.r0 : self.r1]) / std)
            )
            hist, bin_edges = np.histogram(
                self.nps.PSD[self.r0 : self.r1],
                bins=max(numofbin_input, 6),
                density=False,
            )
            binwidth = abs(bin_edges[1] - bin_edges[0])
            sumofcounts = abs(self.r1 - self.r0)
            assert type(sumofcounts) is int

            hist_x = []
            hist_y = []
            for i, count in enumerate(hist):
                if count > 0:
                    hist_x.append((bin_edges[i] + bin_edges[i + 1]) / 2.0)
                    hist_y.append(count)
            hist_ax.scatter(
                hist_x,
                hist_y,
                color="green",
                edgecolors="k",
                linewidths=1,
                marker="o",
                s=6,
                zorder=6,
                label=f"histogram",
            )
            hist_ax.vlines(
                x=threshold,
                ymin=0.1,
                ymax=np.amax(hist_y),
                color="tab:grey",
                label="threshold",
                linestyles="--",
            )
            PlotGaussianLine(ax=hist_ax, x=bin_edges, std=std, sumofcounts=sumofcounts)
            hist_ax.set_yscale("log")

            NoPulsePSD_ax.legend(loc="upper right")  #
            NonNegative_ax.legend(loc="upper right")  #
            CND_ax.legend(loc="upper right")  #
            hist_ax.legend(loc="upper right")
            fig.tight_layout()
            plt.show()

        return 0

    def GetExclusion(
        self,
    ):
        """
        make the exclusion plot determined by the measurement
        """
        return 0


class dmScan:
    def __init__(
        self,
        name,
        basepath=rf"C:\Users\Admin\Desktop\CASPEr-thermal_DMdata22",
        analysisname="",
        step_list=[],
        verbose=False,
    ):
        """
        Initialize DMscan class for a dark-matter measurement scan by CASPEr-gradient-lowfield
        This design of this object should and is capable of dealing with large data file,
        which is important to the earth- / solar- dark-matter halo data analysis.

        """

        if isinstance(name, str):
            # print('single scan input')
            self.name = [name]
        elif isinstance(name, list) and all(isinstance(i, str) for i in name):
            # print('list of scans input')
            self.name = name
        else:
            raise OSError("no valid scan inputted. must be string or list of strings.")

        if step_list is None:
            self.step_list = []
        else:
            self.step_list = step_list
        self.basepath = basepath
        self.analysisname = (
            analysisname  # name of the folder where the axionfinder results are saved
        )
        self.scanInfo = {}
        self.allDMfiles = []
        self.allNMRfiles = []
        self.allCPMGfiles = []
        self.allMATfiles = []
        self.singlerunfile = []
        self.singlerunindex = []

        self.iCND = []  # indices of candidate frequencies in convolved PSD
        self.fCND = []  # candidate frequencies
        self.pcnvCND = (
            []
        )  # convolved PSD values at the candidate frequencies # pcnvCND=(10^powerOfTenScaling)*convolvedPSD(candind)
        self.prawCND = (
            []
        )  # raw PSD values at the candidate frequencies # prawCND=PSDwindowNoBuffer(psdcandind)
        self.gCND = []  # coupling strength of the candidate axions
        self.threshold_rescan = (
            []
        )  # the PSD value at the rescan threshold # pLIM=(10^powerOfTenScaling)*candThreshVal
        self.threshold_exclude = (
            []
        )  # the convolved PSD value at the exclusion threshold # SNRtarget=(10^powerOfTenScaling)*excThreshVal
        self.powerOfTenScaling = 0

        self.cands_res = []  # candidates close to the larmor freq of their steps
        self.cands_bkg = []  # cands far from the larmor freq of their steps
        self.cands_filtered = (
            []
        )  # ntuples of cands after various steps of filtering done in FilterCandidates()

        self.fLIM = []  # step frequencies (for limit plot)
        self.gLIM = []  # exclusion limits of steps

        self.fakeAx_fIN = []
        self.fakeAx_fOUT = []
        self.fakeAx_gIN = []
        self.fakeAx_gOUT = []
        self.fakeAx_frecovery = []
        self.fakeAx_grecovery = []

    def LoadScanInfo(
        self,
        singlerunfile=None,  # rf'./Supplementary/20240627 Thermal Paper/TESTrun_vars.mat'
        singlerunindex=0,
        verbose: bool = True,
    ):
        """
        structure of the scan information dictionary
        level-1
        ['dmScanStep']: list
        ['MeasTime']: list, optional
        ['FreqRange']: list, optional

        level-2
        ['dmScanStep'][]: dict
            dictionary for dmScanStep information
        ['MeasTime'][]: dict
            ['period']: list, optional
                start and end of the scannings in str
            ['duration']: float, optional
                duration of the scannings in [s]
        ['FreqRange'][]: list, optional
            frequency range of the scannings in [Hz]

        """

        Scan_dict = None
        # Your dictionary
        scanInfor = {"key1": "value1", "key2": "value2"}

        # Loading a dictionary:
        # Load from a file
        # with open('my_dict.pkl', 'rb') as f:
        #    loaded_dict = pickle.load(f)
        # print(loaded_dict)

        for subscan in self.name:
            if "20221223" in subscan:
                # print("a scan from day 2")
                DMfiles = glob.glob(rf"{self.basepath}/{subscan}/AxionScan_*.h5")
                NMRfiles = glob.glob(rf"{self.basepath}/{subscan}/OnePulse1_*.h5")
                CPMGfiles = glob.glob(rf"{self.basepath}/{subscan}/CPMG_*.h5")
                MATfiles = glob.glob(
                    rf"{self.basepath}/{subscan}/AxionFinder results/{self.analysisname}/*/*vars.mat"
                )
            elif "20221214" in subscan:
                # print("a scan from day 1")
                DMfiles = glob.glob(
                    rf"{self.basepath}/{subscan}/0Pulse data/0Pulse_*/stream_*.h5"
                )
                NMRfiles = glob.glob(
                    rf"{self.basepath}/{subscan}/1Pulse data/1Pulse_*/stream_*.h5"
                )
                CPMGfiles = glob.glob(
                    rf"{self.basepath}/{subscan}/CPMG data/*/CPMG_data.h5"
                )
                MATfiles = []  # not analyzed yet

            self.allNMRfiles.append(sorted(NMRfiles, key=extract_number))
            self.allCPMGfiles.append(sorted(CPMGfiles, key=extract_number))
            self.singlerunfile = singlerunfile
            if singlerunfile is not None:
                self.singlerunindex = singlerunindex
                self.allDMfiles.append([singlerunfile])
                self.allMATfiles.append([singlerunfile])
            else:
                self.allDMfiles.append(sorted(DMfiles, key=extract_number))
                self.allMATfiles.append(sorted(MATfiles, key=extract_number))

    def LoadScanFromDict(self, infoDict):
        """
        example 

        # 
        scanstep7_dict = {
            'NoPulseDataPath_list': [],
            'OnePulseDataPath_list': [],
            'CPMGDataPath_list':[]
        }
        scanstep7_dict['CPMGDataPath_list'] = \
            [maindatafolder + rf'\CPMG data\40\CPMG_data.h5',
            maindatafolder + rf'\CPMG data\41\CPMG_data.h5']
        scanstep7_dict['OnePulseDataPath_list'] = \
            [maindatafolder + rf'\stream_009' + f'stream_00009.h5',
            maindatafolder + rf'\stream_011' + f'stream_000013.h5']
        for i in range(9, 13 + 1):
            scanstep7_dict['NoPulseDataPath_list'].append(\
                maindatafolder + rf'\stream_010' + f'/stream_000{i:02}.h5')
        
        info_dict['stepinfo'] = [scanstep0_dict, scanstep1_dict, scanstep2_dict, scanstep3_dict,\
                                scanstep4_dict, scanstep5_dict, scanstep6_dict, scanstep7_dict]
        """
        # for step_dict in scanDict:
        total = len(infoDict["stepinfo"])
        for i, stepInfo in enumerate(infoDict["stepinfo"]):
            print_progress_bar(i, total=total, prefix="Loading scan steps")
            step = dmScanStep(
                name="test data",
                NoPulse_dataFile_list=stepInfo["NoPulseDataPath_list"],
                pNMR_dataFile_list=None,
                CPMG_dataFile_list=None,
                station=Mainz,
                verbose=False,
            )
            # step.LoadMeasInfo()
            step.LoadMeasInfo()
            # step.NoPulseStream.LoadStream()
            # step.NoPulseStream.dataX += np.linspace(0, 5*np.std(step.NoPulseStream.dataX), num=len(step.NoPulseStream.dataX))
            # step.NoPulseStream.GetNoPulsePSD(chunksize=0.2)
            # step.NoPulseStream.GetSpectrum(showtimedomain=False, showfreqdomain=True)
            self.step_list.append(step)
        return

    def UpdateScanInfor(
        self,
    ):
        return

    def SaveScanInfor(
        self,
    ):
        # Your dictionary
        my_dict = {"key1": "value1", "key2": "value2"}

        # Save to a file
        with open("my_dict.pkl", "wb") as f:
            pickle.dump(my_dict, f)

    def MakeDataTextfiles(
        self,
        basepath,
        scannames,
        analysisname,
    ):
        print("MakeDataTextfiles() start")
        if scannames == None:
            scannames = ["20221223_154712", "20221223_175006", "20221223_194651"]
        scans = dmScan(basepath=basepath, name=scannames, analysisname=analysisname)
        scans.LoadScanInfo()
        # NMRfluxes=[]
        for scanindex, scanfiles in enumerate(scans.allDMfiles):
            allALPparams = []
            taus = []
            for runindex, runfile in enumerate(scans.allDMfiles[scanindex]):
                nu_n, FWHM_n, amp_n, nu_n_err, FWHM_n_err, amp_n_err = ReturnNMRdata(
                    basepath, scannames[scanindex], runindex
                )
                year, month, day, time_str, datetime_int = ReturnMeasureTimFromFile(
                    runfile
                )  # 2022, 12, 23, '15:47:18'
                ALPparams = AxionWind().GetALP_Data(
                    year=year,
                    month=month,
                    day=day,
                    time_str=time_str,
                    datetime_int=datetime_int,
                    T_acq=0.5,
                    T2time=ReturnT2time(basepath, scannames[scanindex]),
                    NMRfreq=[nu_n, nu_n_err],
                    NMRwidth=[FWHM_n, FWHM_n_err],
                    NMRamp=[amp_n, amp_n_err],
                    nu_ALP=0,
                    g_aNN=1,
                    usedailymod=True,
                )[
                    :-1
                ]  # don't load uncertainties
                allALPparams.append(ALPparams)
                print_progress_bar(
                    runindex,
                    total=len(scans.allDMfiles[scanindex]),
                    prefix="Progress",
                    suffix="Complete",
                    length=50,
                )
                # NMRfluxes.append(ALPparams[8])
                taus.append(ALPparams[3])
            print(f"average tau: {np.mean(taus)}")
            np.savetxt(
                f"{basepath}/{scannames[scanindex]}/_DM_measurements_data.txt",
                allALPparams,
            )
        # print(f"average mag. flux during pi/2 pulse: {np.mean(NMRfluxes)}")

    def tsCheckJump(
        self,
    ):
        return

    def tsCheckDrift(
        self,
    ):
        return

    def tsCheckSanity(self, plotIfInsane: bool = False, verbose=False):
        total = len(self.step_list)
        for i, step in enumerate(self.step_list):
            print_progress_bar(
                i, total, length=50, prefix="Checking sanity of scan steps"
            )
            step.nps.LoadStream(verbose=verbose)
            report = step.nps.tsCheckSanity(plotIfInsane=plotIfInsane, verbose=verbose)
            del step.nps.dataX, step.nps.dataY
            # if i >= 0:
            #     clear_lines()
            #     print(report)

        print_progress_bar(
            total, total, length=50, prefix="Checking sanity of scan steps"
        )
        sys.stdout.write(
            "\n"
        )  # Move to the next line after the progress bar is complete
        return

    def SumScanInfor(
        self,
    ):
        """
        Summarize the information of the scan.
        """
        return 0

    def SumAnalysisWindows(
        self,
    ):
        """
        Summarize the analysis windows of all scan steps.
        """
        return 0

    def VerticalComb(
        self,
    ):
        """
        Vertical combination of the scan.
        """
        return 0

    def HorizontalComb(
        self,
    ):
        """
        Horizontal combination of the whole spectrum.
        """
        return 0

    def FindCandidate(
        self,
        howtothresh=6,
        resavemat_opt=False,
        normalizebysigma=False,
        plot_opt=False,
        save_opt=False,
        savepath="",
        verbose=False,
    ):
        """
        Find the axion candidates in the resonant window.
        List their information, including axion Compton frequency,
        coupling strength g_aNN, linewidth and etc.(?)

        howtothresh: in case you want to redo the outlier identification from matlab. how to calculate the detection threshold?
            * 0: take fcands AND gcands found by matlab
            * 1: take ONLY fcands found by matlab, calculate coupling manually (if it went wrong in matlab)
            * 2: calculate threshold & candidates again from Gaussian fit info done in matlab
            * 3: calculate threshold & candidates again from Chi^2 fit info done in matlab
            * 4: do a chi^2 fit now to the convolved PSD
            * 5: do a gamma distribution fit now to the convolved PSD
            * 6: find the n-sigma threshold numerically (the correct p-value has been found beforehand using, for example, MC simulation)
            * 7: calculate the CDF to find threshold p-values

        resavemat_opt: write candidate data found here into the matlab log files, for later use

        normalizebysigma: whether to do normalization of optimally filtered PSD by its standard deviation. should have already been done in matlab

        """
        print("FindCandidate() start")

        def chi_squared_pdf(bincnt, v):
            return v[0] * np.max(counts) * chi2.pdf(bincnt / v[1] + v[2], v[3])

        def RNCF_chi2(v):
            return np.linalg.norm(counts - chi_squared_pdf(bincenters, v))

        def gamma_pdf(bincnt, v):
            return gamma.pdf(bincnt, v[0], v[1], v[2])  # shape, loc, scale

        def RNCF_gamma(v):
            return np.linalg.norm(counts - gamma_pdf(bincenters, v))

        def gamma_pdf2(bincnt, v):
            return (
                v[0]
                * np.max(counts)
                * gamma.pdf(bincnt / v[1] + v[2], v[3], v[4], v[5])
            )

        def RNCF_gamma2(v):
            return np.linalg.norm(counts - gamma_pdf2(bincenters, v))

        scans = self.name
        all_candidates = []
        for scanindex, scan in enumerate(scans):
            if verbose:
                check(self.allDMfiles[scanindex])
            check(scan)

            measdatafile = f"{self.basepath}/{scan}/_DM_measurements_data.txt"
            larmors = np.genfromtxt(
                measdatafile,
                unpack=True,
                delimiter=" ",
                skip_header=0,
                filling_values=0,
                invalid_raise=False,
            )[5]
            all_iCND = []
            all_fCND = []
            all_gCND = []
            all_pcnvCND = []
            # all_prawCND=[]
            all_threshold_rescan = []
            all_threshold_exclude = []
            all_cands_exp = []
            for runind, file in enumerate(self.allDMfiles[scanindex]):

                if self.singlerunfile is not None:
                    matlogfile = self.singlerunfile
                    if verbose:
                        print("looking for singlerunfile")
                else:
                    matlogfile = rf"{self.basepath}/{scan}/AxionFinder results/{self.analysisname}/{runind+1}/{self.analysisname}_vars.mat"
                    if verbose:
                        print("looking for this run's matfile")
                mat_data = loadmat(matlogfile)
                # mat_dict = convert_matlab_cells(mat_data)

                self.powerOfTenScaling = (
                    mat_data["powerOfTenScaling"].flatten().tolist()[0]
                )  # scaling factor

                if howtothresh == 0 or howtothresh == 1:
                    fCND = mat_data["fCND"].flatten().tolist()  # candidate frequencies
                    prawCND = (
                        mat_data["prawCND"].flatten().tolist()
                    )  # raw PSD values at the candidate frequencies # prawCND=PSDwindowNoBuffer(psdcandind)
                    pcnvCND = (
                        mat_data["pcCND"].flatten().tolist()
                    )  # convolved PSD values at the candidate frequencies # pcnvCND=(10^powerOfTenScaling)*convolvedPSD(candind)
                    pLIM = (
                        mat_data["pLIM"].flatten().tolist()[0]
                    )  # the PSD value at the rescan threshold # pLIM=(10^powerOfTenScaling)*candThreshVal
                    threshold_rescan = pLIM / (
                        10**self.powerOfTenScaling
                    )  # the NPE plot value at the rescan threshold
                    # the convolved PSD value at the exclusion threshold # pEXC=(10^powerOfTenScaling)*excThreshVal
                    try:
                        SNRtarget = mat_data["SNRtarget"].flatten().tolist()[0]
                    except KeyError:
                        SNRtarget = mat_data["pEXC"].flatten().tolist()[0]
                    threshold_exclude = SNRtarget / (
                        10**self.powerOfTenScaling
                    )  # the NPE plot value at the exclusion threshold
                    try:
                        iCND = mat_data["candind"].flatten().tolist()
                    except KeyError:
                        # convolvedFreq = mat_data['convolvedFreq'].flatten().tolist() # frequencies array of convolved PSD
                        convolvedPeak = (
                            mat_data["convolvedPeak"].flatten().tolist()
                        )  # convolved PSD values array
                        iCND = [
                            index
                            for index, x in enumerate(convolvedPeak)
                            if convolvedPeak[index] > threshold_rescan
                        ]

                elif howtothresh == 2:
                    # normalFitInfo contains:
                    # for Gaussian fit
                    # 0) window center freq
                    # 1) fitcurve amplitude
                    # 2) fitcurve mean
                    # 3) fitcurve std
                    # 4) 0
                    # 5) 0
                    # 6) 0
                    # 7) residual
                    # 8) fl
                    # 9) logres
                    # 10) std of PSD
                    # 11) GSC
                    # 12) length of PSD
                    # 13) normtest result
                    # 14) normtest p-value
                    # 15) candThreshVal
                    # 16) 0
                    # 17) binwidth
                    # 18) num of cut freqs in SG filter
                    # for Chi^2 fit
                    # 0) window center freq
                    # 1) chi^2 v1
                    # 2) chi^2 v2
                    # 3) chi^2 v3
                    # 4) chi^2 v4 (degrees of freedom)
                    # ... rest is the same

                    normalFitInfo = mat_data["normalFitInfo"].flatten().tolist()
                    fitlabel = "Gaussian fit"
                    # GSC = 2*np.max(counts)*convolvedPSDstd
                    fitparams = [
                        normalFitInfo[11],
                        normalFitInfo[1],
                        normalFitInfo[2],
                        normalFitInfo[3],
                    ]
                    fitcurve = (
                        fitparams[0]
                        * fitparams[1]
                        * norm.pdf(bincenters, fitparams[2], fitparams[3])
                    )
                    threshold_rescan = fitparams[2] + 3.355 * fitparams[3]
                    threshold_exclude = fitparams[2] + 5.0 * fitparams[3]

                elif howtothresh == 3:

                    chi2FitInfo = mat_data["normalFitInfo"].flatten()
                    fitlabel = "Chi-Squared fit"
                    fitparams = [
                        chi2FitInfo[1],
                        chi2FitInfo[2],
                        chi2FitInfo[3],
                        chi2FitInfo[4],
                    ]
                    fitcurve = chi_squared_pdf(bincenters, fitparams)
                    threshold_rescan = chi2FitInfo[
                        15
                    ]  # this threshold is probably incorrect. it was calculated still using the gaussian params
                    threshold_exclude = 5.0 * chi2FitInfo[4]  # 5 sigma

                elif howtothresh == 4:
                    v0 = [
                        15,
                        0.01,
                        -50,
                        20,
                    ]  # this worked OK as a guess for fitting in matlab
                    v0 = [
                        405.16583457911196,
                        -0.0009256657609884802,
                        15363.066801881694,
                        15376.542179937464,
                    ]  # result of fit in matlab
                    bounds = [
                        (1e2, 1e8),  # prefactor
                        (-1e5, 1e5),  # scale
                        (-1e5, 1e5),  # shift
                        (2, 1e2),
                    ]  # dof # should not be much more than 10 I guess
                    result = minimize(
                        RNCF_chi2,
                        x0=v0,
                        # bounds=bounds,
                        method="Nelder-Mead",
                    )
                    fitparams = result.x
                    fitcurve = chi_squared_pdf(bincenters, fitparams)
                    fitlabel = "Chi-Squared fit"

                    p95 = chi2.ppf(
                        1 - 0.95, fitparams[3]
                    )  # the 95 percentile for a chi-square curve with dof as found by our fit
                    # we are trying to find the position where bincenters / v[1] + v[2] = p95
                    threshold_rescan = fitparams[1] * (p95 - fitparams[2])
                    threshold_exclude = 5.0 * fitparams[3]  # 5 sigma

                elif howtothresh == 5:
                    # fitparams = gamma.fit(bincenters) # shape, loc, scale
                    # fitcurve = gamma.pdf(bincenters, fitparams[0], fitparams[1], fitparams[2])
                    # threshold = fitparams[0]*fitparams[2] + 3.62 * np.sqrt(fitparams[0])*fitparams[2]

                    v0 = [1, 1, 0, 9, 0, 10]
                    bounds = [
                        (1e-3, 1e3),
                        (1e-5, 1e5),
                        (-1e2, 1e2),
                        (9, 20),  # shape
                        (-1, 1),  # location
                        (-10, 10),  # scale
                    ]
                    result = minimize(
                        RNCF_gamma2,
                        x0=v0,
                        bounds=bounds,
                        method="Nelder-Mead",
                    )
                    fitparams = result.x  # amp, prefactor, offset, shape, loc, scale
                    fitcurve = gamma_pdf2(bincenters, fitparams)
                    fitlabel = "Gamma fit"
                    # threshold = fitparams[3]*fitparams[5] + 3.62 * np.sqrt(fitparams[3])*fitparams[5]

                    p95 = gamma.ppf(
                        1 - 0.95,
                        fitparams[3],
                        loc=fitparams[4] + 6.0 * np.sqrt(fitparams[3]) * fitparams[5],
                        scale=fitparams[5],
                    )
                    threshold_rescan = fitparams[1] * (p95 - fitparams[2])
                    threshold_exclude = 5.0 * fitparams[3]

                elif howtothresh == 6:
                    convolvedFreq = (
                        mat_data["convolvedFreq"].flatten().tolist()
                    )  # frequencies array of convolved PSD
                    convolvedPeak = (
                        mat_data["convolvedPeak"].flatten().tolist()
                    )  # convolved PSD values array
                    # convolvedPSDmean = mat_data['convolvedPSDmean'].flatten().tolist()[0]
                    # convolvedPSDstd = mat_data['convolvedPSDstd'].flatten().tolist()[0] # <- this is not reliable since it will be nonsense if the fit in matlab is bad

                    PSD_sigma = np.std(convolvedPeak)
                    PSD_mean = np.mean(convolvedPeak)
                    # check(PSD_mean)
                    # check(PSD_sigma)
                    threshold_rescan = PSD_mean + 3.43 * PSD_sigma
                    threshold_exclude = PSD_mean + 5.0 * PSD_sigma

                if (
                    howtothresh > 0
                ):  # coupling values have to be recalculated for the candidates found

                    if howtothresh > 1:
                        convolvedFreq = (
                            mat_data["convolvedFreq"].flatten().tolist()
                        )  # frequencies array of convolved PSD
                        convolvedPeak = (
                            mat_data["convolvedPeak"].flatten().tolist()
                        )  # convolved PSD values array
                        if verbose:
                            check(convolvedFreq)
                        if verbose:
                            check(convolvedPeak)

                        iCND = [
                            index
                            for index, p in enumerate(convolvedPeak)
                            if convolvedPeak[index] > threshold_rescan
                        ]
                        fCND = [
                            f
                            for index, f in enumerate(convolvedFreq)
                            if convolvedPeak[index] > threshold_rescan
                        ]
                        pcnvCND = [
                            p * (10**self.powerOfTenScaling)
                            for index, p in enumerate(convolvedPeak)
                            if convolvedPeak[index] > threshold_rescan
                        ]
                        # A = convolvedFreq[iCND] - FreqWindow[0]
                        # B = FreqWindow - FreqWindow[0]
                        # common_elements, psdcandind = np.intersect1d(A, B, assume_unique=False, return_indices=True)
                        # prawCND = [ PSDWindow[index] for index in psdcandind[1] ]

                    gCND = []
                    nu_n, FWHM_n, amp_n, nu_n_err, FWHM_n_err, amp_n_err = (
                        ReturnNMRdata(self.basepath, scan, runind)
                    )
                    year, month, day, time_str, datetime_int = ReturnMeasureTimFromFile(
                        file
                    )  # 2022, 12, 23, '15:47:18'
                    for index, i in enumerate(
                        fCND
                    ):  # calculate ALP signal for each candidate frequency
                        fakeAxion_Amp = AxionWind().GetALP_Data(
                            year=year,
                            month=month,
                            day=day,
                            time_str=time_str,
                            datetime_int=datetime_int,
                            T_acq=0.5,
                            T2time=ReturnT2time(self.basepath, scan),
                            NMRfreq=[nu_n, nu_n_err],
                            NMRwidth=[FWHM_n, FWHM_n_err],
                            NMRamp=[amp_n, amp_n_err],
                            nu_ALP=i,
                            g_aNN=1,
                        )[9]
                        coupling_cand = np.sqrt(pcnvCND[index] / fakeAxion_Amp)
                        # check(coupling_cand)
                        gCND.append(coupling_cand)

                else:
                    gCND = (
                        mat_data["gCND"].flatten().tolist()
                    )  # coupling strength of the candidate axions

                cands_exp = mat_data["expected"].flatten().tolist()[0]
                # check(cands_exp)
                # cands_exp = (larmor * b2) / df * (empiricalCDF(candThreshIndex))
                # b2 = mat_data['b2'].flatten().tolist()[0]
                # df = mat_data['df'].flatten().tolist()[0]
                # Npts = mat_data['Npts'].flatten().tolist()[0]
                # candCDF = cands_exp * df / (larmors[runind] * b2)
                # cands_exp = Npts * (candCDF)
                # check(cands_exp)
                # all_cands_exp += cands_exp
                all_cands_exp.append(cands_exp)

                if verbose:
                    print(f"{len(fCND)} outliers found in file: {matlogfile}")

                all_iCND.append(iCND)
                all_fCND.append(fCND)
                all_gCND.append(gCND)
                all_pcnvCND.append(pcnvCND)
                # all_prawCND.append(prawCND)
                all_threshold_rescan.append(threshold_rescan)
                all_threshold_exclude.append(threshold_exclude)

                for candindex, cand in enumerate(fCND):
                    all_candidates.append(
                        [scan, runind, fCND[candindex], gCND[candindex]]
                    )

                if resavemat_opt:
                    mat_data["iCND"] = iCND
                    mat_data["fCND"] = fCND
                    mat_data["gCND"] = gCND
                    mat_data["pcnvCND"] = pcnvCND
                    # mat_data['prawCND'] = prawCND
                    mat_data["threshold_rescan"] = threshold_rescan
                    mat_data["threshold_exclude"] = threshold_exclude
                    savemat(matlogfile, mat_data)

                if plot_opt:
                    convhist = mat_data["convhist"].flatten().tolist()
                    # convhist contains a bunch of zeroes after some matlab changes made after 9.9.24. I am trying to get rid of them here
                    # convhistcut = [value for value in convhist if value != 0]
                    firstnonzero_ind = next(
                        (i for i, x in enumerate(convhist) if x != 0), len(convhist)
                    )
                    convhist = convhist[firstnonzero_ind:]
                    centerfreq = convhist[0]
                    bincenters = convhist[1 : int((len(convhist) - 1) / 2 + 1)]
                    counts = convhist[int((len(convhist) - 1) / 2 + 1) :]

                    if normalizebysigma:
                        bincenters = bincenters / np.std(convolvedPeak)

                    plt.figure(figsize=(16 * 0.9, 9 * 0.9), dpi=100)
                    plt.scatter(
                        bincenters,
                        counts,
                        s=100,
                        marker="o",
                        label="Filtered PSD",
                        alpha=0.9,
                    )  # color='blue'

                    try:
                        check(fitparams)
                        plt.plot(
                            bincenters,
                            fitcurve,
                            color="green",
                            linewidth=4,
                            label=fitlabel,
                            alpha=0.9,
                        )
                    except NameError:
                        pass

                    plt.axvline(
                        x=threshold_rescan,
                        color="green",
                        linestyle="--",
                        linewidth=4,
                        alpha=0.7,
                        label=rf"Detection threshold ($95~\%$ confidence)",
                    )
                    plt.axvline(
                        x=threshold_exclude,
                        color="red",
                        linestyle="--",
                        linewidth=4,
                        alpha=0.7,
                        label=rf"Sensitivity threshold ($5\sigma$)",
                    )

                    outlier_inds = [
                        index
                        for index, x in enumerate(bincenters)
                        if x > threshold_rescan
                    ]
                    plt.scatter(
                        [bincenters[i] for i in outlier_inds],
                        [counts[i] for i in outlier_inds],
                        s=100,
                        color="green",
                        marker="o",
                        label="Signal candidates",
                        alpha=0.9,
                    )

                    # plt.yscale('log')
                    plt.ylim(0.5, 1.5 * max(counts))
                    plt.xlabel(rf"Normalized power excess")  # [$\Phi_0^2$]
                    plt.ylabel("Counts")
                    # plt.xticks(np.arange(-0.6, 2.0, 0.5))
                    # plt.grid(True, linestyle='--', alpha=0.7)
                    plt.legend()  # loc='upper right'
                    # plt.autoscale()
                    plt.tight_layout()
                    plt.savefig(
                        rf"{savepath}/convhistplot_{scan}_{runind}.png",
                        bbox_inches="tight",
                    )
                    # plt.show()
                    plt.clf()
                    plt.close()

                print_progress_bar(
                    runind,
                    total=len(self.allDMfiles[scanindex]),
                    prefix="Progress",
                    suffix="Complete ",
                    length=50,
                )

            num_outliers = sum(len(cands) for cands in all_fCND)
            print(f"total # outliers in all runs: {num_outliers}")
            print(
                f"avg # outliers per run: {num_outliers / len(self.allDMfiles[scanindex])}"
            )
            print(f"avg # expected outliers per run: {np.mean(all_cands_exp)}")

            self.iCND.append(all_iCND)
            self.fCND.append(all_fCND)
            self.gCND.append(all_gCND)
            self.pcnvCND.append(all_pcnvCND)
            # self.prawCND.append(all_prawCND)
            self.threshold_rescan.append(all_threshold_rescan)
            self.threshold_exclude.append(all_threshold_exclude)

        if save_opt:
            cand_data = {
                "scan": np.array([cand[0] for cand in all_candidates]),
                "run": np.array([cand[1] for cand in all_candidates]),
                "frequency": np.array([cand[2] for cand in all_candidates]),
                "coupling": np.array([cand[3] for cand in all_candidates]),
            }
            df = pd.DataFrame(cand_data)
            df.to_excel(rf"{savepath}/candidates.xlsx")
            # df.to_excel(rf'{self.basepath}/{scan}/AxionFinder results/{self.analysisname}/{runind+1}/candidates.xlsx')

    def FilterCandidates(
        self,
        mainscan_name="",
        candidates_log="",
        cut0=False,
        cut1=False,
        cut2=False,
        cut3=False,
        compare_candidates=False,
        candidate_window=100,
        samepeak_cut=1,
        background_cut=1,
        rescan_cut=10,
        centerfreq=np.array(1.3485e6),
        loadfrommat=False,
        savepath="",
        save_to_excel=False,
        saveplot_opt=False,
        showplot_opt=False,
    ):
        """
        looks at all data points above the detection threshold found in the input scans, performs some checks to test if they can be axion signal candidates


        mainscan_name: if one of the inputted scans should be considered main scan, enter it here

        cut0: do same-signal merging. see samepeak_cut

        cut1: do background check. see background_cut

        cut2: do rescan check. see rescan_cut

        cut3: do neighbor check. not yet implemented

        candidate_window: frequency in Hz around Larmor for which we check candidates

        samepeak_cut: distance in Hz within which candidates are merged

        background_cut: distance from background outliers within which candidates are cut

        rescan_cut: margin for candidates to appear within in rescans, otherwhise they're cut

        centerfreq: in Hz, central freq of the plot showing all candidates from all scans

        """

        print("FilterCandidate() start")

        scans = self.name
        larmors = []
        widths = []
        plotcolors = []
        # plotcolors = list([(0.8638877185589081, 0.39749148121791367, 0.12247407737249927),
        # (0.3421344976171118, 0.6526999917908147, 0.7767062824010426),
        # (0.12169239941374965, 0.7107835051991649, 0.03768516437474634)])
        plt.figure(figsize=(16, 9), dpi=100)

        for scanindex, scan in enumerate(scans):
            check(scan)
            measdatafile = f"{self.basepath}/{scan}/_DM_measurements_data.txt"
            Log = np.genfromtxt(
                measdatafile,
                unpack=True,
                delimiter=" ",
                skip_header=0,
                filling_values=0,
                invalid_raise=False,
            )
            larmors.append(Log[5])
            widths.append(Log[6])

            # construct candidate dictionaries from matlab files
            # divide resonant from background cands
            if (  # self.fCND == [] or self.gCND == []
                # or len(self.fCND) != len(self.gCND)
                loadfrommat
            ):
                print("loading info from associated matlab file")
                for runind, file in enumerate(self.allDMfiles[scanindex]):

                    if self.singlerunfile is not None:
                        matlogfile = self.singlerunfile
                    else:
                        matlogfile = rf"{self.basepath}/{scan}/AxionFinder results/{self.analysisname}/{runind+1}/{self.analysisname}_vars.mat"
                    mat_data = loadmat(matlogfile)
                    try:
                        fCND = (
                            mat_data["fCND_new"].flatten().tolist()
                        )  # all run cand fs
                        gCND = (
                            mat_data["gCND_new"].flatten().tolist()
                        )  # all run cand gs
                    except KeyError:
                        fCND = mat_data["fCND"].flatten().tolist()  # all run cand fs
                        gCND = mat_data["gCND"].flatten().tolist()  # all run cand gs
                    print(f"found {len(fCND)} outliers in run {runind} of scan {scan}")

                    run_larmor = larmors[scanindex][runind]
                    for i, candidate in enumerate(fCND):
                        if np.abs(candidate - run_larmor) < candidate_window:
                            self.cands_res.append([scan, runind, fCND[i], gCND[i]])
                        else:
                            self.cands_bkg.append([scan, runind, fCND[i], gCND[i]])

            else:  # FindCandidate() python function has already been run
                # print(f'found python outlier lists for scan {scan}')
                for runind, file in enumerate(self.allDMfiles[scanindex]):
                    fCND = self.fCND[scanindex]
                    gCND = self.gCND[scanindex]
                    run_larmor = larmors[scanindex][runind]
                    for candind, candfreq in enumerate(fCND[runind]):
                        if np.abs(candfreq - run_larmor) < candidate_window:
                            self.cands_res.append(
                                [
                                    scan,
                                    runind,
                                    fCND[runind][candind],
                                    gCND[runind][candind],
                                ]
                            )
                            # print(f'diff {np.abs(candfreq-run_larmor)} between candfreq {candfreq} and larmor {run_larmor} INSIDE window {candidate_window}')
                        else:
                            self.cands_bkg.append(
                                [
                                    scan,
                                    runind,
                                    fCND[runind][candind],
                                    gCND[runind][candind],
                                ]
                            )
                            # print(f'diff {np.abs(candfreq-run_larmor)} between candfreq {candfreq} and larmor {run_larmor} OUT OF window {candidate_window}')

            cands_filtered = self.cands_res
            print(
                f"keeping only outliers within a {candidate_window}Hz resonance window around larmor freq"
            )
            fvals_res = [cand[2] for cand in cands_filtered if cand[0] == scan]
            gvals_res = [cand[3] for cand in cands_filtered if cand[0] == scan]
            check(fvals_res)
            check(gvals_res)

            if len(plotcolors) < len(scans):
                plotcolors.append(get_random_color())
            # plt.scatter(thiscut_fvals-centerfreq, thisscan_gvals, s=180, color=plotcolor, label=f'Candidates from scan {scanindex+1}')
            # plt.plot(thiscut_fvals-centerfreq, thisscan_gvals,'p', markersize=25, markeredgewidth=1, markerfacecolor=plotcolor, markeredgecolor='black')
            plt.plot(
                fvals_res - centerfreq,
                gvals_res,
                marker="*",
                linestyle="None",
                markersize=30,
                color=plotcolors[scanindex],
                label=f"Candidates from scan {scan}",
            )

        # plot a number at each point stating the run where the cand was found
        for cand in cands_filtered:
            plt.text(
                cand[2] - centerfreq,
                cand[3],
                str(cand[1]),
                ha="center",
                va="center",
                fontsize=15,
                color="black",
            )

        plt.xlabel(f"ALP Compton frequency $-{int(np.round(centerfreq))}~[Hz]$")
        plt.ylabel(f"Candidate coupling $[GeV^{-1}]$")
        plt.legend(loc="upper right")
        plt.autoscale()
        plt.tight_layout()
        if len(scans) > 1:
            scanstr = "all"
        else:
            scanstr = scan
        if saveplot_opt:
            plt.savefig(
                f"{savepath}/AxionFinder_{self.analysisname}_Cands_cut_resonant_{scanstr}.png",
                bbox_inches="tight",
            )
        if showplot_opt:
            plt.show()
        plt.clf()
        plt.close()

        ################################################################################
        # merge any candidates too close to each other
        if cut0:
            print(f"merging outliers closer from each other than {samepeak_cut} Hz")
            thiscut_fvals = [cand[2] for cand in cands_filtered]
            # merged_fvals, merged_inds = merge_peak_bins(thiscut_fvals, threshold=samepeak_cut)
            merged_fvals = merge_close_entries2(thiscut_fvals, threshold=samepeak_cut)
            cands_merged = [cand for cand in cands_filtered if cand[2] in merged_fvals]
            cands_filtered = cands_merged
            # check(cands_filtered)

            plt.figure(figsize=(16, 9), dpi=100)
            for scanindex, scan in enumerate(scans):
                fvals_merge = [cand[2] for cand in cands_filtered if cand[0] == scan]
                gvals_merge = [cand[3] for cand in cands_filtered if cand[0] == scan]
                check(scan)
                check(fvals_merge)
                check(gvals_merge)
                plt.plot(
                    fvals_merge - centerfreq,
                    gvals_merge,
                    marker="*",
                    linestyle="None",
                    markersize=30,
                    color=plotcolors[scanindex],
                    label=f"Candidates from scan {scan}",
                )

            # plot a number at each point stating the run where the cand was found
            for cand in cands_filtered:
                plt.text(
                    cand[2] - centerfreq,
                    cand[3],
                    str(cand[1]),
                    ha="center",
                    va="center",
                    fontsize=15,
                    color="black",
                )

            plt.xlabel(f"ALP Compton frequency $-{int(np.round(centerfreq))}~[Hz]$")
            plt.ylabel(f"Candidate coupling $[GeV^{-1}]$")
            plt.legend(loc="upper right")
            plt.autoscale()
            plt.tight_layout()
            if len(scans) > 1:
                scanstr = "all"
            else:
                scanstr = scan
            if saveplot_opt:
                plt.savefig(
                    f"{self.basepath}/AxionFinder_{self.analysisname}_Cands_cut_merge_{scanstr}.png",
                    bbox_inches="tight",
                )
            if showplot_opt:
                plt.show()
            plt.clf()
            plt.close()

        ################################################################################
        # background check, cut if there are any bkg candidates close to a resonant candidate
        if cut1:
            print(
                f"removing outliers that appear off-resonance in other spectra within {background_cut}Hz"
            )
            cands_to_cut = []
            for candindex, cand_res in enumerate(cands_filtered):
                close_to_cand_within_bkg = [
                    cand_bkg
                    for cand_bkg in self.cands_bkg
                    if np.abs(cand_bkg[2] - cand_res[2]) <= background_cut
                ]
                # check(cand_res)
                # check(close_to_cand_within_bkg)
                if len(close_to_cand_within_bkg) > 0:
                    cands_to_cut.append(cand_res)
            cands_filtered = [
                cand for cand in cands_filtered if cand not in cands_to_cut
            ]

            plt.figure(figsize=(16, 9), dpi=100)
            for scanindex, scan in enumerate(scans):
                fvals_bkg = [cand[2] for cand in cands_filtered if cand[0] == scan]
                gvals_bkg = [cand[3] for cand in cands_filtered if cand[0] == scan]
                check(scan)
                check(fvals_bkg)
                check(gvals_bkg)
                plt.plot(
                    fvals_bkg - centerfreq,
                    gvals_bkg,
                    marker="*",
                    linestyle="None",
                    markersize=30,
                    color=plotcolors[scanindex],
                    label=f"Candidates from scan {scan}",
                )

            # plot a number at each point stating the run where the cand was found
            for cand in cands_filtered:
                plt.text(
                    cand[2] - centerfreq,
                    cand[3],
                    str(cand[1]),
                    ha="center",
                    va="center",
                    fontsize=15,
                    color="black",
                )
            plt.xlabel(f"ALP Compton frequency $-{int(np.round(centerfreq))}~[Hz]$")
            plt.ylabel(f"Candidate coupling $[GeV^{-1}]$")
            plt.legend(loc="upper right")
            plt.autoscale()
            plt.tight_layout()
            if len(scans) > 1:
                scanstr = "all"
            else:
                scanstr = scan
            if saveplot_opt:
                plt.savefig(
                    f"{self.basepath}/AxionFinder_{self.analysisname}_Cands_cut_bkg_{scanstr}.png",
                    bbox_inches="tight",
                )
            if showplot_opt:
                plt.show()
            plt.clf()
            plt.close()

        ################################################################################
        if compare_candidates:
            # plot each candidate frequency between all scans
            def return_spectrum_window(scan, run, centerfreq, width, convolved=False):

                if convolved == True:
                    matfile = rf"{self.basepath}/{scan}/AxionFinder results/{self.analysisname}/{run+1}/{self.analysisname}_vars.mat"
                    mat_data = loadmat(matfile)
                    freqs = (
                        mat_data["convolvedFreq"].flatten().tolist()
                    )  # frequencies array of convolved PSD
                    specs = (
                        mat_data["convolvedPeak"].flatten().tolist()
                    )  # convolved PSD values array
                    powerOfTenScaling = (
                        mat_data["powerOfTenScaling"].flatten().tolist()[0]
                    )
                    specs = [spec * (10**powerOfTenScaling) for spec in specs]

                elif convolved == False:
                    # laptop
                    freqfile = rf"C:\Users\Admin\Desktop\CASPEr-thermal_DMdata22\{scan}\PSDs_100s\AxionScan_{run}_frequencies_100s.bin"
                    specfile = rf"C:\Users\Admin\Desktop\CASPEr-thermal_DMdata22\{scan}\PSDs_100s\AxionScan_{run}_spectrum_100s.bin"
                    # SQUID PC
                    # freqfile = rf"G:\SQUID NMR\DM measurements\{scans[2]}\PSDs_100s\AxionScan_0_frequencies_100s.bin"
                    # specfile = rf"G:\SQUID NMR\DM measurements\{scans[2]}\PSDs_100s\AxionScan_0_spectrum_100s.bin"
                    freqs = read_double_precision_floats(freqfile)
                    specs = read_double_precision_floats(specfile)

                freq_window = [
                    freq
                    for freq in freqs
                    if (centerfreq - width / 2) <= freq <= (centerfreq + width / 2)
                ]
                spec_window = [
                    spec
                    for spec in specs
                    if (centerfreq - width / 2)
                    <= freqs[specs.index(spec)]
                    <= (centerfreq + width / 2)
                ]

                return freq_window, spec_window

            # get all candidates per scan
            rescan_inds = [1, 2]
            cands_main = [
                cands for cands in cands_filtered if cands[0] == mainscan_name
            ]
            num_plots = len(cands_main)
            num_columns = 2
            num_rows = (num_plots + num_columns - 1) // num_columns
            check(num_rows)
            # plt.figure(figsize=(16*0.9, 9*0.9), dpi=100)
            fig, axes = plt.subplots(
                num_rows, num_columns, figsize=(16 * 0.9, 9 * 0.9), dpi=100
            )
            plt.rc("font", size=10)
            specwidth = 20

            axes = axes.flatten()
            # if num_plots == 1: # If there is only one subplot, axes won't be a list
            #    axes = [axes]

            wind = AxionWind()
            for i, cand in enumerate(cands_main):
                # scan C spectrum & candidates
                print(f"now plotting plot{i}")
                freq_windowC, spec_windowC = return_spectrum_window(
                    scan=mainscan_name, run=cand[1], centerfreq=cand[2], width=specwidth
                )
                print("main spec window retrieved")
                axion_line = axion_lineshape(
                    wind.v_0,
                    wind.v_lab,
                    nu_a=cand[2],
                    nu=np.array(freq_windowC),
                    case="grad_perp",
                    alpha=np.pi / 2,
                )  # the axion lineshape (Gramolin paper)
                # ALP_linewidth = 1.3 # Hz
                # conv_line_len = int(3 * ALP_linewidth / abs(freq_windowC[1] - freq_windowC[0]))
                # conv_freqstamp, conv_PSD = AxionWind.MatchFilter(freq_windowC, spec_windowC, conv_step=1.0*ALP_linewidth, \
                #                                    conv_line=axion_line[len(freq_windowC)//2-10:len(freq_windowC)//2+ conv_line_len])

                nu_n, FWHM_n, amp_n, nu_n_err, FWHM_n_err, amp_n_err = ReturnNMRdata(
                    self.basepath, scan, cand[1]
                )
                year, month, day, time_str, datetime_int = ReturnMeasureTimFromFile(
                    file
                )  # 2022, 12, 23, '15:47:18'

                fakeAxion_Amp = wind.GetALP_Data(
                    year=year,
                    month=month,
                    day=day,
                    time_str=time_str,
                    datetime_int=datetime_int,
                    T_acq=0.5,
                    T2time=ReturnT2time(self.basepath, scan),
                    NMRfreq=[nu_n, nu_n_err],
                    NMRwidth=[FWHM_n, FWHM_n_err],
                    NMRamp=[amp_n, amp_n_err],
                    nu_ALP=i,
                    g_aNN=1,
                    usedailymod=True,
                )[9]

                axion_line *= fakeAxion_Amp * (cand[3] ** 2)
                # check(axion_line)
                mainscan_ind = scans.index(mainscan_name)
                axes[i].scatter(
                    np.array(freq_windowC) - cand[2],
                    spec_windowC,
                    label=cand[0],
                    color=plotcolors[mainscan_ind],
                )
                axes[i].plot(
                    0,
                    fakeAxion_Amp * (cand[3] ** 2),
                    marker="*",
                    linestyle="None",
                    markersize=25,
                    color=plotcolors[mainscan_ind],
                )
                axes[i].text(
                    0,
                    fakeAxion_Amp * (cand[3] ** 2),
                    str(f"{cand[1]}"),
                    ha="center",
                    va="center",
                    fontsize=10,
                    color="black",
                )
                axes[i].plot(
                    np.array(freq_windowC) - cand[2],
                    axion_line,
                    color="red",
                    linestyle="--",
                )
                # axes[i].plot(np.array(conv_freqstamp)-cand[2], conv_PSD, color='red', linestyle='--')

                # scan B
                freq_windowB, spec_windowB = return_spectrum_window(
                    scan=scans[rescan_inds[1]],
                    run=cand[1],
                    centerfreq=cand[2],
                    width=specwidth,
                )
                print("sub-1 spec window retrieved")
                axes[i].scatter(
                    np.array(freq_windowB) - cand[2],
                    spec_windowB,
                    label=scans[rescan_inds[1]],
                    color=plotcolors[rescan_inds[1]],
                )
                cands_B = [
                    candB
                    for candB in cands_filtered
                    if candB[0] == scans[rescan_inds[1]]
                    and candB[2] >= np.min(freq_windowC)
                    and candB[2] <= np.max(freq_windowC)
                ]
                cands_B_fvals = [candB[2] for candB in cands_B]
                cands_B_gvals = [candB[3] for candB in cands_B]
                cands_B_runs = [candB[1] for candB in cands_B]
                axes[i].plot(
                    np.array(cands_B_fvals) - cand[2],
                    fakeAxion_Amp * (np.array(cands_B_gvals) ** 2),
                    marker="*",
                    linestyle="None",
                    markersize=20,
                    color=plotcolors[rescan_inds[1]],
                )
                for candB in cands_B:
                    axes[i].text(
                        candB[2] - cand[2],
                        fakeAxion_Amp * (np.array(candB[3]) ** 2),
                        str(candB[1]),
                        ha="center",
                        va="center",
                        fontsize=10,
                        color="black",
                    )

                # scan A
                freq_windowA, spec_windowA = return_spectrum_window(
                    scan=scans[rescan_inds[0]],
                    run=cand[1],
                    centerfreq=cand[2],
                    width=specwidth,
                )
                print("sub-2 spec window retrieved")
                axes[i].scatter(
                    np.array(freq_windowA) - cand[2],
                    spec_windowA,
                    label=scans[rescan_inds[0]],
                    color=plotcolors[rescan_inds[0]],
                )
                cands_A = [
                    candA
                    for candA in cands_filtered
                    if candA[0] == scans[1]
                    and candA[2] >= np.min(freq_windowC)
                    and candA[2] <= np.max(freq_windowC)
                ]
                cands_A_fvals = [candA[2] for candA in cands_A]
                cands_A_gvals = [candA[3] for candA in cands_A]
                axes[i].plot(
                    np.array(cands_A_fvals) - cand[2],
                    fakeAxion_Amp * (np.array(cands_A_gvals) ** 2),
                    marker="*",
                    linestyle="None",
                    markersize=20,
                    color=plotcolors[rescan_inds[0]],
                )
                for candA in cands_A:
                    axes[i].text(
                        candA[2] - cand[2],
                        fakeAxion_Amp * (np.array(candA[3]) ** 2),
                        str(candA[1]),
                        ha="center",
                        va="center",
                        fontsize=10,
                        color="black",
                    )

                axes[i].set_xlabel(
                    f"Frequency $-{round(cand[2]*1e-8, 6)}~[MHz]$", fontsize=10
                )
                axes[i].set_ylabel("PSD", fontsize=10)
                axes[i].tick_params(axis="both", which="major", labelsize=10)
                axes[i].set_yscale("log")
                # axes[i].set_ylim(10e-14, np.max(spec_windowC)*(1+0.5))
                axes[i].legend()

            # Hide any unused subplots
            for j in range(i + 1, len(axes)):
                fig.delaxes(axes[j])
            plt.tight_layout()
            if saveplot_opt:
                plt.savefig(
                    f"{self.basepath}/AxionFinder_Cands_Comparison_Raw.png",
                    bbox_inches="tight",
                )
            if showplot_opt:
                plt.show()
            plt.close()

        # consistency check between all scans
        if cut2:
            consistent_sets = []
            if mainscan_name != None:
                print(f"using {mainscan_name} as the main scan")
                cands_mainscan = [
                    cand for cand in cands_filtered if cand[0] == mainscan_name
                ]
                cands_rescans = [
                    cand for cand in cands_filtered if cand[0] != mainscan_name
                ]

                for mainscan_cand in cands_mainscan:
                    consistent_rescan_cands = [
                        rescan_cand
                        for rescan_cand in cands_rescans
                        if np.abs(rescan_cand[2] - mainscan_cand[2]) <= rescan_cut
                    ]
                    if len(consistent_rescan_cands) > 0:
                        if mainscan_cand not in consistent_rescan_cands:
                            consistent_rescan_cands.append(mainscan_cand)
                        consistent_sets.append(consistent_rescan_cands)

            else:
                # we need to make a list of lists for candidates from each scan separately
                cands_sorted_by_scans = []
                for scan in scans:
                    thisscan_fvals = [
                        cand[2] for cand in cands_filtered if cand[0] == scan
                    ]
                    cands_sorted_by_scans.append(thisscan_fvals)
                consistent_freqs = find_matching_tuples(
                    cands_sorted_by_scans, margin=rescan_cut
                )  # rescan_cut # consistent set appears at 38Hz margin
                check(consistent_freqs)
                # find the indices of the consistent candidates
                for ntuple in consistent_freqs:
                    cands = []
                    for freq in ntuple:
                        scan = [cand[0] for cand in cands_filtered if cand[2] == freq][
                            0
                        ]
                        run = [cand[1] for cand in cands_filtered if cand[2] == freq][0]
                        gval = [cand[3] for cand in cands_filtered if cand[2] == freq][
                            0
                        ]
                        cands.append([scan, run, freq, gval])
                    consistent_sets.append(cands)

            print(
                f"checked for sets of outliers consistent between main- and rescans to within {rescan_cut} Hz"
            )
            check(consistent_sets)
            self.cands_filtered = consistent_sets

            plt.figure(figsize=(16, 9), dpi=100)
            for scanindex, scan in enumerate(scans):
                thisscan_fvals = [cand[2] for cand in cands_filtered if cand[0] == scan]
                thisscan_gvals = [cand[3] for cand in cands_filtered if cand[0] == scan]
                plt.plot(
                    thisscan_fvals - centerfreq,
                    thisscan_gvals,
                    marker="*",
                    linestyle="None",
                    markersize=30,
                    color=plotcolors[scanindex],
                    label=f"Candidates from scan {scan}",
                )
            for ntuple in consistent_sets:
                for cand in ntuple:
                    # plt.plot(cand[2]-centerfreq, cand[3], marker='*', linestyle='None', markersize=30, color=plotcolors[scans.index(cand[0])]) #label=f'Candidates from scan {cand[0]}'
                    plt.text(
                        cand[2] - centerfreq,
                        cand[3],
                        str(f"{cand[0]}-{cand[1]}"),
                        ha="center",
                        va="center",
                        fontsize=15,
                        color="black",
                    )

            plt.xlabel(f"Larmor frequency $-{int(np.round(centerfreq))}~[Hz]$")
            plt.ylabel(f"Candidate coupling $[GeV^{-1}]$")
            plt.legend(loc="upper right")
            plt.autoscale()
            plt.tight_layout()
            if len(scans) > 1:
                scanstr = "all"
            else:
                scanstr = scan
            if saveplot_opt:
                plt.savefig(
                    f"{self.basepath}/AxionFinder_{self.analysisname}_Cands_cut_rescan_{scanstr}.png",
                    bbox_inches="tight",
                )
            if showplot_opt:
                plt.show()
            plt.clf()
            plt.close()

        consistent_mainscan_cand = [
            cand
            for cand_tuple in consistent_sets
            for cand in cand_tuple
            if cand[0] == mainscan_name
        ]

        if save_to_excel:
            cand_data = {
                "scan": np.array([cand[0] for cand in consistent_mainscan_cand]),
                "run": np.array([cand[1] for cand in consistent_mainscan_cand]),
                "frequency": np.array([cand[2] for cand in consistent_mainscan_cand]),
                "coupling": np.array([cand[3] for cand in consistent_mainscan_cand]),
            }
            df = pd.DataFrame(cand_data)
            df.to_excel(rf"{savepath}/candidates_filtered.xlsx")

    def GetExclusion(
        self,
        load_from_where=0,
        limitfilepath="",
        use_modulation=True,
        merge_limits=False,
        limit_errs=False,
        corr_SGefficiency=1.0,
        corr_BlindAxRecovery=1.0,
        corr_rescans=False,
        mainscan_name="",
        candidates_log="",
        scan_labels=[
            "Main scan",
            "Rescan A",
            "Rescan B",
        ],
        saveplot_opt=False,
        showplot_opt=False,
        savepath="",
        plotcolors=list(
            [
                (0.8638877185589081, 0.39749148121791367, 0.12247407737249927),
                (0.3421344976171118, 0.6526999917908147, 0.7767062824010426),
                (0.12169239941374965, 0.7107835051991649, 0.03768516437474634),
            ]
        ),
        coupling_unit="GeV",
    ):
        """
        make the exclusion plot determined by the measurement


        scan_labels: what to call the scans (specified when initializing the dmScan class) in the legend of this plot. for example, main scan, rescan A, B, C,..

        load_from_where:
            * 0: mat files, and write limits into excel files for easy plotting later on. this only needs to be done once
            * 1: recalculate the limits here from the optimally filtered PSD threshold
            * 2: excel files in the repository. use this normally

        use_modulation: whether to account for the daily changing axion wind angle at the time of measurement, or use the statistical factor sqrt(1/3)

        merge_limits: whether to calculate a merged limit of all inputted scans as the weighted average

        limit_errs: calculate uncertainties from propagation of measured quantities and plot as error bars

        corr_SGefficiency: multiplicative correction factor for the efficiency of the SG filter. we found from the MC simuations a factor of eta_SG = 88.2+-4.6% => sqrt(eta_SG) = 93.9+-2.5%. so this factor will be defined as 0.939

        corr_BlindAxRecovery: multiplicative correction factor for the efficiency of recovering injected fake axion signals. we found a recovery bias of 1.420, so this factor is defined as sqrt(1/1.42)

        corr_rescans: if signal candidates have been found in the main scan, reduce the sensitivity around their frequencies. instead, use the average sensitivity of all rescans

        candidates_log: in case rescan candidates have been found. reduce exclusion to rescan limits at their frequencies

        coupling_unit: inverse eV or GeV etc.

        """

        print("GetExclusion() start")

        coupling_limits = []
        centerfreqs = []
        scans = self.name

        if coupling_unit == "eV":
            unitfactor = 1e-9
        elif coupling_unit == "MeV":
            unitfactor = 1e-3
        elif coupling_unit == "GeV":
            unitfactor = 1
        else:
            print("unknown coupling unit specified")

        for scanindex, scan in enumerate(scans):

            flims = []
            glims = []
            glims_errs = []

            measdatafile = rf"{self.basepath}/{scan}/_DM_measurements_data.txt"
            measLog = np.genfromtxt(
                measdatafile,
                unpack=True,
                delimiter=" ",
                skip_header=0,
                filling_values=0,
                invalid_raise=False,
            )
            larmors = measLog[5].tolist()
            centerfreqs.append(int(np.round(larmors[0], 0)))

            if load_from_where == 2:  # load from previously saved txt files
                limitfile = rf"{limitfilepath}/exclusion_limits_{scan}.txt"
                limitlog = np.genfromtxt(
                    limitfile,
                    unpack=True,
                    delimiter=" ",
                    skip_header=1,
                    filling_values=0,
                    invalid_raise=False,
                )
                for step in range(len(limitlog[0])):
                    flim = limitlog[0][step]
                    glim = limitlog[1][step]
                    coupling_limits.append([scan, step, flim, glim])

            else:
                for run, file in enumerate(self.allDMfiles[scanindex]):
                    matfile = rf"{self.basepath}/{scan}/AxionFinder results/{self.analysisname}/{run+1}/{self.analysisname}_vars.mat"
                    mat_data = loadmat(matfile)

                    if load_from_where == 0:  # load data from mat logfiles
                        flim = larmors[run]
                        # flim = mat_data['fLIM'].flatten()[0] # center frequency of this run's analysis window
                        glim = mat_data["gLIM"].flatten()[
                            0
                        ]  # calculated coupling for 5sigma signal threshold
                        # check(flim)
                    elif load_from_where == 1:  # recalculate now
                        flim = mat_data["fLIM"].flatten()[
                            0
                        ]  # center frequency of this run's analysis window

                        if self.singlerunfile is not None:
                            nu_n, FWHM_n, amp_n, nu_n_err, FWHM_n_err, amp_n_err = (
                                ReturnNMRdata(self.basepath, scan, self.singlerunindex)
                            )
                            year, month, day, time_str, datetime_int = (
                                ReturnMeasureTimFromFile(
                                    rf"{self.basepath}/{scan}/AxionScan_{self.singlerunindex}.h5"
                                )
                            )
                            AxionParams = (
                                mat_data["fakeAxionParameters"].flatten().tolist()
                            )
                            fakeAxion_Amp = AxionParams[5]
                        else:
                            nu_n, FWHM_n, amp_n, nu_n_err, FWHM_n_err, amp_n_err = (
                                ReturnNMRdata(self.basepath, scan, run)
                            )
                            year, month, day, time_str, datetime_int = (
                                ReturnMeasureTimFromFile(file)
                            )  # 2022, 12, 23, '15:47:18'
                            fakeAxion_Amp = AxionWind().GetALP_Data(
                                year=year,
                                month=month,
                                day=day,
                                time_str=time_str,
                                datetime_int=datetime_int,
                                T_acq=0.5,
                                T2time=ReturnT2time(self.basepath, scan),
                                NMRfreq=[nu_n, nu_n_err],
                                NMRwidth=[FWHM_n, FWHM_n_err],
                                NMRamp=[amp_n, amp_n_err],
                                nu_ALP=flim,
                                g_aNN=1,
                                usedailymod=use_modulation,
                            )[9]

                        # SNRtarget = self.threshold_exclude[run] * (10**self.powerOfTenScaling)
                        # glim_mat = np.sqrt(SNRtarget / fakeAxion_Amp)
                        # check(glim_mat)
                        # std_PSD = mat_data['filteredPSDstd'].flatten()[0]
                        std_convPSD_mat = mat_data["convolvedPSDstd"].flatten()[0] * (
                            10**self.powerOfTenScaling
                        )  # don't use this, it's nonsense if the matlab fit fails
                        check(std_convPSD_mat)
                        std_convPSD = np.std(
                            mat_data["convolvedPeak"].flatten().tolist()
                        ) * (10**self.powerOfTenScaling)
                        check(std_convPSD)
                        glim = np.sqrt(5.0 * std_convPSD / fakeAxion_Amp)
                        check(glim)
                        if np.abs(glim) > 5 * np.mean(glims) and len(glims) > 1:
                            glim = np.mean(glims)  # something went wrong

                    # account for SG filter efficiency: eta_SG = 88.2+-4.6% => sqrt(eta_SG) = 93.9+-2.5%
                    glim *= corr_SGefficiency
                    # account for signal suppression (tested with fake ALPs)
                    glim *= corr_BlindAxRecovery

                    if limit_errs:
                        if self.singlerunfile is not None:
                            nu_n, FWHM_n, amp_n, nu_n_err, FWHM_n_err, amp_n_err = (
                                ReturnNMRdata(self.basepath, scan, self.singlerunindex)
                            )
                            year, month, day, time_str, datetime_int = (
                                ReturnMeasureTimFromFile(
                                    rf"{self.basepath}/{scan}/AxionScan_{self.singlerunindex}.h5"
                                )
                            )
                        else:
                            nu_n, FWHM_n, amp_n, nu_n_err, FWHM_n_err, amp_n_err = (
                                ReturnNMRdata(self.basepath, scan, run)
                            )
                            year, month, day, time_str, datetime_int = (
                                ReturnMeasureTimFromFile(file)
                            )  # 2022, 12, 23, '15:47:18'

                        ax_data = AxionWind().GetALP_Data(
                            year=year,
                            month=month,
                            day=day,
                            time_str=time_str,
                            datetime_int=datetime_int,
                            T_acq=0.5,
                            T2time=ReturnT2time(self.basepath, scan),
                            NMRfreq=[nu_n, nu_n_err],
                            NMRwidth=[FWHM_n, FWHM_n_err],
                            NMRamp=[amp_n, amp_n_err],
                            nu_ALP=flim,
                            g_aNN=1,
                            usedailymod=use_modulation,
                            returnerrors=True,
                        )
                        ax_amp = ax_data[9]
                        ax_amp_err = ax_data[-1][-1]
                        # check(ax_amp)
                        # check(ax_amp_err)

                        # glim = np.sqrt( 5.0 * std_convPSD / fakeAxion_Amp)
                        std_err = 0  # no idea
                        dglim_dstd = 2.5 / (glim * ax_amp)
                        dglim_dAmp = -glim / (2 * ax_amp)

                        glim_err = np.sqrt(
                            (dglim_dstd * std_err) ** 2 + (dglim_dAmp * ax_amp_err) ** 2
                        )

                        # check(glim)
                        # check(glim_err)

                    else:
                        glim_err = 0

                    flims.append(flim)
                    glims.append(glim)
                    glims_errs.append(glim_err)
                    coupling_limits.append([scan, run, flim, glim, glim_err])
                    print_progress_bar(
                        run,
                        total=len(self.allDMfiles[scanindex]),
                        prefix="Progress",
                        suffix="Complete",
                        length=50,
                    )

                np.savetxt(
                    rf"{savepath}/exclusion_limits_{scan}.txt",
                    np.column_stack((flims, glims, glims_errs)),
                    header="Frequency (Hz) & g_aNN_lim (GeV^-1) & Delta g_aNN_lim",
                )
                check(glims)

        # check(coupling_limits)
        glims_mainscan = [
            limit[3] for limit in coupling_limits if limit[0] == mainscan_name
        ]

        # this is not fully working and we do not merge limits anymore
        if merge_limits:
            print("now merging limits")
            glims_merged_prop = []
            glims_merged_meas = []
            glims_merged_err = []
            glims_merged_SNR = []

            if load_from_where == 1:
                for index in range(len(coupling_limits[0])):
                    top = (
                        (coupling_limits[0][index] / glims_errs[0][index] ** 2)
                        + (coupling_limits[1][index] / glims_errs[1][index] ** 2)
                        + (coupling_limits[2][index] / glims_errs[2][index] ** 2)
                    )
                    down = (
                        (1 / glims_errs[0][index] ** 2)
                        + (1 / glims_errs[1][index] ** 2)
                        + (1 / glims_errs[2][index] ** 2)
                    )
                    glims_merged_meas.append(top / down)
                    glims_merged_err.append(down ** (-1 / 2))
                    glims_merged_SNR.append(
                        np.sqrt(
                            5
                            / np.sqrt(
                                (5 / coupling_limits[0][index] ** 2) ** 2
                                + (5 / coupling_limits[1][index] ** 2) ** 2
                                + (5 / coupling_limits[2][index] ** 2) ** 2
                            )
                        )
                    )
                    glims_merged_prop.append(
                        (
                            1
                            / (
                                1 / coupling_limits[0][index] ** 4
                                + 1 / coupling_limits[1][index] ** 4
                                + 1 / coupling_limits[2][index] ** 4
                            )
                        )
                        ** (1 / 4)
                    )

                np.savetxt(
                    rf"{savepath}/exclusion_limt_combined.txt",
                    np.column_stack((flims, glims_merged_prop)),
                    header="Frequency (Hz) g_aNN_lim (GeV^-1)",
                )

            elif load_from_where == 2:
                limitcombfile = rf"{savepath}/exclusion_limt_combined.txt"
                limitcomblog = np.genfromtxt(
                    limitcombfile,
                    unpack=True,
                    delimiter=" ",
                    skip_header=1,
                    filling_values=0,
                    invalid_raise=False,
                )
                for step in range(len(limitcomblog)):
                    glims_merged_prop.append(limitcomblog[step][1])

            check(glims_merged_prop)

        # check if there are candidates in the main scan at any step. if yes, replace the limit here with the average limit of the rescans
        if corr_rescans and len(scans) > 1 and candidates_log is not None:
            print("correcting main scan limits for candidates")

            couplings_rescan = []
            for scan in [
                scan for scan in scans if scan != mainscan_name
            ]:  # all rescans
                couplings_rescan.append(
                    [
                        [limit[0], limit[1], limit[2], limit[3]]
                        for limit in coupling_limits
                        if limit[0] == scan
                    ]
                )
            # check(couplings_rescan)

            candidates_log = pd.read_excel(rf"{candidates_log}")
            scan = candidates_log["scan"].values
            run = candidates_log["run"].values
            freq = candidates_log["frequency"].values
            gval = candidates_log["coupling"].values
            candidates_list = []
            for index, i in enumerate(freq):
                candidates_list.append(
                    [scan[index], run[index], freq[index], gval[index]]
                )
            # check(candidates_list)

            for cand_runindex in list(
                {cand[1] for cand in candidates_list if cand[0] == mainscan_name}
            ):  # all runs in the main scan where candidates were found
                limits_rescan = []
                for rescan in couplings_rescan:
                    limits_rescan.append(
                        [
                            coupling[3]
                            for coupling in rescan
                            if coupling[1] == cand_runindex
                        ]
                    )
                glims_mainscan[cand_runindex] = np.mean(limits_rescan)

        # save the (corrected) main scan sensitivity as a list
        ALP_masses = [
            limit[2] * AxionWind().h_eV
            for limit in coupling_limits
            if limit[0] == mainscan_name
        ]
        # check(ALP_masses)
        check(glims_mainscan)
        np.savetxt(
            rf"{savepath}/exclusion_limits_main.txt",
            np.column_stack((ALP_masses, glims_mainscan)),
            header="CASPEr-gradient low-field\n m_a [eV]  g_an/m_n [GeV^-1]",
        )

        # start plotting
        if saveplot_opt:
            plt.close()
            plt.rc("font", size=20)
            plt.rcParams["font.family"] = "Times New Roman"
            plt.rcParams["mathtext.fontset"] = "cm"
            fig, ax = plt.subplots(figsize=(16 * 0.5, 9 * 0.5), dpi=300)

            # iterate from last to first scan, filling the space in between with the right color
            # (if the main scan is loaded last: start from len(scans)-1 and go down by in the while loop, else the last color will overwrite the previous ones)
            max = (1 + 0.1) * np.amax([limit[3] for limit in coupling_limits])

            for scanindex, scan in enumerate(scans):
                flims_thisscan = [
                    limit[2] for limit in coupling_limits if limit[0] == scan
                ]
                glims_thisscan = [
                    limit[3] for limit in coupling_limits if limit[0] == scan
                ]
                gerrs_thisscan = [
                    limit[4] for limit in coupling_limits if limit[0] == scan
                ]

                plotcolors.append(get_random_color())
                # plotletter = chr(97+scanindex)
                # plt.plot(np.array(flims_thisscan)-centerfreqs[scanindex], np.array(glims_thisscan)*unitfactor, marker='o', color=plotcolors[scanindex], linestyle='-', label=f"Exclusion for {scan_labels[scanindex]}")
                ax.errorbar(
                    np.array(flims_thisscan) - centerfreqs[scanindex],
                    np.array(glims_thisscan) * unitfactor,
                    yerr=gerrs_thisscan,
                    fmt="o",
                    color=plotcolors[scanindex],
                    linestyle="-",
                    label=f"{scan_labels[scanindex]}",
                )

            # if merge_limits:
            # plt.errorbar(flims_all[1]-centerfreq, np.array(glims_merged_meas), yerr=np.array(glims_merged_err), fmt='o', color=get_random_color(), linestyle='-', label=f"Combined exclusion A")
            # plt.plot(flims_all[0]-centerfreqs[0], np.array(glims_merged_prop), marker='o', linestyle='-', label=f"Combined exclusion")

            scan_ind = len(scans) - 1
            while scan_ind >= 0:
                flims_thisscan = [
                    limit[2] for limit in coupling_limits if limit[0] == scans[scan_ind]
                ]
                glims_thisscan = [
                    limit[3] for limit in coupling_limits if limit[0] == scans[scan_ind]
                ]
                ax.fill_between(
                    np.array(flims_thisscan) - centerfreqs[scan_ind],
                    np.array(glims_thisscan) * unitfactor,
                    max,
                    color=plotcolors[scan_ind],
                    alpha=0.3,
                )
                scan_ind -= 1

            plt.yscale("log")
            # plt.xlim(0, 250)
            # coupling_limits = [limit[3] for limit in coupling_limits]
            # plt.ylim(0, np.max(coupling_limits)*unitfactor)
            # plt.ylim(1e-2, 1e-1)

            # plt.gca().yaxis.set_major_formatter(ScalarFormatter(useMathText=False))
            # plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.2f'))  # 2 decimal places
            # plt.gca().yaxis.get_major_formatter().set_scientific(False)
            # plt.gca().yaxis.get_major_formatter().set_useOffset(False)
            # from matplotlib.ticker import FuncFormatter
            # def decimal_formatter(x, pos): return f'{x:.2f}'
            # ax.yaxis.set_major_formatter(FuncFormatter(decimal_formatter))
            # plt.tick_params(axis='y', labelsize=8)
            # ax.tick_params(axis='y', labelsize=8)

            # plt.xlabel(rf"Frequency, $\nu_a$-{centerfreqs[0]/1e3} [kHz]")
            # plt.xlabel(rf"Frequency, $\nu_a$-{centerfreqs[0]} [Hz]")
            plt.xlabel(rf"Frequency, $\nu_a$-{1348450} [Hz]")
            plt.ylabel(
                rf"$|g_{{\mathrm{{ap}}}}|~[\mathrm{{GeV}}^{-1}]$"
            )  # ~[{coupling_unit}^{-1}]$") # g_{{\mathrm{{aNN}}}}
            plt.legend(loc="best")
            # plt.grid()
            # plt.autoscale()
            plt.tight_layout()
            if use_modulation:
                plt.savefig(rf"{savepath}/limitsplot.png", bbox_inches="tight")
            else:
                plt.savefig(
                    rf"{savepath}/limitsplot_no_modulation.png", bbox_inches="tight"
                )
            if showplot_opt:
                plt.show()
            plt.clf()

    def GetFakeAxionRecovery(
        self,
        scan="",
        mat_analysisname="",
        savepath="",
        check_cands=False,
        identify_cand_by=1,
        is_stoc_signal=False,
        is_SG_scan=False,
        compare_SGstepsize=False,
        plotopt=1,
        psdplotpath="",
        corr_SGefficiency=1,
        showplot=True,
    ):
        """
        looks at the results of a fake axion scan where multiple couplings have been injected.

        plots the efficiency of recovering their coupling through the matlab analysis.



        scan: the scan folder within basepath. str or list of str

        mat_analysisname: the folder containing matlab analysis output files
            * fakeAxionScan2: sg fit with non-central chi^2
            * fakeAxionScan3: direct sg filter output
            * fakeAxionScan4: sg size 0.3, random freqs
            * fakeAxionScan5: sg size 0.5, same freq
            * fakeAxionScan6: sg filter off, same freq

        savepath: where to save the recovery logs files (xlsx) and plots

        check cands:
            * if false, just compare optimally filtered PSD at ALP injection point with the input coupling
            * if true, look at data above detection threshold and find the outlier point most likely belonging to the injected signal

        identify_cand_by:
            * 1: freq closest to input freq     <- this is where the injected axion peak *should* end up after convolution
            * 2: freq closest to input freq + 0.5 axion width   <- closer to original line shape
            * 3: coupling closest to input coupling     <- does not care about frequency, don't use

        plotopt:
            * 0: just plot input vs output coupling
            * 1: also plot an example PSD with the fake axion injection (for article)
            * 2: plot a 2x2 of coupling and frequency recovery
        """

        def linear_func(x, a):
            return a * x

        def linear_func_offset(x, a, b):
            return a * x + b

        def find_first_differing_element(numbers, threshold):
            for i in range(1, len(numbers)):
                for j in range(i):
                    if abs(numbers[i] - numbers[j]) > threshold:
                        return numbers[i]
            return None

        def find_first_differing_element2(numbers_is, numbers_should, threshold):
            for i in range(1, len(numbers_is)):
                if abs(numbers_is[i] - numbers_should[i]) > threshold:
                    return numbers_is[i]
            return None

        print("GetFakeAxionRecovery() start")

        if scan == "" or scan == None:
            scan = self.name
        else:
            if isinstance(scan, str):
                scan = [scan]
            # print('single scan input')
            elif isinstance(scan, list) and all(isinstance(i, str) for i in scan):
                scan = scan
            # print('list of scans input')
            else:
                raise OSError(
                    "no valid scan inputted. must be string or list of strings."
                )

        for subscan in scan:
            if compare_SGstepsize:
                scanpath = (
                    rf"{self.basepath}/{subscan}/AxionFinder results/fakeAxionScan1"
                )
                SGstepsizes = []
            else:
                scanpath = (
                    rf"{self.basepath}/{subscan}/AxionFinder results/{mat_analysisname}"
                )
            check(scanpath)
            all_subfolders = [
                f for f in glob.glob(os.path.join(scanpath, "*/")) if os.path.isdir(f)
            ]
            run_directories = [
                os.path.basename(folder[:-1])
                for folder in all_subfolders
                if "fakeAxion" in os.path.basename(folder[:-1])
            ]
            # if all(not run_directory for run_directory in run_directories):
            run_directories = run_directories + [
                os.path.basename(folder[:-1])
                for folder in all_subfolders
                if "AxionScan" in os.path.basename(folder[:-1])
            ]

            for run_num, run in enumerate(run_directories):
                # retrieve input freq & coupling
                mat_data = loadmat(glob.glob(scanpath + "/" + run + "/*/*vars.mat")[0])
                fakeAxParams = mat_data["fakeAxionParameters"].flatten().tolist()
                self.fakeAx_fIN.append(fakeAxParams[0])
                self.fakeAx_gIN.append(fakeAxParams[1])

                # retrieve output freq & coupling
                if check_cands:
                    fcands = mat_data["fCND"].flatten().tolist()  # all run cand freqs
                    gcands = (
                        mat_data["gCND"].flatten().tolist()
                    )  # all run cand couplings
                    cands_thisrun = []  # make pairs for easy indexing
                    for index, i in enumerate(fcands):
                        cands_thisrun.append([fcands[index], gcands[index]])

                    if len(cands_thisrun) < 1:
                        self.fakeAx_fOUT.append(0)
                        self.fakeAx_gOUT.append(0)
                    elif len(cands_thisrun) > 1:
                        fcands = [cand[0] for cand in cands_thisrun]
                        gcands = [cand[1] for cand in cands_thisrun]

                        if identify_cand_by == 1:
                            closestcand_ind = min(
                                range(len(fcands)),
                                key=lambda i: abs(fcands[i] - self.fakeAx_fIN[run_num]),
                            )
                        elif identify_cand_by == 2:
                            axion_tau = 10**6 / self.fakeAx_fIN[run_num]
                            axion_width = 2 / axion_tau
                            axionpeakfreq = self.fakeAx_fIN[run_num] + axion_width / 2
                            # check(axionpeakfreq)
                            closestcand_ind = min(
                                range(len(fcands)),
                                key=lambda i: abs(fcands[i] - axionpeakfreq),
                            )
                        elif identify_cand_by == 3:
                            # closestcand_ind = np.argmin([abs(self.fakeAx_fIN[run_num] - gcand) for gcand in gcands])
                            closestcand_ind = min(
                                range(len(gcands)),
                                key=lambda i: abs(gcands[i] - self.fakeAx_gIN[run_num]),
                            )

                        truecand = cands_thisrun[closestcand_ind]
                        print(f"injected axion frequency: {self.fakeAx_fIN[run_num]}")
                        print(
                            f"closest cand found within {abs(self.fakeAx_fIN[run_num] - fcands[closestcand_ind])} Hz at: {truecand}"
                        )
                        self.fakeAx_fOUT.append(truecand[0])
                        self.fakeAx_gOUT.append(truecand[1] * corr_SGefficiency)

                    else:
                        self.fakeAx_fOUT.append(fcands)
                        self.fakeAx_gOUT.append(gcands * corr_SGefficiency)

                    cands_str = "_cands"

                else:  # just look at filtered PSD value
                    try:
                        fout = mat_data["fakeAxion_fOut"].flatten().tolist()[0]
                        # pout = mat_data['fakeAxion_pOut'].flatten().tolist()[0]
                    except KeyError:
                        fout = mat_data["fakeAxion_convolvedFreq"].flatten().tolist()[0]
                        # pout = mat_data['fakeAxion_convolvedPeak'].flatten().tolist()[0]
                    try:
                        gout = mat_data["fakeAxion_gOut"].flatten().tolist()[0]
                    except KeyError:
                        gout = (
                            mat_data["fakeAxion_convolvedCoupling"]
                            .flatten()
                            .tolist()[0]
                        )

                    if (
                        gout.real == 0 and gout.imag != 0
                    ):  # this happens if the value of the convolution is negative
                        print(f"\n imaginary coupling: {gout}")
                        # gout = -gout.imag
                        gout = 0

                    self.fakeAx_fOUT.append(fout)
                    self.fakeAx_gOUT.append(gout * corr_SGefficiency)

                    cands_str = "_nocands"

                if compare_SGstepsize:
                    SGparams = mat_data["sgParameters"].flatten().tolist()
                    SGstepsizes.append([SGparams[1], run])

                self.fakeAx_frecovery.append(
                    [self.fakeAx_fIN[run_num], self.fakeAx_fOUT[run_num]]
                )
                self.fakeAx_grecovery.append(
                    [self.fakeAx_gIN[run_num], self.fakeAx_gOUT[run_num]]
                )
                # check(self.fakeAx_frecovery)
                # check(self.fakeAx_grecovery)

            print_progress_bar(
                run_num,
                total=len(run_directories),
                prefix="Progress",
                suffix="Complete",
                length=50,
            )

            # for smooth axline: write recovery into excel (uncomment first 2 lines) if running a smooth axline analysis, or read from excel & for plotting
            if not is_stoc_signal:
                smoothaxdata = pd.DataFrame(
                    self.fakeAx_grecovery, columns=["g_in", "g_out"]
                )
                if is_SG_scan:
                    smoothaxdata.to_excel(
                        rf"{savepath}/fakeAxionScanTest_SG_smooth.xlsx", index=False
                    )
                else:
                    smoothaxdata.to_excel(
                        rf"{savepath}/fakeAxionScanTest_smooth.xlsx", index=False
                    )
            else:
                if is_SG_scan:
                    smoothaxdata = pd.read_excel(
                        rf"{savepath}/fakeAxionScanTest_SG_smooth.xlsx"
                    )
                else:
                    smoothaxdata = pd.read_excel(
                        rf"{savepath}/fakeAxionScanTest_smooth.xlsx"
                    )
            fakeAx_grecovery_smooth = smoothaxdata.values.tolist()

            if compare_SGstepsize:

                stepsizes = []
                f_diffs = []
                g_diffs = []
                for stepsize in SGstepsizes:
                    f_IN = [
                        cand[0] for cand in self.fakeAx_fIN if cand[2] == stepsize[1]
                    ][0]
                    f_OUT = [
                        cand[0] for cand in self.fakeAx_fOUT if cand[2] == stepsize[1]
                    ][0]
                    g_IN = [
                        cand[1] for cand in self.fakeAx_fIN if cand[2] == stepsize[1]
                    ][1]
                    g_OUT = [
                        cand[1] for cand in self.fakeAx_fOUT if cand[2] == stepsize[1]
                    ][1]
                    stepsizes.append(stepsize[0])
                    f_diffs.append(np.abs(f_IN - f_OUT) / f_IN)
                    g_diffs.append(np.abs(g_IN - g_OUT) / g_IN)
                    # f_diffs.append(f_IN-f_OUT)
                    # g_diffs.append(g_IN-g_OUT)

                fig, ax1 = plt.subplots(figsize=(16 * 0.9, 9 * 0.9), dpi=100)
                ax1.scatter(
                    stepsizes, f_diffs, s=180, color="blue", alpha=0.9, linewidth=1
                )
                ax1.set_xlabel(r"Narrow-SG step size (factors of ALP linewidth)")
                ax1.set_ylabel(r"Recovered frequency error")
                ax1.set_ylim(np.min(f_diffs) * (1 - 1e-3), np.max(f_diffs) * (1 + 1e-3))
                ax1.tick_params(axis="y", labelcolor="b")

                ax2 = ax1.twinx()
                ax2.scatter(
                    stepsizes, g_diffs, s=180, color="green", alpha=0.9, linewidth=1
                )
                ax2.set_ylabel(r"Recovered coupling error")
                ax2.tick_params(axis="y", labelcolor="g")

                plt.gca().xaxis.set_major_locator(MultipleLocator(0.1))
                plt.gca().xaxis.set_minor_locator(MultipleLocator(0.01))
                # plt.xticks(rotation=45)
                # plt.autoscale()
                plt.tight_layout()
                plt.title(
                    rf"Fake ALP $f={self.fakeAx_fIN[0][0]}$, $g={self.fakeAx_gIN[0][1]}$"
                )
                plt.savefig(
                    rf"{savepath}/FakeAxion_SGstepsizes.png", bbox_inches="tight"
                )
                plt.show()
                plt.clf()

            else:
                jump_thresh = 0.005  # small recovered couplings that vary less than this are considered to be below the detection cutoff
                # first_nonzero_index = next(i for i, val in enumerate(self.fakeAx_gOUT) if val != 0) # does not work
                diff_thresh = 0.005  # recovered coupling should not differ from input more than this

                # fuzzy axline recovery
                # first_jump = find_first_differing_element(self.fakeAx_gOUT,jump_thresh)
                first_jump = find_first_differing_element2(
                    self.fakeAx_gOUT, self.fakeAx_gIN, diff_thresh
                )
                # cutoff_lo = self.fakeAx_gIN[self.fakeAx_gOUT.index(first_jump)]
                # check(cutoff_lo)
                cutoff_lo = 0.030
                cutoff_hi = 0.500
                fitfunc = linear_func

                gvals_x = np.array(
                    [
                        gval[0]
                        for gval in self.fakeAx_grecovery
                        if gval[0] >= cutoff_lo and gval[0] <= cutoff_hi
                    ]
                )
                gvals_y = np.array(
                    [
                        gval[1]
                        for gval in self.fakeAx_grecovery
                        if gval[0] >= cutoff_lo and gval[0] <= cutoff_hi
                    ]
                )
                fvals_x = np.array(
                    [
                        fval[0]
                        for index, fval in enumerate(self.fakeAx_frecovery)
                        if self.fakeAx_grecovery[index][0] >= cutoff_lo
                    ]
                )
                fvals_y = np.array(
                    [
                        fval[1]
                        for index, fval in enumerate(self.fakeAx_frecovery)
                        if self.fakeAx_grecovery[index][0] >= cutoff_lo
                    ]
                )
                popt_fuzzy, pcov_fuzzy = curve_fit(fitfunc, gvals_x, gvals_y)

                # smooth axline recovery
                if is_stoc_signal:
                    smooth_gIN = [gval[0] for gval in fakeAx_grecovery_smooth]
                    smooth_gOUT = [gval[1] for gval in fakeAx_grecovery_smooth]
                    first_jump = find_first_differing_element2(
                        smooth_gOUT, smooth_gIN, diff_thresh
                    )
                    # cutoff_smooth = smooth_gIN[smooth_gOUT.index(first_jump)]
                    cutoff_smooth = 0.030
                    smooth_x = np.array(
                        [
                            gval[0]
                            for gval in fakeAx_grecovery_smooth
                            if gval[0] >= cutoff_smooth and gval[0] <= cutoff_hi
                        ]
                    )
                    smooth_y = np.array(
                        [
                            gval[1]
                            for gval in fakeAx_grecovery_smooth
                            if gval[0] >= cutoff_smooth and gval[0] <= cutoff_hi
                        ]
                    )
                    popt_smooth, pcov_smooth = curve_fit(fitfunc, smooth_x, smooth_y)

                # old method before NPE
                # old_recovery=[]
                # old_gIN=[]
                # old_gOUT=[]
                # old_axfile=rf"{savepath}/smooth_axion_recovery.txt"
                # with open(old_axfile, 'r') as file:
                #    for line in file:
                #        gin, gout = map(float, line.split())
                #        old_gIN.append(gin)
                #        old_gOUT.append(gout)
                #        old_recovery.append([gin,gout])
                # first_jump = find_first_differing_element2(old_gOUT,old_gIN,diff_thresh)
                # cutoff_old = old_gIN[old_gOUT.index(first_jump)]
                # old_x = np.array([gval[0] for gval in old_recovery if gval[0] >= cutoff_old and gval[0] <= cutoff_hi])
                # old_y = np.array([gval[1] for gval in old_recovery if gval[0] >= cutoff_old and gval[0] <= cutoff_hi])
                # popt_old, pcov_old = curve_fit(fitfunc, old_x, old_y)

                if plotopt == 0:

                    plt.close()
                    plt.rc("font", size=30)
                    plt.rcParams["font.family"] = "Times New Roman"
                    plt.rcParams["mathtext.fontset"] = "cm"
                    fig = plt.figure(figsize=(16, 6), dpi=400)

                    # plt.scatter(old_gIN, old_gOUT, color='blue', alpha=0.5, linewidth=1)
                    # plt.plot(old_x, fitfunc(old_x, *popt_old), color='blue', alpha=0.5, linewidth=1, label=r"Smooth ALP bias: ${:.3f}$".format(popt_old[0]))
                    # plt.axvline(x=cutoff_old, color='violet', linestyle='--', linewidth=4, alpha=0.7, label=r"Smooth cutoff: {:.3f} $\mathrm{{GeV}}^{{-1}}$".format(cutoff_old))

                    plt.scatter(
                        self.fakeAx_gIN,
                        self.fakeAx_gOUT,
                        color="green",
                        alpha=0.9,
                        linewidth=1,
                    )
                    plt.plot(
                        gvals_x,
                        fitfunc(gvals_x, *popt_fuzzy),
                        color="green",
                        alpha=0.9,
                        linewidth=1,
                        label=r"Analysis bias: ${:.3f}$".format(popt_fuzzy[0]),
                    )
                    plt.axvline(
                        x=cutoff_lo,
                        color="green",
                        linestyle="--",
                        linewidth=4,
                        alpha=0.5,
                        label=r"Detection cutoff: {:.3f} $\mathrm{{GeV}}^{{-1}}$".format(
                            cutoff_lo
                        ),
                    )

                    if is_stoc_signal:
                        plt.scatter(
                            smooth_gIN, smooth_gOUT, color="red", alpha=0.9, linewidth=1
                        )
                        plt.plot(
                            smooth_x,
                            fitfunc(smooth_x, *popt_smooth),
                            color="red",
                            alpha=0.9,
                            linewidth=1,
                            label=r"Analysis bias: ${:.3f}$".format(popt_smooth[0]),
                        )
                        plt.axvline(
                            x=cutoff_smooth,
                            color="pink",
                            linestyle="--",
                            linewidth=4,
                            alpha=0.5,
                            label=r"Detection cutoff: {:.3f} $\mathrm{{GeV}}^{{-1}}$".format(
                                cutoff_smooth
                            ),
                        )

                    plt.xlabel(r"coupling IN [$\mathrm{GeV}^{-1}$]", fontsize=14)
                    plt.ylabel(r"coupling OUT [$\mathrm{GeV}^{-1}$]", fontsize=14)
                    plt.tick_params(axis="both", which="major", labelsize=12)
                    plt.legend(fontsize=14)

                    plottype = "fakeAxionRecovery"

                elif plotopt == 1:  # <- use this!
                    # load and plot a raw psd with axion signal injected in matlab
                    axdata = pd.read_excel(
                        f"{self.basepath}/{subscan}/AxionFinder results/{psdplotpath}"
                    )
                    freqs = axdata.iloc[:, 0].tolist()
                    specs = axdata.iloc[:, 1].tolist()
                    smooth_axline = axdata.iloc[:, 2].tolist()
                    try:
                        random_axline = axdata.iloc[:, 3].tolist()
                    except (
                        IndexError
                    ):  # if non-stochastic axions have been injected, this column will be empty
                        random_axline = np.zeros(shape=np.array(smooth_axline).shape)

                    # make spectrum easier to handle
                    centerfreq = 1.3485  # MHz
                    ax_index = np.abs(np.array(freqs) - centerfreq).argmin()
                    buffer = int(1e4)
                    # freqs = freqs[ax_index-buffer:ax_index+buffer]
                    # specs = specs[ax_index-buffer:ax_index+buffer]
                    # smooth_axline = smooth_axline[ax_index-buffer:ax_index+buffer]
                    # random_axline = random_axline[ax_index-buffer:ax_index+buffer]

                    specxlabel = rf"Frequency - {centerfreq} [MHz]"
                    specyunit = "$\cdot 10^{%d}\ \\Phi_{0}^2/\\mathrm{Hz}$" % (-10)
                    specyunit = "$\Phi_{0}^2/\mathrm{Hz}$"

                    plt.close()
                    plt.rc("font", size=22)
                    plt.rcParams["font.family"] = "Times New Roman"
                    plt.rcParams["mathtext.fontset"] = "cm"
                    fig, axs = plt.subplots(2, 1, figsize=(7, 8), dpi=300)

                    axs[0].scatter(
                        np.array(freqs),
                        np.array(specs),
                        label="Flux PSD",
                        color="blue",
                        zorder=0,
                        alpha=0.8,
                        linewidth=0.2,
                    )
                    axs[0].plot(
                        np.array(freqs),
                        (np.mean(np.array(specs) - random_axline) + random_axline),
                        color="green",
                        alpha=0.9,
                        linewidth=1,
                        zorder=1,
                        label="Stochastic ALP",
                    )
                    axs[0].plot(
                        np.array(freqs),
                        (np.mean(np.array(specs) - smooth_axline) + smooth_axline),
                        color="red",
                        alpha=0.9,
                        linewidth=2,
                        zorder=2,
                        label="Averaged ALP",
                    )
                    axs[0].set_xlabel(specxlabel)  # , fontsize=20
                    axs[0].set_ylabel(rf"PSD [{specyunit}]")
                    axs[0].tick_params(axis="both", which="major")  # , labelsize=20
                    axs[0].set_xlim(98, 108)
                    axs[0].set_ylim(
                        1e-12, np.max(specs) * (1 + 1e-1)
                    )  # min 1e10*np.min(specs)
                    axs[0].set_yscale("log")
                    axs[0].legend(fontsize=20, loc="lower right")  #

                    # ax_old_label="Averaged axion, no NPE: ${:.4f}$".format(popt_old[0])
                    # ax_old_linelabel="" #r"Detection cutoff: {:.3f} $\mathrm{{GeV}}^{{-1}}$".format(cutoff_old)
                    # axs[1].scatter(old_gIN, old_gOUT, color='blue', alpha=0.8, linewidth=1)
                    # axs[1].plot(old_x, fitfunc(old_x, *popt_old), color='blue', alpha=0.5, linewidth=1, label=ax_old_label)
                    # axs[1].axvline(x=cutoff_old, color='blue', linestyle='--', linewidth=2, alpha=0.4, label=ax_old_linelabel)

                    if is_stoc_signal:
                        if not is_SG_scan:
                            ax_avg_label = "Averaged axion, SG off: ${:.4f}$".format(
                                popt_smooth[0]
                            )
                            ax_stoc_label = "Stochastic axion, SG off: ${:.4f}$".format(
                                popt_fuzzy[0]
                            )
                        else:
                            ax_avg_label = "Averaged axion: ${:.4f}$".format(
                                popt_smooth[0]
                            )
                            ax_stoc_label = "Stochastic axion: ${:.4f}$".format(
                                popt_fuzzy[0]
                            )
                            check(popt_fuzzy[0])
                        ax_avg_linelabel = ""  # r"Detection cutoff: {:.3f} $\mathrm{{GeV}}^{{-1}}$".format(cutoff_smooth)
                        # axs[1].scatter([gval[0] for gval in fakeAx_grecovery_smooth if gval[0] <= cutoff_hi], [gval[1] for gval in fakeAx_grecovery_smooth if gval[0] <= cutoff_hi], color='red', alpha=0.8, linewidth=1)
                        # axs[1].plot(smooth_x, fitfunc(smooth_x, *popt_smooth), color='red', alpha=0.5, linewidth=1, label=ax_avg_label)
                        # axs[1].axvline(x=cutoff_smooth, color='red', linestyle='--', linewidth=2, alpha=0.4, label=ax_avg_linelabel)
                    else:
                        if not is_SG_scan:
                            ax_stoc_label = "Averaged axion, SG off: ${:.4f}$".format(
                                popt_fuzzy[0]
                            )
                        else:
                            ax_stoc_label = "Averaged axion: ${:.4f}$".format(
                                popt_fuzzy[0]
                            )
                    ax_stoc_linelabel = ""  # r"Detection cutoff: {:.3f} $\mathrm{{GeV}}^{{-1}}$".format(cutoff_lo)
                    axs[1].scatter(
                        [
                            gval[0]
                            for gval in self.fakeAx_grecovery
                            if gval[0] <= cutoff_hi
                        ],
                        [
                            gval[1]
                            for gval in self.fakeAx_grecovery
                            if gval[0] <= cutoff_hi
                        ],
                        color="green",
                        alpha=0.8,
                        linewidth=1,
                    )
                    axs[1].plot(
                        gvals_x,
                        fitfunc(gvals_x, *popt_fuzzy),
                        color="green",
                        alpha=0.9,
                        linewidth=1,
                    )  # label=ax_stoc_label
                    axs[1].axvline(
                        x=cutoff_lo,
                        color="green",
                        linestyle="--",
                        linewidth=2,
                        alpha=0.4,
                    )  # label=ax_stoc_linelabel

                    axs[1].set_xlabel(r"Injected coupling [$\mathrm{GeV}^{-1}$]")
                    axs[1].set_ylabel(r"Recovered coupling [$\mathrm{GeV}^{-1}$]")
                    axs[1].tick_params(axis="both", which="major")
                    # axs[1].legend()

                    axs[0].text(
                        -0.1, 1.05, "(a)", transform=axs[0].transAxes, va="top"
                    )  # fontweight='bold'
                    axs[1].text(-0.1, 1.05, "(b)", transform=axs[1].transAxes, va="top")

                    plottype = "FakeAxionInjection"

                elif plotopt == 2:
                    fig, axs = plt.subplots(2, 2, figsize=(16 * 0.9, 9 * 0.9), dpi=300)

                    # (top-left)
                    axs[0, 0].scatter(
                        gvals_x, gvals_y, color="green", alpha=0.9, linewidth=1
                    )
                    axs[0, 0].plot(
                        gvals_x,
                        fitfunc(gvals_x, *popt_fuzzy),
                        color="green",
                        alpha=0.9,
                        linewidth=1,
                        label=r"Analysis bias: ${:.3f}$".format(popt_fuzzy[0]),
                    )
                    # axs[0, 0].axvline(x=cutoff_lo, color='red', linestyle='--', linewidth=4, alpha=0.7, label=r"Detection cutoff: {:.3f} $\mathrm{{GeV}}^{{-1}}$".format(cutoff_lo))
                    axs[0, 0].set_xlabel(
                        r"coupling IN [$\mathrm{GeV}^{-1}$]", fontsize=14
                    )
                    axs[0, 0].set_ylabel(
                        r"coupling OUT [$\mathrm{GeV}^{-1}$]", fontsize=14
                    )
                    axs[0, 0].tick_params(axis="both", which="major", labelsize=12)
                    axs[0, 0].legend(fontsize=14)

                    # (top-right)
                    axs[0, 1].scatter(
                        gvals_x, fvals_y * 1e-6, color="orange", alpha=0.9, linewidth=1
                    )
                    axs[0, 1].set_xlabel(
                        r"coupling IN [$\mathrm{GeV}^{-1}$]", fontsize=14
                    )
                    axs[0, 1].set_ylabel(r"frequency OUT [MHz]", fontsize=14)
                    axs[0, 1].tick_params(axis="both", which="major", labelsize=12)

                    # (bottom-left)
                    axs[1, 0].scatter(
                        fvals_x * 1e-6, gvals_y, color="pink", alpha=0.9, linewidth=1
                    )
                    axs[1, 0].set_xlabel(r"frequency IN [MHz]", fontsize=14)
                    axs[1, 0].set_ylabel(
                        r"coupling OUT [$\mathrm{GeV}^{-1}$]", fontsize=14
                    )
                    axs[1, 0].tick_params(axis="both", which="major", labelsize=12)

                    # (bottom-right)
                    popt_f, pcov_f = curve_fit(
                        fitfunc,
                        fvals_x * 1e-6,
                        fvals_y * 1e-6,
                    )
                    axs[1, 1].scatter(
                        fvals_x * 1e-6,
                        fvals_y * 1e-6,
                        color="blue",
                        alpha=0.9,
                        linewidth=1,
                    )
                    axs[1, 1].plot(
                        fvals_x * 1e-6,
                        fitfunc(fvals_x * 1e-6, *popt_f),
                        color="green",
                        alpha=0.9,
                        linewidth=1,
                        label=r"Frequency accuracy: ${:.3f}$".format(popt_f[0]),
                    )
                    axs[1, 1].set_xlabel(r"frequency IN [MHz]", fontsize=14)
                    axs[1, 1].set_ylabel(r"frequency OUT [MHz]", fontsize=14)
                    axs[1, 1].tick_params(axis="both", which="major", labelsize=12)
                    axs[1, 1].legend(fontsize=14)

                    plottype = "FakeAxionRecovery2x2"

                # Adjust layout to prevent overlap
                # fig.suptitle(rf'Fake ALP signals with ${min([gin for gin in self.fakeAx_gIN])} < g_{{aNN}} < {max([gin for gin in self.fakeAx_gIN])} (1/GeV)$, random frequencies', fontsize=16)
                # plt.autoscale()
                plt.tight_layout()
                # plt.grid()
                plt.savefig(
                    rf"{savepath}/{subscan}_{mat_analysisname}_{plottype}_{cands_str}.png",
                    bbox_inches="tight",
                    dpi=400,
                )
                if showplot:
                    plt.show()
                plt.clf()
                plt.close()


class dmScan_2024:
    def __init__(self, name="", verbose=False):
        """
        Initialize DMscan class for a dark-matter measurement scan by CASPEr-gradient-lowfield
        This design of this object should and is capable of dealing with large data file,
        which is important to the earth- / solar- dark-matter halo data analysis.

        """
        self.info = None
        return

    def LoadStepInfo(
        self,
        load_dmScanStep: bool = False,
        load_MeasTime: bool = False,
        load_FreqRange: bool = False,
        verbose: bool = True,
    ):
        """
        structure of the scan information dictionary
        level-1
        ['stepinfo']: list
        ['dmScanStep']: list
        ['MeasTime']: list, optional
        ['FreqRange']: list, optional
        ['Notes']: string, optional

        level-2
        ['dmScanStep'][]: dict
            dictionary for dmScanStep information
        ['MeasTime'][]: dict
            ['period']: list, optional
                start and end of the scannings in str
            ['duration']: float, optional
                duration of the scannings in [s]
        ['FreqRange'][]: list, optional
            frequency range of the scannings in [Hz]

        """

        return

    def tsCheckJump(
        self,
    ):
        return

    def tsCheckDrift(
        self,
    ):
        return

    def tsCheckSanity(self, plotIfInsane: bool = False, verbose=False):
        total = len(self.step_list)
        for i, step in enumerate(self.step_list):
            print_progress_bar(
                i, total, length=50, prefix="Checking sanity of scan steps"
            )
            step.nps.LoadStream()
            report = step.nps.tsCheckSanity(plotIfInsane=plotIfInsane, verbose=verbose)
            del step.nps.dataX, step.nps.dataY
            # if i >= 0:
            #     clear_lines()
            #     print(report)

        print_progress_bar(
            total, total, length=50, prefix="Checking sanity of scan steps"
        )
        sys.stdout.write(
            "\n"
        )  # Move to the next line after the progress bar is complete
        return

    def SumScanInfor(
        self,
    ):
        """
        Summarize the information of the scan.
        """
        return 0

    def SumAnalysisWindows(
        self,
    ):
        """
        Summarize the analysis windows of all scan steps.
        """
        return 0

    def VerticalComb(
        self,
    ):
        """
        Vertical combination of the scan.
        """
        return 0

    def HorizontalComb(
        self,
    ):
        """
        Horizontal combination of the whole spectrum.
        """
        return 0

    def FindCandidate(
        self,
        howtothresh=6,
        resavemat_opt=False,
        normalizebysigma=False,
        plot_opt=False,
        showplot_opt=False,
        savepath="",
    ):
        return


class Station:
    def __init__(
        self,
        name="Station name",
        NSsemisphere=None,  # 'N' or 'S'
        EWsemisphere=None,  # 'E' or 'W'
        unit="deg",
        latitude_deg=None,  # in [deg]
        longitude_deg=None,  # in [deg]
        elevation=None,  # in [m]
        verbose=False,
    ):
        """
        initialize a station on Earth
        """
        if name is None:
            raise ValueError("name is None")

        self.name = name
        self.NSsemisphere = NSsemisphere
        self.EWsemisphere = EWsemisphere
        # if latitude is None:
        #     raise ValueError('latitude is None')
        self.latitude_deg = latitude_deg
        self.longitude_deg = longitude_deg
        # if NSsemisphere == 'N':
        #     self.theta = np.pi/2 - self.latitude
        # elif NSsemisphere == 'S':
        #     self.theta = np.pi/2 + self.latitude
        # else:
        #     raise ValueError('NSsemisphere != \'N\' nor \'S\'')

        # if EWsemisphere == 'E':
        #     self.phi = self.longitude
        # elif EWsemisphere == 'W':
        #     self.phi = (-1.)* self.longitude
        # else:
        #     raise ValueError('NSsemisphere != \'N\' nor \'S\'')

        # self.nvec = np.array([np.sin(self.theta)*np.cos(self.phi), np.sin(self.theta)*np.sin(self.phi), np.cos(self.theta)])
        self.elevation = elevation
        self.R = self.elevation + 6356.7523e3  # radius
        # self.rvec = self.R * self.nvec


Mainz = Station(
    name="Mainz",
    NSsemisphere="N",  # 'N' or 'S'
    EWsemisphere="E",  # 'E' or 'W'
    unit="deg",
    latitude_deg=49.9916,  # in [deg]
    longitude_deg=(8.0 + 16.0 / 60.0 + 26.2056 / 3600),  # in [deg]
    elevation=130.0,
    verbose=False,
)


class AxionWind:
    # Create the "axion wind" (axion field) object
    # you can get propeties of the axion field, computed based on the input information
    def __init__(
        self,
        name="axion",
        nu_a: float = 1e6,  # compton frequency in [Hz]
        # Gamma=1.*10.**(-6),  # spectrum linewidth. refer to -> https://journals.aps.org/prd/pdf/10.1103/PhysRevD.105.035029
        gaNN=None,  # in [eV^-1]
        Qa=None,
        year=None,
        month=None,
        day=None,
        time_hms=None,  # Use UTC time!
        # example
        # year=2024, month=9, day=10, time='14:35:16.235812',
        timeastro=None,
        station: Station = None,
        v_0=220.0
        * 10
        ** 3,  # [m/s] # Local (@ solar radius) galaxy circular rotation speed (in m/s)
        v_lab=233.0
        * 10
        ** 3,  # [m/s] # Laboratory speed relative to the galactic rest frame (in m/s)
        windangle=None,
        # dark matter axion density
        # PDG 2023: 0.3
        # Derek's book: 0.4, also used by Dani
        # SHM+: 0.55
        rho_DM=(0.55 * 1e9)
        * (
            1e6
        ),  # eV/m^3 !!!! unit changed from cm^3 to m^3  #### 3.6e-42 GeV^4 = 3.6e-6 eV^4 # what CASPEr-e uses
        verbose=False,
    ):
        """
        initialize NMR simulation
        """
        self.name = name
        self.c = 299792458.0  # [m/s]. speed of light in vacuum
        self.v_0 = v_0
        self.v_lab = v_lab
        self.windangle = windangle
        self.rho_DM = rho_DM
        self.nu_a = nu_a
        self.gaNN = gaNN

        self.hbar_Joule = 1.05457182e-34  # [J/Hz]
        self.hbar_eV = 6.582119569e-16  # [eV/Hz]
        self.h_eV = 4.135667696e-15  # [eV/Hz]V^4

        self.year = year
        self.month = month
        self.day = day
        self.time_hms = time_hms
        self.timeastro = timeastro
        self.station = station
        # self.lat = station.latitude_deg
        # self.lon = station.longitude_deg
        # self.elev = station.elevation
        if Qa is None:
            self.Qa = (self.c / self.v_0) ** 2.0
        self.lw_Hz = 1.0 * nu_a / self.Qa
        self.lw_p = None

    def GetAxionWind(
        self,
        year=None,
        month=None,
        day=None,
        time_hms=None,
        latitude=None,
        longitude=None,
        elevation=None,
        verbose=False,
    ):
        """
        Parameters
        ----------
        time_hms: needs to be in the format "15:47:18"
            if none is specified, use current time

        lat: latitude of experiment location
            if none is specified, use Mainz: 49.9916 deg north

        lon: longitude of experiment location
            if none is specified, use Mainz: 8.2353 deg east

        elev: height of experiment location
            if none is specified, use Uni Campus Mainz: 130 m

        Returns
        ---------
            1. the velocity 'v_lab' and 'v_ALP_perp' between lab frame and
        DM halo (SHM), in the galactic rest frame, for the specified
        coordinates and time
            2. angle [rad] between the CASPEr sensitive axis (z-direction =
        earth surface normal)
            3. v_ALP, v_ALP_perp, alpha_ALP go into self.

        """
        if verbose:
            print("now calculating wind angle")

        year = year or self.year
        month = month or self.month
        day = day or self.day
        time_hms = time_hms or self.time_hms

        if self.timeastro is None:
            if (year or month or day or time_hms) is None:
                time_DMmeasure = Time.now()  # UTC time
                # example of the astropy.time.Time.now() return value
                # 2024-09-11 14:27:44.732284
                print(
                    f"no date and time input provided, using current date and time: {time_DMmeasure}"
                )
            else:
                time_DMmeasure = rf"{year}-{month}-{day}T{time_hms}"
            if verbose:
                print(f"time input: {time_DMmeasure}")
            self.timeastro = Time(time_DMmeasure, format="isot", scale="utc")
            # example of timeastro
            # 2024-09-11T14:35:16.236

        if self.station is None:
            self.station = Mainz
            if verbose:
                print("no station specified, defaulting to CASPEr Mainz")

        lat = latitude or self.station.latitude_deg
        lon = longitude or self.station.longitude_deg
        elev = elevation or self.station.elevation

        DMtimefrac = wind.FracDay(Y=2022, M=12, D=23)
        if verbose:
            print("time of DM measurement (fractional days): ", DMtimefrac)

        LABvel = wind.ACalcV(DMtimefrac)
        if verbose:
            print("velocity (lab frame) @DM time: ", LABvel)

        DMtime, unit_North, unit_East, unit_Up, Vhalo = wind.get_CASPEr_vect(
            time=self.timeastro,
            lat=lat,
            lon=lon,
            elev=elev,
        )

        # print(type(Vhalo))
        Vlab = Vhalo.get_d_xyz()  # convert into a vector
        Bz = (
            unit_Up.get_xyz()
        )  # our leading field is pointing up perpendicular to earth's surface

        alpha_ALP = angle_between(Vlab, Bz).value
        v_ALP = np.linalg.norm(Vlab.value) * 1e3
        v_ALP_perp = v_ALP * math.sin(alpha_ALP)

        if verbose:
            # print("time of DM measurement: ", DMtime)
            print("Bz vector @DM time (galaxy frame):", Bz)
            print("v_halo @DM time (galaxy frame):", Vhalo)
            print("v_lab @DM time:", Vlab)
            print("angle between sensitive axis & lab velocity @DM time: ", alpha_ALP)

        ###############################################################################################
        # do not delete
        self.windangle = alpha_ALP
        self.v_lab = v_ALP
        self.v_ALP_perp = v_ALP_perp
        self.alpha_ALP = alpha_ALP
        return v_ALP, v_ALP_perp, alpha_ALP

    # check Gramolin paper for functions:
    # axion_beta, axion_lambda, axion_C_para, axion_C_perp
    def axion_beta(self, nu_a, nu):
        if nu <= nu_a:
            return 0.0
        else:
            return (
                (2 * self.c * self.v_lab)
                / self.v_0**2
                * np.sqrt(2 * (nu - nu_a) / nu_a)
            )

    def axion_lambda(
        self,
        nu_a,
        nu,
        alpha,
    ):
        p0 = (2 * self.c**2) / (np.sqrt(np.pi) * self.v_0 * self.v_lab * nu_a)
        p1 = np.exp(
            -self.axion_beta(nu_a=nu_a, nu=nu) ** 2 * self.v_0**2 / (4 * self.v_lab**2)
            - (self.v_lab / self.v_0) ** 2
        )
        p2 = np.sinh(self.axion_beta(nu_a=nu_a, nu=nu))
        return p0 * p1 * p2

    def axion_C_para(
        self,
        alpha,
    ):
        return self.v_0**2 / 2.0 + self.v_lab**2 * np.cos(alpha) ** 2

    def axion_C_perp(
        self,
        alpha,
    ):
        return self.v_0**2 + self.v_lab**2 * np.sin(alpha) ** 2

    def lineshape_t(self, nu, nu_a=None, case="grad_perp") -> np.ndarray:
        """
        axion_lineshape at time t
        """
        # v_ALP, v_ALP_perp, alpha_ALP = get_ALP_wind(\
        if nu_a is None:
            nu_a = self.nu_a
        if self.station is None:
            self.station = Mainz
            # print('no station specified, defaulting to CASPEr Mainz')
        self.GetAxionWind(
            year=self.year,
            month=self.month,
            day=self.day,
            time_hms=self.time_hms,
            latitude=self.station.latitude_deg,
            longitude=self.station.longitude_deg,
            elevation=self.station.elevation,
            verbose=False,
        )

        return axion_lineshape(
            v_0=self.v_0,
            v_lab=self.v_lab,
            nu_a=nu_a,
            nu=nu,
            case=case,
            alpha=self.windangle,
        )  # type: ignore

    # def GetALPamplitude(self, nu_ALP, g_aNN, DMfiles, filepath, index, usedailymod=True):
    def GetALP_Data(
        self,
        T_acq,
        T2time,
        NMRfreq,
        NMRwidth,
        NMRamp,
        nu_ALP,
        g_aNN,
        year=None,
        month=None,
        day=None,
        time_str=None,
        datetime_int=None,
        usedailymod=True,
        returnerrors=False,
    ):

        if year is None:
            year = self.year
        if month is None:
            month = self.month
        if time_str is None:
            time_str = self.time_hms
        if datetime_int is None:
            datetime_int = GetDateTimeSimple(
                year, month, day, time_str, return_as_int=True
            )

        if (not T_acq) or (T_acq == 0):
            T_acq = 0.5  # [s] acquisition time
        T_acq_err = T_acq * 0.01  # 1% error, idk

        if len(T2time) == 2:
            T2 = T2time[0]
            T2_err = T2time[1]
        else:
            T2 = T2time
            T2_err = 0

        if len(NMRfreq) == 2:
            nu_n = NMRfreq[0]
            nu_n_err = NMRfreq[1]
        else:
            nu_n = NMRfreq
            nu_n_err = 0

        if len(NMRwidth) == 2:
            FWHM_n = NMRwidth[0]
            FWHM_n_err = NMRwidth[1]
        else:
            FWHM_n = NMRwidth
            FWHM_n_err = 0

        if len(NMRamp) == 2:
            amp_n = NMRamp[0]
            amp_n_err = NMRamp[1]
        else:
            amp_n = NMRamp
            amp_n_err = 0

        if self.v_lab is None or self.windangle is None:
            # print('calc axion wind')
            self.GetAxionWind(year=year, month=month, day=day, time_hms=time_str)

        if nu_ALP == 0:
            # print("using fit larmor freq")
            nu_ALP = nu_n

        Q_ALP = (self.c / self.v_lab) ** 2  # roughly 1e6
        # tau_ALP = Q_ALP / nu_ALP # may be mising a factor pi?
        # check(tau_ALP)

        # FWHM_ALP = nu_ALP * (self.v_lab / self.c)**2
        FWHM_ALP = (
            nu_ALP
            * (self.v_0**2 + self.v_lab**2 * np.sin(self.windangle) ** 2)
            / self.c**2
        )  # this is the same as nu_a*b2 in MATLAB

        tau_ALP = 1 / (np.pi * FWHM_ALP)  # * (Q_ALP / nu_ALP) # <- ???
        T2star = 1 / (np.pi * FWHM_n)

        # check(Q_ALP)
        # check(tau_ALP)
        # check(T2star)
        # check(T2)
        # check(FWHM_n)
        # check(FWHM_ALP)

        # spinfraction = FWHM_ALP / FWHM_n
        delta2 = 1 / (np.pi * T2) + 1 / (np.pi * tau_ALP)
        delta3 = 1 / (np.pi * T2star) + 1 / (np.pi * tau_ALP)
        spinfraction = delta2 / delta3

        V90 = np.sqrt(
            4 * amp_n * T_acq / T2star
        )  # SQUID readout voltage after a pi/2 pulse
        # phi90 *= (SQUID.Mf / SQUID.Rf) # decay signal correction # Is SQUID intialized? I comment it out because the script cannot run with it. -- Yuzhe
        SQUID_Mf = 22665.457842248416
        SQUID_Rf = 3000.0
        phi90 = V90 * (SQUID_Mf / SQUID_Rf)  # magn. flux in SQUID after pi/2 pulse

        M0 = 1.859e-10  # (T), it is M0*mu0
        tpuls_90 = 189.82530271640852e-6  # (s) from pulse sweep fit
        tpuls_is = 190e-6  # the pulse length we really used
        # 90° tipping: rabiNMR * tpuls_90 = pi/2, sin(rabiNMR * tpuls_90) = 1
        # real tipping: sin(rabiNMR * tp_is) = sin(pi/2 * (tp_is / tpuls_90))
        # SQUIDpickup = phi90 / (M0 * np.sin(np.pi/2 * (tpuls_is / tpuls_90)))
        SQUIDpickup = phi90 / M0

        if usedailymod:
            v_ALP_perp = self.v_lab * np.sin(self.windangle)
        else:  # Younggeun's remark. Use stochastic average of random angle cos²(theta) between ALP gradient and B0 instead of v_ALP_perp
            v_ALP_perp = self.v_lab * np.sqrt(1 / 3)
        # rabi = 1/2 * g_aNN * m_a *a0 *c/hbar * v_perp
        rabi = (
            1
            / 2
            * g_aNN
            * np.sqrt(2 * self.hbar_eV * self.c * self.rho_DM)
            * v_ALP_perp
        )

        rms_tipping = rabi * T2 * np.sqrt(tau_ALP / (tau_ALP + T2))
        # if tau_ALP < T2:
        #    rms_tipping = rabi * np.sqrt(tau_ALP * T2)
        # elif T2 < tau_ALP:
        #    rms_tipping = rabi * T2

        # axionamp corresponds to P_perp in Gramolin's paper
        # the PSD amplitude |S|² corresponds to axionamp * lambda_perp
        axionamp = 1 / 2 * (SQUIDpickup * M0 * spinfraction * rms_tipping) ** 2
        # check(axionamp)

        # uncertainties
        ##############################################################
        errors = []
        if returnerrors:
            # first-order uncertainties from measurements: T2_err, nu_n_err, FWHM_n_err, amp_n_err
            # derved uncertainties: T2star_err<-FWHM_n_err, tau_err<-nu_n_err, phi90err<-(T2star_err,amp_n_err), etc...

            T2star_err = FWHM_n_err / (np.pi * FWHM_n**2)

            # tau_err = nu_n_err * (Q_ALP / nu_n**2)
            tau_err = np.abs(
                -self.c**2
                / (
                    np.pi
                    * nu_n**2
                    * (self.v_0**2 + self.v_lab**2 * np.sin(self.windangle) ** 2)
                )
            )

            tpuls_err = (tpuls_is - tpuls_90) * (
                1 + 0.05
            )  # add some 5% for fit uncertainty estimation

            # phi90 error
            # dphi90_damp_n = T_acq / (phi90 * amp_n)
            # dphi90_dT_acq = amp_n / (phi90 * T_acq)
            # dphi90_dT2star = -phi90 / (2 * T2star)
            dphi90_damp_n = np.sqrt(T_acq / amp_n / T2star)
            dphi90_dT_acq = np.sqrt(amp_n / T_acq / T2star)
            dphi90_dT2star = -np.sqrt(amp_n * T_acq / T2star**3)

            phi90_err = np.sqrt(
                (dphi90_damp_n * amp_n_err) ** 2
                + (dphi90_dT_acq * T_acq_err) ** 2
                + (dphi90_dT2star * T2star_err) ** 2
            )

            # SQUIDpickup error
            dSQUIDpickup_dtpuls = (
                np.pi
                * phi90
                * 1
                / math.tan(np.pi * tpuls_is / 2 / tpuls_90)
                * 1
                / math.sin(np.pi * tpuls_is / 2 / tpuls_90)
            ) / (2 * M0 * tpuls_90)
            dSQUIDpickup_dphi90 = 1 / math.sin(np.pi * tpuls_is / 2 / tpuls_90) / M0
            SQUIDpickup_err = np.sqrt(
                (dSQUIDpickup_dtpuls * tpuls_err) ** 2
                + (dSQUIDpickup_dphi90 * phi90_err) ** 2
            )

            # spinfraction error
            a = 1 / (np.pi * T2)
            b = 1 / (np.pi * tau_ALP)
            c = 1 / (np.pi * T2star)
            spinfraction = (a + b) / (c + b)

            ds_dT2 = -1 / (np.pi * T2**2) * 1 / (c + b)
            ds_dtau_ALP = (
                -1 / (np.pi * tau_ALP**2) * (1 / (c + b) - (a + b) / (c + b) ** 2)
            )
            ds_dT2star = -1 / (np.pi * T2star**2) * (a + b) / (c + b) ** 2

            spinfraction_err = np.sqrt(
                (ds_dT2 * T2_err) ** 2
                + (ds_dT2star * T2star_err) ** 2
                + (ds_dtau_ALP * tau_err) ** 2
            )

            # axionamp error
            daxionamp_dpickup = (
                (M0 * rabi * spinfraction * T2) ** 2
                * tau_ALP
                * SQUIDpickup
                / (tau_ALP + T2)
            )
            daxionamp_dspinfraction = (
                (M0 * rabi * SQUIDpickup * T2) ** 2
                * tau_ALP
                * spinfraction
                / (tau_ALP + T2)
            )
            # daxionamp_drabi = (M0*spinfraction*SQUIDpickup*T2)**2*tau_ALP*rabi / (tau_ALP + T2)
            daxionamp_dT2 = (
                (M0 * rabi * spinfraction * SQUIDpickup) ** 2
                * tau_ALP
                * T2
                * (2 * tau_ALP + T2)
                / (2 * (tau_ALP + T2) ** 2)
            )
            daxionamp_dtau_ALP = (
                (M0 * rabi * SQUIDpickup * spinfraction) ** 2
                * T2**3
                / (2 * (tau_ALP + T2) ** 2)
            )

            axionamp_err = np.sqrt(
                (daxionamp_dpickup * SQUIDpickup_err) ** 2
                + (daxionamp_dspinfraction * spinfraction_err) ** 2
                +
                # (daxionamp_drabi * rabi_err)**2 +
                (daxionamp_dT2 * T2_err) ** 2
                + (daxionamp_dtau_ALP * tau_err) ** 2
            )

            # check(nu_n)
            # check(nu_n_err)
            # check(FWHM_n)
            # check(FWHM_n_err)
            # check(amp_n)
            # check(amp_n_err)
            # check(T2)
            # check(T2_err)
            # check(T2star)
            # check(T2star_err)
            # check(tau_ALP)
            # check(tau_err)
            # check(rabi)
            # check(rabi_err)
            # check(phi90)
            # check(phi90_err)
            # check(spinfraction)
            # check(spinfraction_err)
            # check(axionamp)
            # check(axionamp_err)
            # print('Yuzhe: here we can choose a better way to return the values -- add variables in the class. Let us discuss on it. ')
            errors = [
                tau_err,
                T2_err,
                SQUIDpickup_err,
                nu_n_err,
                FWHM_n_err,
                amp_n_err,
                axionamp_err,
            ]

        return (
            datetime_int,
            self.v_lab,
            self.windangle,
            tau_ALP,
            T2,
            nu_n,
            FWHM_n,
            amp_n,
            phi90,
            axionamp,
            errors,
        )

    def GetFFTsignal(
        self,
        freq,
        case="grad_perp",
        timeastro=None,
        rand_amp: bool = True,
        rand_phase: bool = True,
        verbose: bool = False,
    ):
        if timeastro is not None:
            self.timeastro = timeastro
        axion_lineshape = self.lineshape_t(nu=freq, case=case)
        # check(np.sum(axion_lineshape) * np.abs(freq[0]-freq[1]))
        # check(np.sum(axion_lineshape * axion_lineshape))

        if rand_amp:
            rvs_amp = expon.rvs(loc=0.0, scale=1.0, size=len(freq))
            # check(np.mean(rvs_amp))
        else:
            rvs_amp = 1.0

        if rand_phase:
            rvs_phase = np.exp(1j * uniform.rvs(loc=0, scale=2 * np.pi, size=len(freq)))
        else:
            rvs_phase = 1.0

        ax_FFT = np.sqrt(axion_lineshape * rvs_amp) * rvs_phase
        if verbose:
            # check(np.sum(np.abs(ax_FFT)**2.) * abs(freq[0]-freq[1]))
            check(np.sum(np.abs(ax_FFT) ** 2.0))
            check(np.sum(axion_lineshape))
        return ax_FFT

    def MatchFilter(self, xstamp, signal, conv_step, conv_line):
        assert len(signal) > len(conv_line)
        conv_signal = []
        conv_xstamp = []
        conv_line_len = len(conv_line)

    #     conv_step_len = 1.0 * conv_step / abs(xstamp[1]-xstamp[0])
    #     if conv_step_len < 1.0:
    #         check(conv_step_len)
    #         raise ValueError('conv_step_len too short. Increase conv_step.')
    #     conv_step_len = int(conv_step_len)

    #     conv_step_num = int(1.0 * abs(xstamp[-1]-xstamp[0]) / conv_step)
    #     for i in range(conv_step_num):
    #         if i * conv_step_len + conv_line_len > len(signal)-1:
    #             break
    #         conv_xstamp.append([i * conv_step + xstamp[0]])
    #         p = signal[i * conv_step_len:i * conv_step_len + conv_line_len] * conv_line
    #         conv_signal.append(np.sum(p)/np.sum(conv_line)**2)
    #     return conv_xstamp, conv_signal


def amp_ga_1iter(
    g_a,
    ifstochastic=False,
    use_sg=False,
    # use_sg = True
    sg_axlw_frac=10,
    sg_order=2,
    dmodfreq=1e6,
    samprate=230.0,
    total_dur=10.0,
    year=2022,
    month=12,
    day=23,
    time_hms="00:00:00.0",  # Use UTC time!
    nu_a=1e6 + 1,
    Qa=None,
):
    std = 1.0 / samprate
    step = dmScanStep(
        name="test data",
        NoPulse_dataFile_list=[],
        pNMR_dataFile_list=None,
        CPMG_dataFile_list=None,
        station=Mainz,
        verbose=False,
    )
    # '20221223_154712', '20221223_175006', '20221223_194651'
    step.nps.CreateArtificialStream(
        dmodfreq=dmodfreq,
        samprate=samprate,
        total_dur=total_dur,
        year=year,
        month=month,
        day=day,
        time_hms=time_hms,  # Use UTC time!
    )
    step.nps.GetNoPulseFFT()
    #
    step.InsertFakeAxion(
        nu_a=nu_a,
        g_a=g_a,
        Qa=Qa,
        rand_amp=ifstochastic,
        rand_phase=ifstochastic,
        # verbose=True
    )
    step.nps.PSD = np.abs(step.nps.FFT) ** 2.0
    # step.nps.GetSpectrum(showtimedomain=False, showfreqdomain=True, spectype='PSD')
    if use_sg:
        step.nps.PSD = step.nps.sgFilterPSD(
            window_length=step.axion.lw_p // sg_axlw_frac,
            polyorder=sg_order,
            makeplot=False,
        )
    # step.nps.GetSpectrum(showtimedomain=False, showfreqdomain=True, spectype='PSD')
    step.MovAvgByStep(step_len=max(1, step.axion.lw_p // 10), verbose=False)
    # step.nps.GetSpectrum(showtimedomain=False, showfreqdomain=True, spectype='PSD')
    r1 = step.nps.Hz2Index(step.axion.nu_a - 5.0 * step.axion.lw_Hz)
    # check(r1)
    noisePSDmean = np.mean(step.nps.PSD[0:r1])
    noisePSDstd = np.std(step.nps.PSD[0:r1])
    # check(noisePSDmean)
    # check(noisePSDstd)
    step.nps.PSD -= noisePSDmean
    # step.nps.GetSpectrum(showtimedomain=False, showfreqdomain=True, spectype='PSD')

    nu_a_index = step.nps.Hz2Index(step.axion.nu_a)
    # check(step.nps.Index2Hz(nu_a_index))

    ax_amp = step.nps.PSD[nu_a_index] / std
    # check(nu_a_index)
    # check(step.nps.avgPSD[nu_a_index])
    # check(np.amax(step.nps.avgPSD))
    del step
    return ax_amp


# TODO integrate the following code in the axionwind class
def amp_ga_10iter(g_a):
    # amp_ga_1iter_partial =
    g_a_10 = g_a * np.ones(10)
    with Pool() as pool:
        axion_signal_amps_for1g_a = pool.map(amp_ga_1iter, g_a_10)
    return axion_signal_amps_for1g_a


# def amp_ga(
#     # numofIter = 100
#     storedata = False,

#     ifstochastic=False,

#     use_sg = False,
#     # use_sg = True
#     sg_axlw_frac = 10,
#     sg_order = 2,

#     dmodfreq=1e6, samprate=230., total_dur=10.,
#     year=2022, month=12, day=23, time_hms='00:00:00.0',  # Use UTC time!

#     nu_a=1e6+1,
#     Qa=None,
#     g_a_arr = np.linspace(start=.001, stop=.819, num=50),


# # ):
# #     def quadratic(x, a):
# #         return a * x ** 2


#     axion_signal_amps = []
#     list_of_steps = []

#     tic = time.time()

#     # Loop style
#     # for i, g_a in enumerate(g_a_arr):
#     #     axion_signal_amps_for1g_a = []
#     #     for j in range(10):
#     #         axion_signal_amps_for1g_a.append(amp_ga_1iter(g_a))
#     #     axion_signal_amps.append(axion_signal_amps_for1g_a)

#     # Multi-task style
#     with Pool() as pool:
#         axion_signal_amps = pool.map(amp_ga_10iter, g_a_arr)
#     toc = time.time()
#     print(f'time consumption: {toc-tic:.1f} [s]')
#     # print_progress_bar(i, total=len(g_a_arr), prefix='Progress', suffix='Complete', length=50)
#     check(axion_signal_amps)

#     #
#     fig = plt.figure(figsize=(5, 4), dpi=150)  #
#     gs = gridspec.GridSpec(nrows=1, ncols=1)  #
#     # fig = plt.figure(figsize=(10, 4), dpi=150)  #
#     # gs = gridspec.GridSpec(nrows=1, ncols=2)  #
#     ax00 = fig.add_subplot(gs[0,0])
#     ax00.errorbar(x=g_a_arr, y=np.mean(axion_signal_amps, axis=1), yerr=np.std(axion_signal_amps, axis=1),\
#                 label='MC simulation',\
#                 fmt='o', markerfacecolor='none', markeredgecolor='black', \
#                     ecolor='black', capsize=5, capthick=2)
#     # check(np.mean(axion_signal_amps, axis=1))


#     popt, pcov = curve_fit(quadratic, g_a_arr, np.mean(axion_signal_amps, axis=1), [155.])
#     check(popt)

#     ax00.plot(g_a_arr, quadratic(g_a_arr, popt[0]), label='fit curve')
#     print(f'5 sigma coupling strength is {np.sqrt(5. / popt[0])}. ')

#     ax00.set_xlabel('input g_a')
#     ax00.set_ylabel('axion signal [$\\sigma$]')
#     ax00.set_title('')

#     title = f'Stochastic (spiky)? {str(ifstochastic)}\n'
#     title += f'SG filter? {str(use_sg)}'
#     title += f'. SG window length = axion linewidth // {sg_axlw_frac}' if use_sg else ''

#     fname = f'Stochastic (spiky) {str(ifstochastic)}_'
#     fname += f'SG filter {str(use_sg)}'
#     fname += f'_SG window length = axion linewidth by {sg_axlw_frac}' if use_sg else ''
#     print(fname)

#     fig.suptitle(title)
#     fig.tight_layout()

#     plt.show()
#     return


class SensiCompar:
    def __init__(
        self,
        name,
        nu_start=None,
        nu_end=None,
        fraction=None,
        frequencies=None,
        discovery_threshold=5,
        detuning_correction=1.0,
        gr=1.0,
        gDM=120.0,  # GeV.Hz
        Q=10**6,
        sigma=1e-12,  # reference noise level [Phi0^2/Hz]
        Phipito2=1e-8,  # reference SQUID flux [Phi0/sqrt(Hz)]
        T=1e-20,  # initial measurement time approx 0
    ):
        """
        Initialize the (class) LIAsignal object

        Parameters
        ----------
        name: str
            name of the Exclusion.

        frequencies : np.array
            measurement range

        """
        self.name = name

        if nu_start is not None and nu_end is not None and fraction is not None:
            Nnu = int(np.log(nu_end / nu_start) / np.log(1 + fraction))
            self.frequencies = nu_start * (1 + fraction) ** np.arange(Nnu)
        elif frequencies is not None:
            self.frequencies = np.sort(frequencies)

        self.discovery_threshold = discovery_threshold
        self.detuning_correction = detuning_correction
        self.gr = gr
        self.gDM = gDM
        self.Q = Q
        self.sigma = sigma
        self.Phipito2 = Phipito2
        self.T = T
        self.Teff_arr = T * np.ones(shape=self.frequencies.shape)

        self.gaNNexc = None
        del (
            frequencies,
            discovery_threshold,
            detuning_correction,
            gr,
            gDM,
            Q,
            sigma,
            Phipito2,
            T,
        )

    def UpdateExc(
        self,
        freq_arr=None,
        PSD=None,
        T2star=None,
        acqdelay=None,
        acqtime=None,
        measurementT=None,
        sigma=None,
        verbose=False,
    ):

        if freq_arr is None or PSD is None or measurementT is None or sigma is None:
            if verbose:
                print("Updating gaNNexc without new spectrum")
            self.gaNNexc = (
                np.sqrt(self.discovery_threshold * self.sigma)
                * self.Teff_arr ** (-0.25)
                * (self.Q / self.frequencies) ** (-1.25)
                * (0.5 * self.gr * self.gDM * self.Phipito2) ** (-1)
            )
            return 0

        if T2star is None or acqdelay is None or acqtime is None:
            raise ValueError("Input T2star / acqdelay / acqtime is None. ")

        PSDcorrection_val = (
            T2star
            / (4 * acqtime)
            * (1 - np.exp(-2 * acqtime / T2star))
            * np.exp(-2 * acqdelay / T2star)
        ) ** (-1.0)
        PSD_corrected = PSDcorrection_val * PSD

        nustart = np.amin(freq_arr)
        nustop = np.amax(freq_arr)

        if nustart < self.frequencies[0]:
            istart = 0
        elif nustart > self.frequencies[-1]:
            raise ValueError("nustart > self.frequencies[-1]")
        else:
            argminindex_start = np.argmin(abs(self.frequencies - nustart))
            if self.frequencies[argminindex_start] >= nustart:
                istart = argminindex_start
            else:
                istart = argminindex_start + 1

        if nustop > self.frequencies[-1]:
            istop = len(self.frequencies) - 1
        elif nustop < self.frequencies[0]:
            check(nustop)
            check(self.frequencies[0])
            raise ValueError("nustop < self.frequencies[0]")
        else:
            argminindex_stop = np.argmin(abs(self.frequencies - nustop))
            if self.frequencies[argminindex_stop] <= nustop:
                istop = argminindex_stop
            else:
                istop = argminindex_stop - 1
        # yinterp = np.interp(xvals, x, y)
        PSDinterp = np.interp(
            self.frequencies[istart : istop + 1], freq_arr, PSD_corrected
        )

        # self.Teff_arr
        Phipito2_new = np.sqrt(
            PSDinterp * 2
        )  #  abs(self.frequencies[istart:istop+1] * 2. / self.Q)
        # check(np.mean(Phipito2_new))

        self.Teff_arr[istart : istop + 1] += (
            measurementT
            * (np.sqrt(sigma) / Phipito2_new) ** (-4.0)
            / (np.sqrt(self.sigma) / self.Phipito2) ** (-4.0)
        )

        self.gaNNexc = (
            np.sqrt(self.discovery_threshold * self.sigma)
            * self.Teff_arr ** (-0.25)
            * (self.Q / self.frequencies / 2) ** (-1.25)
            * (0.5 * self.gr * self.gDM * self.Phipito2) ** (-1)
        )

    def PlotExc(self, verbose=False):
        self.UpdateExc(verbose=verbose)
        hbar = 6.582119569e-16  # eV.s
        c = 299792458  # m/s
        plt.rc("font", size=12)
        plt.rcParams["font.family"] = "Times New Roman"
        # plt.rcParams["font.family"] = "serif"
        # plt.rcParams["font.serif"] = ["Times New Roman"]
        plt.rcParams["mathtext.fontset"] = "cm"  # 'dejavuserif'

        fig = plt.figure(figsize=(8, 6), dpi=150)  #
        gs = gridspec.GridSpec(nrows=1, ncols=1)  #
        # fig.subplots_adjust(left=left_spc, top=top_spc, right=right_spc,
        #                     bottom=bottom_spc, wspace=xgrid_spc, hspace=ygrid_spc)
        ax = fig.add_subplot(gs[0, 0])
        # ax.plot(GammaandSAmp_arr[:, 0], GammaandSAmp_arr, label='PSD Signal Amp', color='tab:cyan', alpha=1)
        ax.plot(
            hbar / c**2 * self.frequencies,
            self.gaNNexc,
            label="",
            color=[0 / 256.0, 88 / 256.0, 155 / 256.0],
            alpha=1,
        )  # 0, 88, 155
        # \definecolor{JGUred}{RGB}{193, 0, 42} [/256., /256., /256.]
        # \definecolor{JGUgrey}{RGB}{99, 99, 99}
        # \definecolor{HIMblue}{RGB}{0, 88, 155}
        # ax.scatter(GammaandSAmp_arr, GammaandSAmp_arr, marker='x', s=30, color='tab:blue', alpha=1)

        # ax.plot(GammaandSAmp_arr, GammaandSAmp_arr, label='SNR', alpha=1)
        # ax.step(, , where='post', label='', alpha=1)
        ax.set_ylabel("$\\mathrm{g_{aNN}} / \\mathrm{GeV^{-1}}$")
        ax.set_xlabel("ALP mass / $\\mathrm{eV}$")
        # ax.set_title('PSD Signal Amplitude')
        # ax.set_xscale('log')
        ax.set_yscale("log")
        # ax.set_xticks([])
        # ax.set_yticks([])
        # ax.set_xlim(-10, 10)
        # ax.set_ylim(-0.05, 1.1)
        # ax.text(x=-10, y=1,s='(a)')
        # ax.vlines(x=taua, ymin = 1e-5, ymax = 1e3, colors='grey', linestyles='dotted', label='')
        # ax.hlines(y=1 / ((np.pi * homog0 * 1e6) + 1 / T2), xmin = 1e2, xmax = 1e6, colors='black', linestyles='dotted', label='')
        # ax.yaxis.set_major_locator(plt.NullLocator())
        # ax.xaxis.set_major_formatter(plt.NullFormatter())
        # for tick in ax.xaxis.get_major_ticks():
        #         tick.tick1line.set_visible(False)
        #         tick.tick2line.set_visible(False)
        #         tick.label1.set_visible(False)
        #         tick.label2.set_visible(False)
        ax.grid()
        plt.tight_layout()
        plt.show()
        return self.frequencies, self.gaNNexc

    def PlotExc_1PSD(
        self,
        freqoffset=0,
        PSDfreq_range=[-80, 80],
        specxaxis=None,
        spectrum=None,
        specxunit=None,
        specyunit=None,
        massfactor=1e31,
        ax_yticks=None,
        verbose=False,
    ):

        hbar = 6.582119569e-16  # eV.s
        # c = 299792458  # m/s
        # hbar = 1
        c = 1

        self.UpdateExc(verbose=verbose)

        plt.rc("font", size=12)
        plt.rcParams["font.family"] = "Times New Roman"
        # plt.rcParams["font.family"] = "serif"
        # plt.rcParams["font.serif"] = ["Times New Roman"]
        plt.rcParams["mathtext.fontset"] = "cm"  # 'dejavuserif'
        fig = plt.figure(figsize=(8 * 0.8, 6 * 0.8), dpi=150)  #
        if (
            PSDfreq_range is None
            or specxaxis is None
            or spectrum is None
            or specxunit is None
            or specyunit is None
        ):
            raise ValueError(
                "PSDfreq_range is None specxaxis is None or spectrum is None or specxunit is None or specyunit is None"
            )
        else:
            gs = gridspec.GridSpec(nrows=2, ncols=1)  #
            PSD_ax = fig.add_subplot(gs[0, 0])
            ax = fig.add_subplot(gs[1, 0])
        # fig.subplots_adjust(left=left_spc, top=top_spc, right=right_spc,
        #                     bottom=bottom_spc, wspace=xgrid_spc, hspace=ygrid_spc)
        PSD_ax.plot(
            specxaxis - freqoffset,
            spectrum,
            label="PSD ",
            color=[0 / 256.0, 88 / 256.0, 155 / 256.0],
            alpha=1,
        )
        PSD_ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
        PSD_ax.xaxis.set_label_position("top")
        if freqoffset == 0:
            PSD_ax.set_xlabel(f"Frequency / " + specxunit)  # $\\mathrm{Hz}$
        else:
            PSD_ax.set_xlabel(
                f"Frequency - {freqoffset:.0f} / " + specxunit
            )  # $\\mathrm{Hz}$
        PSD_ax.set_ylabel("PSD / " + specyunit)  # $\Phi_{0}^{2}/\\mathrm{Hz}$
        PSD_ax.grid()
        PSD_ax.set_xlim(PSDfreq_range[0], PSDfreq_range[1])
        # PSD_ax.set_ylim(top=3.2, bottom=-0.1)
        freq_start, freq_stop = PSD_ax.get_xlim()
        PSDxticks = PSD_ax.get_xticks()

        ax.plot(
            2 * np.pi * hbar / c**2 * massfactor * (self.frequencies - freqoffset),
            self.gaNNexc,
            label="gaNN",
            color=np.array([193, 0, 42]) / 256.0,
            alpha=1,
        )  # 0, 88, 155
        # ax.scatter(GammaandSAmp_arr, GammaandSAmp_arr, marker='x', s=30, color='tab:blue', alpha=1)
        ax.set_xlabel(
            f"ALP mass - {hbar / c ** 2 * freqoffset * massfactor:.0f}"
            + " / $10^{-%.0f}\\mathrm{eV} c^{-2}$" % (np.log10(massfactor))
        )
        ax.set_ylabel("$\\mathrm{g_{aNN}}$" + " / " + "$\\mathrm{GeV^{-1}}$")
        # ax.set_title('PSD Signal Amplitude')
        # ax.set_xscale('log')
        ax.set_yscale("log")
        ax.set_xticks(2 * np.pi * hbar / c**2 * massfactor * PSDxticks)
        ax.xaxis.set_major_formatter("{x:.2f}")
        # ax.set_yticks([])
        ax.set_xlim(
            2 * np.pi * hbar / c**2 * massfactor * (freq_start),
            2 * np.pi * hbar / c**2 * massfactor * (freq_stop),
        )
        ax.set_ylim(top=10 ** (-1.8), bottom=10 ** (-6.2))
        if ax_yticks is not None:
            ax.set_yticks(ax_yticks)
        ax.fill_between(
            2 * np.pi * hbar / c**2 * massfactor * (self.frequencies - freqoffset),
            self.gaNNexc,
            1e5,
            color="r",
            alpha=0.2,
        )

        ax.grid()
        letters = [
            "(a)     ",
            "(b)     ",
            "(c)",
            " (d)",
            " (e)",
            " (f)",
            " (g)",
            " (h)",
            " (i)",
        ]
        for i, axi in enumerate([PSD_ax]):
            xleft, xright = axi.get_xlim()
            ybottom, ytop = axi.get_ylim()
            axi.text(
                x=xleft,
                y=ytop,
                s=letters[i],
                ha="right",
                va="center",
                color="blue",
                fontsize=14,
            )
        for i, axi in enumerate([ax]):
            xleft, xright = axi.get_xlim()
            ybottom, ytop = axi.get_ylim()
            axi.text(
                x=xleft,
                y=ytop,
                s=letters[i + 1],
                ha="right",
                va="top",
                color="blue",
                fontsize=14,
            )
        plt.tight_layout()
        plt.show()
        return self.frequencies, self.gaNNexc
