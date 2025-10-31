from src.functioncache import axion_lineshape, check

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import matplotlib.ticker as mticker
from mpl_toolkits.mplot3d import Axes3D  # for type hinting

import numba as nb
from math import sin, cos, sqrt
from scipy.stats import maxwell, uniform, expon

import h5py

from src.DataAnalysis import DualChanSig
from src.Sample import Sample
from src.Apparatus import LockinAmplifier, Pickup, SQUID, Magnet
from src.Envelope import PhysicalQuantity


def gate(x: float | np.ndarray, start: float, stop: float) -> float:
    """
    Returns 1 if start <= x <= stop, else returns 0.

    Parameters:
    x : float or array-like
        The input value(s) where the function is evaluated.

    Returns:
    float or array-like
        1 if start <= x <= stop, else 0.
    """
    return np.where((start <= x) & (x < stop), 1.0, 0.0)


class MagField:
    """
    DC / AC (pseudo)magnetic fields
    """

    def __init__(
        self,
        name="B field",
    ):
        self.name = name
        self.nu = None

    def setXYPulse(
        self,
        timeStamp: np.ndarray,
        B1: float,  # amplitude of the excitation pulse in [T]
        nu_rot: float,
        init_phase: float,
        direction: np.ndarray,  #  not needed now
        duty_func,
        verbose: bool = False,
    ):
        """
        generate a pulse in the rotating frame
        """
        direction_norm = direction / np.dot(direction, direction)

        # excitation along x-axis
        Bx_envelope = (
            1.0
            / 2
            * B1
            * duty_func(timeStamp)
            # * np.dot(np.array([1, 0, 0]), direction_norm)
        )
        # check(Bx_envelope[0:10])
        Bx_envelope = np.multiply(
            Bx_envelope, np.cos(2 * np.pi * nu_rot * timeStamp + init_phase)
        )
        Bx = np.outer(Bx_envelope, np.array([1, 0, 0]))

        # excitation along y-axis
        By_envelope = (
            1.0
            / 2
            * B1
            * duty_func(timeStamp)
            # * np.dot(np.array([0, 1, 0]), direction_norm)
        )
        # check(By_envelope)
        By_envelope = np.multiply(
            By_envelope, np.sin(2 * np.pi * nu_rot * timeStamp + init_phase)
        )
        By = np.outer(By_envelope, np.array([0, 1, 0]))

        # 1st order time-derivate of the excitation along x-axis
        dBxdt_envelope = (
            1.0
            / 2
            * B1
            * duty_func(timeStamp)
            # * np.dot(np.array([1, 0, 0]), direction_norm)
        )
        dBxdt_envelope = np.multiply(
            dBxdt_envelope,
            -2 * np.pi * nu_rot * np.sin(2 * np.pi * nu_rot * timeStamp + init_phase),
        )
        dBxdt = np.outer(dBxdt_envelope, np.array([1, 0, 0]))

        # 1st order time-derivate of the excitation along y-axis
        dBydt_envelope = (
            1.0
            / 2
            * B1
            * duty_func(timeStamp)
            # * np.dot(np.array([0, 1, 0]), direction_norm)
        )
        dBydt_envelope = np.multiply(
            dBydt_envelope,
            2 * np.pi * nu_rot * np.cos(2 * np.pi * nu_rot * timeStamp + init_phase),
        )
        dBydt = np.outer(dBydt_envelope, np.array([0, 1, 0]))

        self.B_vec = Bx + By
        self.dBdt_vec = dBxdt + dBydt

        # self.dBdt_vec = np.outer(dBxdt + dBydt, direction)
        # self.nu = nu_rot
        # def envelope(timeStamp):
        #     return duty_func(timeStamp) * B1 * np.sin(2 * np.pi * nu_e * timeStamp + init_phase)
        # return

    def showTSandPSD(
        self, dataX: np.ndarray, dataY: np.ndarray, demodfreq, samprate, showplt_opt
    ):
        """
        dataX=,
        dataY=,
        demodfreq=,
        samprate=,
        showplt_opt=True
        """
        stream = DualChanSig(
            name="ALP field gradient",
            # device="Simulation",
            # device_id="Simulation",
            filelist=[],
            verbose=True,
        )
        stream.attenuation = 0
        stream.filterstatus = "off"
        stream.filter_TC = 0.0
        stream.filter_order = 0
        stream.demodfreq = demodfreq
        stream.samprate = samprate

        stream.dataX = dataX
        stream.dataY = dataY

        stream.GetNoPulsePSD(
            # windowfunction="Hanning",
            windowfunction="rectangle",
            # decayfactor=-10,
            chunksize=None,  # sec
            analysisrange=[0, -1],
            getstd=False,
            stddev_range=None,
            selectshots=[],
            verbose=False,
        )
        # stream.FitPSD(
        #     fitfunction="Lorentzian",  # 'Lorentzian' 'dualLorentzian' 'tribLorentzian' 'Gaussian 'dualGaussian' 'auto' 'Polyeven'
        #     inputfitparas=["auto", "auto", "auto", "auto"],
        #     smooth=False,
        #     smoothlevel=1,
        #     fitrange=["auto", "auto"],
        #     alpha=0.05,
        #     getresidual=False,
        #     getchisq=False,
        #     verbose=False,
        # )
        specxaxis, spectrum, specxunit, specyunit = stream.GetSpectrum(
            showtimedomain=True,
            # showfit=True,
            showresidual=False,
            showlegend=True,  # !!!!!show or not to show legend
            spectype="PSD",  # in 'PSD', 'ASD', 'FLuxPSD', 'FluxASD'
            ampunit="V",
            specxunit="Hz",  # 'Hz' 'kHz' 'MHz' 'GHz' 'ppm' 'ppb'
            # specxlim=[nu_a + demodfreq - 5, nu_a + demodfreq + 20],
            # specylim=[0, 4e-23],
            specyscale="linear",  # 'log', 'linear'
            # showstd=False,
            showplt_opt=showplt_opt,
            return_opt=True,
        )
        return specxaxis, spectrum, specxunit, specyunit

    def setALP_Field(
        self,
        method: str,  # 'inverse-FFT' 'time-interfer'
        timeStamp: np.ndarray,
        simuRate: float,
        duration: float,
        Bamp: float,  # amplitude of the pseudo-magnetic field in [T]
        nu_a: float,  # frequency in the rotating frame
        use_stoch: bool,
        # direction: np.ndarray,  #  = np.array([1, 0, 0])
        demodFreq: float,
        rand_seed: int = None,
        makeplot: bool = False,
        verbose: bool = False,
    ):
        """
        generate a pseudo-magnetic field (ALP field gradient)
        """
        timeStep = np.abs(timeStamp[1] - timeStamp[0])
        timeLen = len(timeStamp)
        # envelope_f = 0
        self.nu = nu_a

        def setALP_Field_timeIntf():
            """
            generate Bx, By, dBxdt, dBydt
            """
            frequencies = np.linspace(
                -0.5 / timeStep, 0.5 / timeStep, num=timeLen, endpoint=True
            )
            lineshape = axion_lineshape(
                v_0=220e3,
                v_lab=233e3,
                nu_a=nu_a + demodFreq,
                nu=frequencies + demodFreq,
                case="grad_perp",
                alpha=0.0,
            )

            rvs_amp = expon.rvs(loc=0.0, scale=1.0, size=timeLen)
            # rvs_amp = 1.0
            rvs_phase = np.exp(1j * uniform.rvs(loc=0, scale=2 * np.pi, size=timeLen))
            # rvs_phase = 1.0

            if use_stoch:
                ax_sq_lineshape = lineshape * rvs_amp
            else:
                ax_sq_lineshape = lineshape
            ax_lineshape = np.sqrt(ax_sq_lineshape)
            # if makeplot:
            #     plt.figure()
            #     plt.plot(ax_lineshape)
            #     plt.show()
            # check lineshape sanity
            # for arr in [lineshape, ax_lineshape]:
            #     has_nan = np.isnan(arr).any()  # Check for NaN
            #     has_inf = np.isinf(arr).any()  # Check for Inf

            #     print(f"Contains NaN: {has_nan}")  # Output: True
            #     print(f"Contains Inf: {has_inf}")  # Output: True

            # inverse FFT method
            # ax_FFT = np.sqrt(stoch_a_sq_lineshape) * rvs_phase
            # Ba_t = np.fft.ifft(ax_FFT)
            # Bx = np.outer(np.real(Ba_t), np.array([1, 0, 0]))
            # By = np.outer(np.imag(Ba_t), np.array([1, 0, 0]))
            # self.B_vec = Bx + By

            # Find the index of the first non-zero element
            # nonzero_indices = np.nonzero(ax_lineshape)[0]
            # first_nonzero_index = (
            #     nonzero_indices[0] if nonzero_indices.size > 0 else None
            # )
            positive_indices = np.where(ax_lineshape > 0)[0]
            if positive_indices.size > 0:
                first_positive_index = positive_indices[0]
            else:
                first_positive_index = 0

            Bx_amp = np.zeros(timeLen)
            By_amp = np.zeros(timeLen)
            dBxdt_amp = np.zeros(timeLen)
            dBydt_amp = np.zeros(timeLen)

            if use_stoch:
                init_phase = uniform.rvs(loc=0, scale=2 * np.pi, size=timeLen)
            else:
                init_phase = uniform.rvs(loc=0, scale=2 * np.pi, size=1) * np.ones(
                    timeLen
                )

            for i in np.arange(first_positive_index, timeLen, dtype=int):
                nu_rot = frequencies[i]
                ax_amp = ax_lineshape[i]
                #  phase
                # init_phase_0 = init_phase[i]
                # By_init_phase = uniform.rvs(loc=0, scale=2 * np.pi, size=1)
                # # fixed phase
                # Bx_init_phase = 0
                # By_init_phase = 0

                # Bx
                Bx_amp += (
                    0.5
                    * Bamp
                    * ax_amp
                    * np.cos(2 * np.pi * nu_rot * timeStamp + init_phase[i])
                )
                # By
                By_amp += (
                    0.5
                    * Bamp
                    * ax_amp
                    * np.sin(2 * np.pi * nu_rot * timeStamp + init_phase[i])
                )
                # dBx / dt
                dBxdt_amp += (
                    0.5
                    * Bamp
                    * ax_amp
                    * (2 * np.pi * nu_rot)
                    * np.cos(2 * np.pi * nu_rot * timeStamp + init_phase[i])
                )
                # dBy / dt
                dBydt_amp += (
                    0.5
                    * Bamp
                    * ax_amp
                    * (-2 * np.pi * nu_rot)
                    * np.sin(2 * np.pi * nu_rot * timeStamp + init_phase[i])
                )

            Bx = np.outer(Bx_amp, np.array([1, 0, 0]))
            By = np.outer(By_amp, np.array([0, 1, 0]))
            dBxdt = np.outer(dBxdt_amp, np.array([1, 0, 0]))
            dBydt = np.outer(dBydt_amp, np.array([0, 1, 0]))

            self.B_vec = Bx + By
            self.dBdt_vec = dBxdt + dBydt

            if makeplot:
                self.B_Stream = DualChanSig(
                    name="ALP field gradient",
                    device="Simulation",
                    device_id="Simulation",
                    filelist=[],
                    verbose=True,
                )
                self.B_Stream.attenuation = 0
                self.B_Stream.filterstatus = "off"
                self.B_Stream.filter_TC = 0.0
                self.B_Stream.filter_order = 0
                self.B_Stream.demodfreq = demodFreq
                saveintv = 1
                self.B_Stream.samprate = 1.0 / timeStep / saveintv

                self.B_Stream.dataX = 1 * self.B_vec[0:-1:saveintv, 0]  # * \
                # np.cos(2 * np.pi * self.nu_rot * self.timestamp[0:-1:saveintv])
                self.B_Stream.dataY = 1 * self.B_vec[0:-1:saveintv, 1]

                self.B_Stream.GetNoPulsePSD(
                    windowfunction="Hanning",
                    # decayfactor=-10,
                    chunksize=None,  # sec
                    analysisrange=[0, -1],
                    getstd=False,
                    stddev_range=None,
                    selectshots=[],
                    verbose=False,
                )
                # self.B_Stream.FitPSD(
                #     fitfunction="Lorentzian",  # 'Lorentzian' 'dualLorentzian' 'tribLorentzian' 'Gaussian 'dualGaussian' 'auto' 'Polyeven'
                #     inputfitparas=["auto", "auto", "auto", "auto"],
                #     smooth=False,
                #     smoothlevel=1,
                #     fitrange=["auto", "auto"],
                #     alpha=0.05,
                #     getresidual=False,
                #     getchisq=False,
                #     verbose=False,
                # )
                specxaxis, spectrum, specxunit, specyunit = self.B_Stream.GetSpectrum(
                    showtimedomain=True,
                    showfit=True,
                    showresidual=False,
                    showlegend=True,  # !!!!!show or not to show legend
                    spectype="PSD",  # in 'PSD', 'ASD', 'FLuxPSD', 'FluxASD'
                    ampunit="V",
                    specxunit="Hz",  # 'Hz' 'kHz' 'MHz' 'GHz' 'ppm' 'ppb'
                    specxlim=[nu_a + demodFreq - 5, nu_a + demodFreq + 20],
                    # specylim=[0, 4e-23],
                    specyscale="linear",  # 'log', 'linear'
                    showstd=False,
                    showplt_opt=True,
                    return_opt=True,
                )

        def setALP_Field_invFFT_2():
            """
            generate Bx, By, dBxdt, dBydt
            """
            frequencies = np.linspace(
                -0.5 / timeStep, 0.5 / timeStep, num=timeLen, endpoint=True
            )
            lineshape = axion_lineshape(
                v_0=220e3,
                v_lab=233e3,
                nu_a=nu_a + demodFreq,
                nu=frequencies + demodFreq,
                case="grad_perp",
                alpha=0.0,
            )

            # # check the normalization of the axion field PSD lineshape
            # # np.sum(lineshape * freq_resol) should be 1
            # freq_resol = np.abs(frequencies[0] - frequencies[1])
            # check(np.sum(lineshape * freq_resol))
            # del freq_resol

            rng = (
                np.random.default_rng(seed=rand_seed) if rand_seed is not None else None
            )

            rvs_amp = expon.rvs(loc=0.0, scale=1.0, size=timeLen, random_state=rng)
            # rvs_amp = 1.0

            rvs_phase = np.exp(
                1j * uniform.rvs(loc=0, scale=2 * np.pi, size=timeLen, random_state=rng)
            )
            # rvs_phase = 1.0

            """
            if rand_seed is not None:
                base_rng = np.random.default_rng(seed=rand_seed)
                rng_amp = np.random.default_rng(base_rng.integers(0, 2**32))
                rng_phase = np.random.default_rng(base_rng.integers(0, 2**32))
            else:
                rng_amp = None
                rng_phase = None

            rvs_amp = expon.rvs(loc=0.0, scale=1.0, size=timeLen, random_state=rng_amp)
            # rvs_amp = 1.0

            rvs_phase = np.exp(
                1j * uniform.rvs(loc=0, scale=2 * np.pi, size=timeLen, random_state=rng_phase)
            )
            # rvs_phase = 1.0
            """

            if use_stoch:
                ax_sq_lineshape = lineshape * rvs_amp
            else:
                ax_sq_lineshape = lineshape
            ax_lineshape = np.sqrt(ax_sq_lineshape)
            # check(np.mean(ax_lineshape))

            # inverse FFT method
            ax_FFT = ax_lineshape * rvs_phase * Bamp * simuRate * np.sqrt(duration)
            # area_simps = np.trapz(ax_lineshape**2, frequencies)

            # check(np.trapz(np.abs(ax_lineshape* rvs_phase)**2, frequencies))

            # the following block is equivalent to ax_FFT_pos_neg = np.fft.fftshift(ax_FFT)
            # length = len(ax_FFT)
            # ax_FFT_pos_neg = np.array(
            #     [ax_FFT[length // 2 :], ax_FFT[: length // 2]]
            # ).flatten()
            # del length

            ax_FFT_0_pos_neg = np.fft.fftshift(ax_FFT)

            Ba_t = np.fft.ifft(ax_FFT_0_pos_neg)
            Bx_amp, By_amp = np.real(Ba_t), np.imag(Ba_t)
            # check(np.mean(np.abs(Bx_amp)))
            # check(np.mean(np.abs(By_amp)))
            N = len(ax_FFT)
            freq = np.fft.fftfreq(N, timeStep)
            dBadt_FFT = 1j * 2 * np.pi * freq * ax_FFT_0_pos_neg
            dBadt = np.fft.ifft(dBadt_FFT)
            dBxdt_amp, dBydt_amp = np.real(dBadt), np.imag(dBadt)

            Bx = np.outer(Bx_amp, np.array([1, 0, 0]))
            By = np.outer(By_amp, np.array([0, 1, 0]))
            dBxdt = np.outer(dBxdt_amp, np.array([1, 0, 0]))
            dBydt = np.outer(dBydt_amp, np.array([0, 1, 0]))

            self.B_vec = Bx + By
            self.dBdt_vec = dBxdt + dBydt

            if makeplot:
                # check(self.B_vec[::1, 0])
                specxaxis, spectrum0, specxunit, specyunit = self.showTSandPSD(
                    dataX=self.B_vec[::1, 0],
                    dataY=self.B_vec[::1, 1],
                    demodfreq=demodFreq,
                    samprate=1.0 / timeStep,
                    showplt_opt=False,
                )
                specxaxis, spectrum1, specxunit, specyunit = self.showTSandPSD(
                    dataX=self.dBdt_vec[::1, 0],
                    dataY=self.dBdt_vec[::1, 1],
                    demodfreq=demodFreq,
                    samprate=1.0 / timeStep,
                    showplt_opt=False,
                )
                fig = plt.figure(figsize=(6.0, 4.0), dpi=150)  # initialize a figure
                gs = gridspec.GridSpec(
                    nrows=2, ncols=2
                )  # create grid for multiple figures
                ax_x = fig.add_subplot(gs[0, 0])
                ax_y = fig.add_subplot(gs[1, 0])
                axPSD = fig.add_subplot(gs[:, 1])

                ax_x.plot(self.B_vec[:, 0] / np.amax(self.B_vec), label="Bx")
                ax_x.plot(self.dBdt_vec[:, 0] / np.amax(self.dBdt_vec), label="dBxdt")

                ax_y.plot(self.B_vec[:, 1] / np.amax(self.B_vec), label="By")
                ax_y.plot(self.dBdt_vec[:, 1] / np.amax(self.dBdt_vec), label="dBydt")

                axPSD.plot(
                    specxaxis,
                    spectrum0 / np.amax(spectrum0),
                    label="ALP field gradient PSD",
                )
                axPSD.plot(
                    specxaxis,
                    spectrum1 / np.amax(spectrum1),
                    label="dBa_dt PSD",
                    linestyle="--",
                )
                axPSD.set_xlabel("")
                axPSD.set_ylabel("")
                # ax00.set_xscale('log')
                # ax00.set_yscale("log")
                # #############################################################################
                ax_x.legend()
                ax_y.legend()
                axPSD.legend()
                # #############################################################################
                fig.suptitle("", wrap=True)
                plt.tight_layout()
                plt.show()

        if method == "inverse-FFT":
            # setALP_Field_invFFT()
            setALP_Field_invFFT_2()
        elif method == "time-interfer":
            setALP_Field_timeIntf()
        else:
            raise ValueError("method not found")

    def plotField(self, demodfreq, samprate, showplt_opt):
        specxaxis, spectrum, specxunit, specyunit = self.showTSandPSD(
            dataX=self.B_vec[:, 0],
            dataY=self.B_vec[:, 1],
            demodfreq=demodfreq,
            samprate=samprate,
            showplt_opt=showplt_opt,
        )
        return specxaxis, spectrum, specxunit, specyunit


class Simulation:
    # NMR simulation in the rotating frame
    def __init__(
        self,
        name="NMR simulation",
        sample: Sample = None,  # class Sample
        pickup: Pickup = None,
        SQUID: SQUID = None,
        LIA:LockinAmplifier=None,
        magnet_pol: Magnet = None,
        magnet_det: Magnet = None,
        station=None,  # refer to class Station
        init_time=0.0,  # [s]. When at time = 0, the ALP wind comes in the direction of (theta, phi) = (90°+23°26′, 0)
        init_mag_amp=1,  # initial magnetization vector amplitude
        init_M_theta=0.0,  # [rad]
        init_M_phi=0.0,  # [rad]
        init_phase=0.0 * np.pi / 180,  # [rad]
        # demodFreq: PhysicalQuantity = None,
        # B0z=0.0,  # [T]  B0 in the laboratory frame.
        simuRate: PhysicalQuantity = None,  # simulation rate in [Hz]. simulation step = 1 / simurate
        duration: PhysicalQuantity = None,
        excField=None,  # class MagField
        verbose=True,
    ):
        """
        initialize NMR simulation
        always assume rotating coordinate frame
        """
        self.name = name
        self.sample = sample
        self.pickup = pickup
        self.SQUID = SQUID
        self.LIA=LIA
        self.magnet_pol = magnet_pol
        self.magnet_det = magnet_det

        self.gamma_HzToT = self.sample.gamma.convert_to("Hz/T").value

        self.station = station

        self.init_time = init_time

        self.init_M_amp = init_mag_amp
        self.init_M_theta = init_M_theta
        self.init_M_phi = init_M_phi

        self.init_phase = init_phase
        self.B0z_T = magnet_det.B0.convert_to("T").value
        self.simuRate_Hz = simuRate.convert_to("Hz").value
        self.timeStep_s = (
            1.0 / self.simuRate_Hz
        )  # the key parameter in setting simulation timing
        self.duration_s = duration.convert_to("s").value
        self.timeStamp = (
            np.arange(int(self.duration_s * self.simuRate_Hz)) * self.timeStep_s
        )
        # self.timestamp_t
        self.numOfSimuSteps: int = len(self.timeStamp) - 1

        self.excField: MagField = excField
        # self.simutimestep = self.simustep * self.ALPwind.cohT
        # self.simutimerate = 1.0 / self.simutimestep
        self.demodFreq_Hz = self.LIA.demodFreq.value_in("Hz")
        print(self.demodFreq_Hz)
        self.nuL_Hz = (
            abs(self.gamma_HzToT * self.B0z_T / (2 * np.pi)) - self.demodFreq_Hz
        )  # Larmor frequency
        # nu_rot is the Larmor frequency of the magnetization in the rotating frame

        # check(self.nu_rot)
        # check(abs(self.sample.gyroratio*self.B0z/(2*np.pi)))
        # check(abs(self.ALPwind.nu))
        # self.T2 = 1.0 * sample.T2
        # self.T1 = 1.0 * sample.T1
        self.T2_s = sample.T2.convert_to("s").value
        self.T1_s = sample.T1.convert_to("s").value

        if self.simuRate_Hz < 10 * abs(self.nuL_Hz):
            print("WARNING: self.simurate < 10*self.nu_rot")
            check(self.simuRate_Hz)
            check(self.nuL_Hz)

        # move it somewhere later
        # if self.simuRate < 10 * abs(
        #     self.sample.gyroratio * self.excField.BALP / (2 * np.pi)
        # ):
        #     print(
        #         "WARNING: self.simurate < 10 * abs(self.gyroratio*self.BALP/(2*np.pi))"
        #     )

        # self.va = 300 * 1e3

        if verbose:
            # print(f"Larmor frequency: {self.B0z_T*self.sample.gamma.value_in("Hz/T")/(2*np.pi):e} Hz")
            # print(f"ALP compton frequency: {self.excField.nu:e} Hz")
            print(f"simulation rate: {self.simuRate_Hz:e} Hz")

        # if abs(
        #     self.B0z * self.sample.gyroratio / (2 * np.pi) - self.excField.nu
        # ) / self.excField.nu > 20 * 10 ** (-6):
        #     print("WARNING: NMR frequency is > 20 ppm away from excitation frequency")
        #     print(f"Larmor frequency: {self.B0z*self.sample.gyroratio/(2*np.pi):e} Hz")
        #     print(f"ALP compton frequency: {self.excField.nu:e} Hz")
        if self.T2_s > self.T1_s:
            print("WARNING: T2 is larger than T1")
            # warnings.warn("T2 is larger than T1", DeprecationWarning)

    # def RandomJump(
    #     self,
    #     numofcohT=None,  # float. number of coherent period for simulation
    #     verbose=False,
    # ):
    #     """
    #     Generate simulation parameters by 'RandomJump' method.

    #     Parameters
    #     ----------
    #     numofcohT : float

    #     Number of coherent period for simulation. Can be integer or float values. Default in None.

    #     verbose : bool

    #     """
    #     numofsampling = int(np.ceil(numofcohT))
    #     check(numofsampling)
    #     BALPamp_array = statRayleigh(
    #         sigma=self.excField.BALP,
    #         num=numofsampling,
    #         showplt=verbose,
    #         verbose=verbose,
    #     )
    #     theta_array = (
    #         statUni2Pi(num=numofsampling, showplt=verbose, verbose=verbose) / 2.0
    #     )
    #     phi_array = statUni2Pi(num=numofsampling, showplt=verbose, verbose=verbose)
    #     phase0_array = statUni2Pi(num=numofsampling, showplt=verbose, verbose=verbose)
    #     BALP_list = []

    #     BALPamp_array[0] = self.excField.BALP
    #     check(BALPamp_array)
    #     theta_array[0] = 90.0 * np.pi / 180.0
    #     phi_array[0] = 0.0 * np.pi / 180.0
    #     phase0_array[0] = 0

    #     numofsimustep_perperiod = int(self.simuRate_Hz * self.excField.cohT)
    #     timestamp = np.linspace(
    #         start=0,
    #         stop=numofsampling * self.excField.cohT,
    #         num=int(np.ceil(self.simuRate_Hz * numofsampling * self.excField.cohT)),
    #     )
    #     for i in range(numofsampling - 1):
    #         BALP = 0.5 * BALPamp_array[i]
    #         theta = theta_array[i]
    #         phi = phi_array[i]
    #         Bx = BALP * np.sin(theta) * np.cos(phi)
    #         By = BALP * np.sin(theta) * np.sin(phi)
    #         Bz = BALP * np.cos(theta)
    #         phase0 = phase0_array[i]
    #         for j in range(numofsimustep_perperiod):
    #             BALP_list.append([Bx, By, Bz, phase0].copy())  # room for modification?
    #     for i in [-1]:
    #         BALP = 0.5 * BALPamp_array[i]
    #         theta = theta_array[i]
    #         phi = phi_array[i]
    #         Bx = (
    #             BALP * np.sin(theta) * np.cos(phi) * np.cos(phi)
    #         )  # *np.cos(2*np.pi*self.nu_rot*timestamp)
    #         By = (
    #             BALP * np.sin(theta) * np.sin(phi) * np.sin(phi)
    #         )  # *np.sin(2*np.pi*self.nu_rot*timestamp)
    #         Bz = BALP * np.cos(theta)
    #         phase0 = phase0_array[i]
    #         numofstep_lastperiod = int(
    #             self.simuRate_Hz * self.excField.cohT * (numofcohT - numofsampling + 1)
    #         )
    #         check(numofstep_lastperiod)
    #         for j in range(numofstep_lastperiod):
    #             BALP_list.append(
    #                 [Bx, By, Bz, phase0].copy()
    #             )  # self.BALP_array [Bx, By, Bz, phase0]
    #     del BALPamp_array, theta_array, phi_array, phase0_array
    #     self.Bexc_t_vec = np.array(BALP_list)  # self.BALP_array [Bx, By, Bz, phase0]
    #     self.dBexc_dt_vec = np.zeros(self.Bexc_t_vec.shape)
    #     # check(self.BALP_array)
    #     self.timeStamp = np.arange(len(self.Bexc_t_vec) + 1) * self.timeStep_s

    # @nb.jit(
    #     [
    #         "int16(int64, int64, float64[:], \
    #     float64, float64[:], float64[:], \
    #     float64[:], float64[:,:], float64[:,:],     \
    #     float64[:], float64, float64, \
    #     float64, float64)"
    #     ],
    #     nopython=True,
    # )
    # def ThermalLightLoop(
    #     numofsimustep,  # int
    #     numofALPparticle,  # int
    #     random_phase,  # float[:]
    #     ALP_B,  # float
    #     ALP_nu_rot,  # float[:]
    #     ALP_phase,  # float[:]
    #     ALP_vtova_arr,  # float[:]
    #     BALP_arr,  # float[:,:]
    #     dBALPdt_arr,  # float[:,:]
    #     ALPwind_direction_earth,  # float[:]
    #     init_time,  # float
    #     simustep,  # float
    #     station_theta,  # float
    #     station_phi,  # float
    # ):

    #     # BALP_list = np.array([])
    #     for i in range(numofsimustep):
    #         # if verbose and i%10000 == 0:
    #         #     check(i)
    #         # decide BALP amplitude and phase0 from 2D random walk and decoherence due to 2 particle collision in each step
    #         # update ALP particles' phase array
    #         (
    #             ALP_phase[(2 * i) % (2 * numofALPparticle)],
    #             ALP_phase[(2 * i + 1) % (2 * numofALPparticle)],
    #         ) = (random_phase[2 * i], random_phase[2 * i + 1])
    #         # BALP = abs(rw)  # ALPwind_BALP *
    #         # phase0 = np.angle(rw)
    #         BALP = np.sum(
    #             ALP_B
    #             * ALP_vtova_arr
    #             * np.sin(
    #                 2 * np.pi * ALP_nu_rot * (init_time + i * simustep) + ALP_phase
    #             )
    #         ) / sqrt(2.0 * numofALPparticle)
    #         dBALPdt = np.sum(
    #             ALP_B
    #             * ALP_vtova_arr
    #             * 2
    #             * np.pi
    #             * ALP_nu_rot
    #             * np.cos(
    #                 2 * np.pi * ALP_nu_rot * (init_time + i * simustep) + ALP_phase
    #             )
    #         ) / sqrt(2.0 * numofALPparticle)
    #         # decide the direciton of B_ALP from the experiment time and motion of celestial bodies
    #         theta_e = ALPwind_direction_earth[1]  #
    #         phi_e = ALPwind_direction_earth[2]  #
    #         theta_s = station_theta  # theta_station
    #         phi_s = (
    #             init_time + i * simustep
    #         ) * 2 * 3.141592 / 86164.0 + station_phi  # phi_station

    #         x = sin(theta_e) * cos(theta_s) * cos(phi_e - phi_s) - cos(theta_e) * sin(
    #             theta_s
    #         )
    #         y = sin(theta_e) * sin(phi_e - phi_s)
    #         z = sin(theta_e) * sin(theta_s) * cos(phi_e - phi_s) + cos(theta_e) * cos(
    #             theta_s
    #         )
    #         Bx, By, Bz = 0.5 * BALP * np.array([x, y, z])
    #         dBxdt, dBydt, dBzdt = (
    #             0.5 * dBALPdt * np.array([x, y, z])
    #         )  # to be improved!!!!!!
    #         BALP_arr[i] = [Bx, By, Bz]  # , phase0 to be improved?
    #         dBALPdt_arr[i] = [dBxdt, dBydt, dBzdt]
    #         # BALP_modu_list.append(direction_lab)
    #     # BALP_array = np.array(BALP_list)
    #     # return BALP_array
    #     return 0

    # def ThermalLight(
    #     self,
    #     numofcohT=None,  # float. number of coherent period for simulation
    #     usenumba=True,
    #     verbose=False,
    # ):
    #     """
    #     Generate parameters by 'thermal light' method.

    #     Parameters
    #     ----------
    #     numofcohT : float

    #     Number of coherent period for simulation. Can be integer or float values. Default in None.

    #     verbose : bool

    #     """
    #     numofsimustep = int(self.simuRate_Hz * self.excField.cohT * numofcohT)
    #     # initialize an array of particles
    #     # rw2D = []
    #     # numofALPparticle = int(self.simurate*self.ALPwind.cohT)
    #     self.numofALPparticle = int(self.simuRate_Hz * self.excField.cohT)
    #     if self.numofALPparticle < 1000:
    #         print("WARNING: self.numofALPparticle < 1000")
    #     ALP_phase = statUni2Pi(
    #         num=2 * self.numofALPparticle, showplt=False, verbose=False
    #     )
    #     sin_arr = np.sin(ALP_phase)
    #     cos_arr = np.cos(ALP_phase)
    #     random_phase = uniform.rvs(loc=0, scale=2 * np.pi, size=2 * numofsimustep)

    #     self.ALP_B = (
    #         self.excField.BALP
    #     )  # why? * (1 + self.ALPwind.Gamma * np.random.standard_cauchy(size=2 * self.numofALPparticle))
    #     ALP_nu = self.excField.nu * (1 + (self.ALP_B / self.excField.BALP * 1e-3) ** 2)
    #     ALP_nu_rot = abs(ALP_nu) - abs(self.sample.gamma.value_in("Hz/T") * self.B0z_T / (2 * np.pi))
    #     self.va = 220 * 1e3  # 220 km/s
    #     self.speedtova = maxwell.rvs(size=2 * self.numofALPparticle)

    #     plt.rc("font", size=16)
    #     if verbose:
    #         fig = plt.figure(figsize=(8 * 0.6, 6 * 0.6), dpi=150)  #
    #         gs = gridspec.GridSpec(nrows=1, ncols=1)  #
    #         # fig.subplots_adjust(left=left_spc, top=top_spc, right=right_spc,
    #         #                     bottom=bottom_spc, wspace=xgrid_spc, hspace=ygrid_spc)
    #         ax = fig.add_subplot(gs[0, 0])
    #         ax.hist(self.speedtova, bins=100, density=True, color="tab:green")
    #         ax.set_xlabel("$|v|/v_0$")
    #         ax.set_ylabel("Probability density")
    #         # plt.title('Speed distribution')
    #         plt.tight_layout()
    #         plt.show()
    #         # check(np.amin(self.speedtova))

    #     self.nu_a_arr = self.excField.nu * (1.0 + (self.speedtova * self.va / 3e8) ** 2)
    #     # check(np.amin(self.nu_a_arr))
    #     if verbose:
    #         fig = plt.figure(figsize=(8 * 0.6, 6 * 0.6), dpi=150)  #
    #         gs = gridspec.GridSpec(nrows=1, ncols=1)  #
    #         # fig.subplots_adjust(left=left_spc, top=top_spc, right=right_spc,
    #         #                     bottom=bottom_spc, wspace=xgrid_spc, hspace=ygrid_spc)
    #         ax = fig.add_subplot(gs[0, 0])
    #         ax.hist(
    #             1e6 * (self.nu_a_arr / self.excField.nu - 1),
    #             bins=100,
    #             density=True,
    #             color="tab:purple",
    #         )
    #         ax.set_xlabel("$(\\nu/\\nu_a-1) \\times 10^6$")

    #         ax.set_ylabel("Probability density")
    #         ax.set_xlim(-0.5, 10.5)  # [0, 1, 2, 3, 4, 5, 6, 7,8 , 9, 10]
    #         ax.set_xticks([0, 2, 4, 6, 8, 10])  # [0, 1, 2, 3, 4, 5, 6, 7,8 , 9, 10]
    #         # plt.title('Speed distribution')
    #         plt.tight_layout()
    #         plt.show()

    #         # plt.hist(1e6 * (self.nu_a_arr/self.ALPwind.nu-1), bins=100, density=True)
    #         # plt.xlabel('$(\\nu/\\nu_a-1) \\times 10^6$')
    #         # plt.ylabel('Distribution density')
    #         # plt.title('Frequency distribution')
    #         # plt.show()

    #     ALP_nu_rot_arr = self.nu_a_arr - abs(
    #         self.sample.gamma.value_in("Hz/T") * self.B0z_T / (2 * np.pi)
    #     )
    #     # plt.hist(ALP_B/self.ALPwind.BALP, bins=1000)
    #     # plt.show()

    #     # for i in range(int(self.simurate*self.ALPwind.cohT)):
    #     #     phase = uniform.rvs(loc=0, scale=2*np.pi,size=numofstep)
    #     #     sin_arr = np.sin(phase)
    #     #     cos_arr = np.cos(phase)
    #     #     rw2D.append(np.sum(cos_arr) + 1j*np.sum(sin_arr))
    #     # rw2D = self.ALPwind.BALP*np.array(rw2D, dtype=np.complex64)/np.sqrt(self.numofALPparticle/2.)
    #     rw = (np.sum(cos_arr) + 1j * np.sum(sin_arr)) / np.sqrt(
    #         self.numofALPparticle / 2.0
    #     )

    #     # check(numofsimustep)
    #     self.Bexc_t_vec = np.zeros((numofsimustep, 3))
    #     self.dBexc_dt_vec = np.zeros((numofsimustep, 3))
    #     # loop
    #     if usenumba:
    #         # check(nb.typeof(numofsimustep))
    #         # check(nb.typeof(self.numofALPparticle))
    #         # check(nb.typeof(self.ALP_B))
    #         # check(nb.typeof(ALP_phase))
    #         # check(nb.typeof(rw))
    #         # check(nb.typeof(self.BALP_array))
    #         # check(nb.typeof(self.ALPwind.direction_earth))
    #         # check(nb.typeof(self.ALPwind.BALP))
    #         # check(nb.typeof(self.init_time))
    #         # check(nb.typeof(self.simustep))
    #         # check(nb.typeof(self.station.theta))
    #         # check(nb.typeof(self.station.phi))
    #         # check(nb.typeof(self.nu_rot))

    #         # numofsimustep,  # int
    #         # numofALPparticle,  # int
    #         # random_phase,  # float[:]

    #         # ALP_B,  # float
    #         # ALP_nu_rot,  # float[:]
    #         # ALP_phase,  # float[:]

    #         # ALP_vtova_arr,  # float[:]
    #         # BALP_arr,  # float[:,:]
    #         # dBALPdt_arr,  # float[:,:]

    #         # ALPwind_direction_earth,  # float[:]
    #         # init_time,  # float
    #         # simustep,  # float

    #         # station_theta,  # float
    #         # station_phi,  # float
    #         Simulation.ThermalLightLoop(
    #             numofsimustep=numofsimustep,  # int
    #             numofALPparticle=self.numofALPparticle,  # int
    #             random_phase=random_phase,  # float[:]
    #             ALP_B=self.ALP_B,  # float
    #             ALP_nu_rot=ALP_nu_rot_arr,  # float[:]
    #             ALP_phase=ALP_phase,  # float[:]
    #             ALP_vtova_arr=self.speedtova,  # float[:]
    #             BALP_arr=self.Bexc_t_vec,  # float[:,:]
    #             dBALPdt_arr=self.dBexc_dt_vec,  # float[:,:]
    #             ALPwind_direction_earth=self.excField.direction_earth,  # float[:]
    #             init_time=self.init_time,  # float
    #             simustep=self.timeStep_s,  # float
    #             station_theta=self.station.theta,  # float
    #             station_phi=self.station.phi,  # float
    #         )

    #     else:
    #         for i in range(numofsimustep):
    #             # if verbose and i%10000 == 0:
    #             #     check(i)
    #             # decide BALP amplitude and phase0 from 2D random walk and decoherence due to 2 particle collision in each step
    #             # tic = time.perf_counter()
    #             phase_2i, phase_2ip1 = (
    #                 ALP_phase[(2 * i) % (2 * self.numofALPparticle)],
    #                 ALP_phase[(2 * i + 1) % (2 * self.numofALPparticle)],
    #             )
    #             rw -= (
    #                 np.cos(phase_2i)
    #                 + 1j * np.sin(phase_2i)
    #                 + np.cos(phase_2ip1)
    #                 + 1j * np.sin(phase_2ip1)
    #             ) / np.sqrt(self.numofALPparticle / 2.0)

    #             phase_2i_new, phase_2ip1_new = (
    #                 random_phase[2 * i],
    #                 random_phase[2 * i + 1],
    #             )
    #             rw += (
    #                 np.cos(phase_2i_new)
    #                 + 1j * np.sin(phase_2i_new)
    #                 + np.cos(phase_2ip1_new)
    #                 + 1j * np.sin(phase_2ip1_new)
    #             ) / np.sqrt(2.0 * self.numofALPparticle)
    #             (
    #                 ALP_phase[(2 * i) % (2 * self.numofALPparticle)],
    #                 ALP_phase[(2 * i + 1) % (2 * self.numofALPparticle)],
    #             ) = (phase_2i_new, phase_2ip1_new)

    #             BALP = self.excField.BALP * np.abs(rw)
    #             phase0 = np.angle(rw)
    #             # decide the direciton of B_ALP from the experiment time and motion of celestial bodies
    #             # toc = time.perf_counter()
    #             # newphase_generation += toc-tic
    #             # tic = time.perf_counter()
    #             direction_lab = Npole2station(
    #                 theta_e=self.excField.direction_earth[1],  #
    #                 phi_e=self.excField.direction_earth[2],  #
    #                 theta_s=self.station.theta,  # theta_station
    #                 phi_s=(self.init_time + i * self.timeStep_s) * 2 * np.pi / 86164.0
    #                 + self.station.phi,  # phi_station
    #                 verbose=False,
    #             )
    #             Bx, By, Bz = BALP * direction_lab
    #             self.Bexc_t_vec[i] = [Bx, By, Bz, phase0]
    #             # BALP_modu_list.append(direction_lab)
    #             # toc = time.perf_counter()
    #             # newBALP_generation += toc-tic
    #     # looptime = newphase_generation + newBALP_generation
    #     # if verbose:
    #     #     check(newphase_generation/looptime)
    #     self.timeStamp = np.arange(len(self.Bexc_t_vec)) * self.timeStep_s
    #     check(self.excField.BALP * abs(self.sample.gamma.value_in("Hz/T")))
    #     # check(self.BALP_array[:, 0])
    #     self.BALPsq_arr = (
    #         self.Bexc_t_vec[:, 0] ** 2
    #         + self.Bexc_t_vec[:, 1] ** 2
    #         + self.Bexc_t_vec[:, 2] ** 2
    #     )
    #     check(np.mean(np.sqrt(self.BALPsq_arr)) * abs(self.sample.gamma.value_in("Hz/T")))
    #     check(np.sqrt(np.mean(self.BALPsq_arr)) * abs(self.sample.gamma.value_in("Hz/T")))
    #     if verbose:
    #         fig = plt.figure(figsize=(8 * 0.6, 6 * 0.6), dpi=150)  #
    #         gs = gridspec.GridSpec(nrows=1, ncols=1)  #
    #         # fig.subplots_adjust(left=left_spc, top=top_spc, right=right_spc,
    #         #                     bottom=bottom_spc, wspace=xgrid_spc, hspace=ygrid_spc)
    #         ax = fig.add_subplot(gs[0, 0])
    #         ax.plot(
    #             self.timeStamp,
    #             2 * self.Bexc_t_vec[:, 0] / self.excField.BALP,
    #             color="tab:brown",
    #         )
    #         ax.set_xlabel("time [s]")
    #         ax.set_ylabel("$B_{a,t} / B_{a} $")
    #         plt.tight_layout()
    #         plt.show()
    #     # plt.hist(np.sqrt(self.BALPsq_arr)/self.ALPwind.BALP, bins=30)
    #     # plt.title('BALP_array/ALPwind.BALP')
    #     # plt.show()
    #     if verbose:
    #         fig = plt.figure(figsize=(8 * 0.6, 6 * 0.6), dpi=150)  #
    #         gs = gridspec.GridSpec(nrows=1, ncols=1)  #
    #         # fig.subplots_adjust(left=left_spc, top=top_spc, right=right_spc,
    #         #                     bottom=bottom_spc, wspace=xgrid_spc, hspace=ygrid_spc)
    #         ax = fig.add_subplot(gs[0, 0])
    #         ax.hist(
    #             2 * np.sqrt(self.BALPsq_arr) / self.excField.BALP,
    #             bins=50,
    #             density=True,
    #             color="blue",
    #         )
    #         ax.set_xticks([0, 1, 2, 3, 4])
    #         ax.set_xlabel("$ |B_{a,t} / B_{a}| $")
    #         ax.set_ylabel("Probability density")
    #         plt.tight_layout()
    #         plt.show()

    # def InfCoherence(
    #     self,
    #     numofcohT=None,  # float. number of coherent period for simulation
    #     verbose=False,
    # ):
    #     """
    #     Generate parameters by 'thermal light' method.

    #     Parameters
    #     ----------
    #     numofcohT : float

    #     Number of coherent period for simulation. Can be integer or float values. Default in None.

    #     verbose : bool

    #     """
    #     numofsimustep = int(self.simuRate_Hz * self.excField.cohT * numofcohT)
    #     # initialize an array of particles
    #     # rw2D = []
    #     # numofALPparticle = int(self.simurate*self.ALPwind.cohT)
    #     self.numofALPparticle = int(1234567)
    #     ALP_phase = statUni2Pi(
    #         num=2 * self.numofALPparticle, showplt=False, verbose=False
    #     )
    #     sin_arr = np.sin(ALP_phase)
    #     cos_arr = np.cos(ALP_phase)
    #     rw = (np.sum(cos_arr) + 1j * np.sum(sin_arr)) / np.sqrt(
    #         self.numofALPparticle / 2.0
    #     )
    #     BALP = self.excField.BALP * np.abs(rw)
    #     phase0 = np.angle(rw)
    #     if verbose:
    #         check(numofsimustep)
    #     BALP_list = []
    #     BALP_modu_list = []
    #     # newphase_generation = 0
    #     # newBALP_generation = 0
    #     # loop
    #     for i in range(numofsimustep):
    #         # if verbose and i%10000 == 0:
    #         #     check(i)
    #         # decide the direciton of B_ALP from the experiment time and motion of celestial bodies
    #         # toc = time.perf_counter()
    #         # newphase_generation += toc-tic
    #         # tic = time.perf_counter()
    #         direction_lab = Npole2station(
    #             theta_e=self.excField.direction_earth[1],  #
    #             phi_e=self.excField.direction_earth[2],  #
    #             theta_s=self.station.theta,  # theta_station
    #             phi_s=(self.init_time + i * self.timeStep_s) * 2 * np.pi / 86164.0
    #             + self.station.phi,  # phi_station
    #             verbose=False,
    #         )
    #         Bx, By, Bz = BALP * direction_lab
    #         BALP_list.append([Bx, By, Bz, phase0].copy())
    #         BALP_modu_list.append(direction_lab)
    #         # toc = time.perf_counter()
    #         # newBALP_generation += toc-tic
    #     # looptime = newphase_generation + newBALP_generation
    #     # if verbose:
    #     #     check(newphase_generation/looptime)
    #     self.Bexc_t_vec = np.array(BALP_list)  # [Bx, By, Bz, phase0]
    #     self.BALP_modu_array = np.array(BALP_modu_list) * self.excField.BALP
    #     self.timeStamp = np.arange(len(self.Bexc_t_vec)) * self.timeStep_s

    # def GenerateParam(
    #     self,
    #     numofcohT=None,  # float. number of coherent period for simulation
    #     excType=None,  # 'inf coherence' 'ThermalLight'
    #     showplt=False,  # whether to plot B_ALP
    #     plotrate=None,
    #     verbose=False,
    # ):
    #     """
    #     Generate parameters for simulation.

    #     Parameters
    #     ----------
    #     numofcohT : float

    #     Number of coherent period for simulation. Can be integer or float values. Default in None.

    #     excType : string

    #     The type of excitation for generating simulation parameters.

    #     'RandomJump' - The amplitude, phase, direction of B_1 field will jump to new random values.

    #             The amplitude is sampled from Rayleigh distribution --> see functioncache.statRayleigh()

    #             phase, direction (theta and phi in spherical coordinates) obey U[0, 2 pi), U[0, pi) and U[0, 2 pi).

    #             theta->azimuthal angle, phi->polar angle.

    #             refer to [1] -> 5. Sensitivity scaling with averaging time.

    #     'thermal light' - Gradually change B_1 by the period of 1 coherent time.

    #             The amplitude, phase, direction of B_1 field obey same distributions as in 'RandomJump' method.

    #             However, there's no sudden jump. The decoherent progress happens gradually over time.

    #             Refer to [2] and [3] for more details in this 'thermal light source' like ensemble.

    #     verbose : bool

    #     Reference
    #     ---------
    #     [1] Budker, D., Graham, P. W., Ledbetter, M., Rajendran, S. & Sushkov, A. O.
    #         Proposal for a cosmic axion spin precession experiment (CASPEr). Phys. Rev.
    #         X 4, 021030 (2014).

    #     [2] Loudon, R. The Quantum Theory of Light 2nd edn (Oxford University Press, 1983).

    #     [3] Dmitry Budker and Alexander O. Sushkov,  Physics on your feet: Berkeley graduate exam questions, DOI: 10.1080/00107514.2016.1156750

    #     """
    #     if self.excField.cohT is None:
    #         raise ValueError("self.ALPwind.cohT is None")
    #     if numofcohT is None:
    #         raise TypeError("numofcohT is None")
    #     # if numofcohT < 1:
    #     #     raise ValueError('numofcohT < 1')
    #     self.numofcohT = numofcohT * 1.0
    #     self.method_dict = {
    #         "RandomJump": Simulation.RandomJump,
    #         "ThermalLight": Simulation.ThermalLight,
    #         "InfCoherence": Simulation.InfCoherence,
    #     }
    #     if excType not in self.method_dict.keys():
    #         raise KeyError("method not in self.method_dict.keys()")
    #     else:
    #         self.excType = excType

    #     self.method_dict[excType](
    #         self,
    #         numofcohT=numofcohT,  # float. number of coherent period for simulation
    #         verbose=verbose,
    #     )
    #     if showplt and excType == "InfCoherence":
    #         if plotrate > self.simuRate_Hz:
    #             print(
    #                 "WARNING: plotrate > self.simurate. plotrate will be decreased to simurate"
    #             )
    #             plotrate = self.simuRate_Hz
    #             plotintv = 1
    #         else:
    #             plotintv = int(1.0 * self.simuRate_Hz / plotrate)
    #         # self.BALP_array = np.array(BALP_list)  # [Bx, By, Bz, phase0]
    #         # self.timestamp = np.arange(len(self.BALP_array)+1)*self.simustep

    #         fig = plt.figure(figsize=(4 * 1.0, 3 * 1.0), dpi=150)  #
    #         gs = gridspec.GridSpec(nrows=1, ncols=1)  #
    #         # fig.subplots_adjust( left=left_spc, top=top_spc, right=right_spc,bottom=bottom_spc,
    #         # wspace=0.1, hspace=0.01)
    #         # BALPamp_ax = fig.add_subplot(gs[0,0])
    #         # BALPamp_ax.plot(self.timestamp[0:-1:plotintv], self.BALP_array[0:-1:plotintv, 3], label='$B_{ALP}$', color='tab:red', alpha=0.9)
    #         # BALPamp_ax.legend(loc='upper right')
    #         # BALPamp_ax.set_ylabel('ALP B field / T')
    #         # BALPamp_ax.set_xlabel('time [s]')

    #         BALPxyz_ax = fig.add_subplot(gs[0, 0])
    #         BALPxyz_ax.plot(
    #             self.timeStamp[0:-1:plotintv],
    #             self.BALP_modu_array[0:-1:plotintv, 0],
    #             label="ALP $B_{x}$",
    #             color="tab:blue",
    #             alpha=0.7,
    #         )  # self.BALP_array[0:-1:plotintv, 0]
    #         BALPxyz_ax.plot(
    #             self.timeStamp[0:-1:plotintv],
    #             self.BALP_modu_array[0:-1:plotintv, 1],
    #             label="ALP $B_{y}$",
    #             color="tab:orange",
    #             alpha=0.7,
    #         )
    #         BALPxyz_ax.plot(
    #             self.timeStamp[0:-1:plotintv],
    #             self.BALP_modu_array[0:-1:plotintv, 2],
    #             label="ALP $B_{z}$",
    #             color="tab:green",
    #             alpha=0.7,
    #         )
    #         BALPxyz_ax.set_ylabel("ALP B field / T")
    #         BALPxyz_ax.set_xlabel("Time / hour")
    #         BALPxyz_ax.legend(loc="upper right")
    #         formatter = mticker.FuncFormatter(lambda y, _: f"{y/3600:.0f}")
    #         BALPxyz_ax.xaxis.set_major_formatter(formatter)
    #         plt.tight_layout()
    #         plt.show()
    #     elif showplt and excType != "InfCoherence":
    #         if plotrate > self.simuRate_Hz:
    #             print(
    #                 "WARNING: plotrate > self.simurate. plotrate will be decreased to simurate"
    #             )
    #             plotrate = self.simuRate_Hz
    #             plotintv = 1
    #         else:
    #             plotintv = int(1.0 * self.simuRate_Hz / plotrate)
    #         # self.BALP_array = np.array(BALP_list)  # [Bx, By, Bz, phase0]
    #         # self.timestamp = np.arange(len(self.BALP_array)+1)*self.simustep

    #         fig = plt.figure(figsize=(7 * 1.0, 3 * 1.0), dpi=150)  #
    #         gs = gridspec.GridSpec(nrows=1, ncols=2)  #
    #         # fig.subplots_adjust( left=left_spc, top=top_spc, right=right_spc,bottom=bottom_spc,
    #         # wspace=0.1, hspace=0.01)
    #         # BALPamp_ax = fig.add_subplot(gs[0,0])
    #         # BALPamp_ax.plot(self.timestamp[0:-1:plotintv], self.BALP_array[0:-1:plotintv, 3], label='$B_{ALP}$', color='tab:red', alpha=0.9)
    #         # BALPamp_ax.legend(loc='upper right')
    #         # BALPamp_ax.set_ylabel('ALP B field / T')
    #         # BALPamp_ax.set_xlabel('time [s]')

    #         BALPxyz_ax = fig.add_subplot(gs[0, 0])
    #         BALPxyz_ax.plot(
    #             self.timeStamp[0:-1:plotintv],
    #             self.Bexc_t_vec[0:-1:plotintv, 0],
    #             label="ALP $B_{x}$",
    #             color="tab:blue",
    #             alpha=0.7,
    #         )  # self.BALP_array[0:-1:plotintv, 0]
    #         BALPxyz_ax.plot(
    #             self.timeStamp[0:-1:plotintv],
    #             self.Bexc_t_vec[0:-1:plotintv, 1],
    #             label="ALP $B_{y}$",
    #             color="tab:orange",
    #             alpha=0.7,
    #         )
    #         BALPxyz_ax.plot(
    #             self.timeStamp[0:-1:plotintv],
    #             self.Bexc_t_vec[0:-1:plotintv, 2],
    #             label="ALP $B_{z}$",
    #             color="tab:green",
    #             alpha=0.7,
    #         )
    #         BALPxyz_ax.set_ylabel("ALP B field / T")
    #         BALPxyz_ax.set_xlabel("Time / hour")
    #         BALPxyz_ax.legend(loc="upper right")
    #         formatter_s2h = mticker.FuncFormatter(lambda x, _: f"{x/3600:.1f}")
    #         BALPxyz_ax.xaxis.set_major_formatter(formatter_s2h)

    #         BALPphase0_ax = fig.add_subplot(gs[0, 1])
    #         BALPphase0_ax.plot(
    #             self.timeStamp[0:-1:plotintv],
    #             self.Bexc_t_vec[0:-1:plotintv, 3],
    #             label="ALP $\Phi_{0}$",
    #             color="tab:cyan",
    #             alpha=1.0,
    #         )  # self.BALP_array[0:-1:plotintv, 0]
    #         BALPphase0_ax.set_ylabel("Phase0/$\pi$ ")
    #         BALPphase0_ax.set_xlabel("Time / hour")
    #         BALPphase0_ax.xaxis.set_major_formatter(formatter_s2h)
    #         # formatter_rad2pi = mticker.FuncFormatter(lambda y, _: f'{y/np.pi:.1f}')
    #         # BALPphase0_ax.yaxis.set_major_formatter(formatter_rad2pi)
    #         BALPphase0_ax.set_ylim(-1 * np.pi, 1 * np.pi)
    #         BALPphase0_ax.set_yticks([-1 * np.pi, 0.0, 1 * np.pi])
    #         BALPphase0_ax.set_yticklabels(["-$\pi$", "0", "$\pi$"])
    #         plt.tight_layout()
    #         plt.show()

    # def generatePulseExcitation(
    #     self,
    #     pulseDur: float = 100e-6,
    #     tipAngle: float = np.pi / 2,
    #     direction: np.ndarray = np.array([1, 0, 0]),
    #     nu_rot: float = None,
    #     showplt: bool = False,  # whether to plot B_ALP
    #     plotrate: float = None,
    #     verbose: bool = False,
    # ):
    #     self.excType = "pulse"
    #     B1 = 2 * tipAngle / (self.sample.gamma.value_in("Hz/T") * pulseDur)
    #     duty_func = partial(gate, start=0, stop=pulseDur)
    #     if nu_rot is None:
    #         nu_rot = self.excField.nu - self.demodFreq_Hz
    #     self.excField.setXYPulse(
    #         timeStamp=self.timeStamp,
    #         B1=B1,  # amplitude of the excitation pulse in [T]
    #         nu_rot=nu_rot,  # Hz
    #         init_phase=0,
    #         direction=direction,
    #         duty_func=duty_func,
    #         verbose=False,
    #     )

    #     # check(duty_func(pulseDur / 2))

    # @nb.jit
    @nb.jit(
        [
            "void(float64[:,:], float64[:,:], \
        float64, float64, float64, float64, float64, float64, float64, float64, float64, \
        float64[:,:], float64[:,:], float64[:,:], float64[:,:])"
        ],
        nopython=True,
    )
    def generateTrajectoryLoop(
        B_t,
        dBdt,
        Mx,
        My,
        Mz,
        nu_rot,
        gamma,
        timeStep,
        M0inf,
        T2,
        T1,
        trjry,
        dMdt,
        McrossB,
        d2Mdt2,
    ):
        for i, Bexc in enumerate(B_t):  # self.BALP_array [Bx, By, Bz, phase0]
            [Bx, By, Bz] = Bexc[
                0:3
            ]  # *np.cos(2*np.pi * nu_rot * (i) * timestep + BALP[-1])
            [dBxdt, dBydt, dBzdt] = dBdt[i][0:3]  #
            dMxdt = gamma * (My * Bz - Mz * By) - Mx / T2
            dMydt = gamma * (Mz * Bx - Mx * Bz) - My / T2
            dMzdt = gamma * (Mx * By - My * Bx) - (Mz - M0inf) / T1

            d2Mxdt2 = (
                gamma * (dMydt * Bz + My * dBzdt - dMzdt * By - Mz * dBydt)
                - dMxdt / T2
            )
            d2Mydt2 = (
                gamma * (dMzdt * Bx + Mz * dBxdt - dMxdt * Bz - Mx * dBzdt)
                - dMydt / T2
            )
            d2Mzdt2 = (
                gamma * (dMxdt * By + Mx * dBydt - dMydt * Bx - My * dBxdt)
                - dMzdt / T1
            )

            Mx1 = Mx + dMxdt * timeStep + d2Mxdt2 / 2.0 * timeStep**2
            My1 = My + dMydt * timeStep + d2Mydt2 / 2.0 * timeStep**2
            Mz1 = Mz + dMzdt * timeStep + d2Mzdt2 / 2.0 * timeStep**2

            trjry[i + 1] = [Mx1, My1, Mz1]
            dMdt[i] = [dMxdt, dMydt, dMzdt]
            McrossB[i] = [My * Bz - Mz * By, Mz * Bx - Mx * Bz, Mx * By - My * Bx]
            d2Mdt2[i] = [d2Mxdt2, d2Mydt2, d2Mzdt2]
            [Mx, My, Mz] = [Mx1, My1, Mz1].copy()

    def generateTrajectory(self, usenumba=True, verbose=False):
        """
        Generate trajectory of magnetization vector in Cartesian coordinate system
        based on kinetic simulation for Bloch equations.
        """
        self.trjry = np.zeros((len(self.excField.B_vec) + 1, 3))
        self.dMdt = np.zeros((len(self.excField.B_vec), 3))
        self.McrossB = np.zeros((len(self.excField.B_vec), 3))
        self.d2Mdt2 = np.zeros((len(self.excField.B_vec), 3))
        [Mx, My, Mz] = np.array(
            [
                self.init_M_amp * np.sin(self.init_M_theta) * np.cos(self.init_M_phi),
                self.init_M_amp * np.sin(self.init_M_theta) * np.sin(self.init_M_phi),
                self.init_M_amp * np.cos(self.init_M_theta),
            ]
        )
        vecM0 = np.array([Mx, My, Mz])
        M0inf = np.vdot(vecM0, vecM0) ** 0.5
        #
        self.trjry[0] = vecM0
        #
        timeStep = self.timeStep_s
        gamma = self.sample.gamma.value_in("Hz/T")
        B0z_rot_amp = self.B0z_T - self.demodFreq_Hz / (self.sample.gamma.value_in("Hz/T") / (2 * np.pi))
        B0z_rot = B0z_rot_amp * np.ones(len(self.excField.B_vec))
        B0_rot = np.outer(B0z_rot, np.array([0, 0, 1]))
        if usenumba:
            """
            check(nb.typeof(self.BALP_array))
            check(nb.typeof(Mx))
            check(nb.typeof(My))
            check(nb.typeof(Mz))
            check(nb.typeof(self.nu_rot))
            check(nb.typeof(self.sample.gyroratio))
            check(nb.typeof(self.simustep))
            check(nb.typeof(M0inf))
            check(nb.typeof(self.T2))
            check(nb.typeof(self.T1))
            check(nb.typeof(self.trjry))
            check(nb.typeof(self.dMdt))
            check(nb.typeof(self.McrossB))
            check(nb.typeof(self.d2Mdt2))
            """
            Simulation.generateTrajectoryLoop(
                B_t=self.excField.B_vec + B0_rot,
                dBdt=self.excField.dBdt_vec,
                Mx=Mx,
                My=My,
                Mz=Mz,
                nu_rot=self.nuL_Hz,
                gamma=self.sample.gamma.value_in("Hz/T"),
                timeStep=self.timeStep_s,
                M0inf=M0inf,
                T2=self.T2_s,
                T1=self.T1_s,
                trjry=self.trjry,
                dMdt=self.dMdt,
                McrossB=self.McrossB,
                d2Mdt2=self.d2Mdt2,
            )
        else:
            for i, BALP in enumerate(
                self.excField.B_vec
            ):  # self.BALP_array [Bx, By, Bz, phase0]
                [Bx, By, Bz] = BALP[0:3] * np.cos(
                    2 * np.pi * self.nuL_Hz * (i) * timeStep + BALP[-1]
                )
                [dBxdt, dBydt, dBzdt] = (
                    (-1.0)
                    * BALP[0:3]
                    * 2
                    * np.pi
                    * self.nuL_Hz
                    * np.sin(2 * np.pi * self.nuL_Hz * (i) * timeStep + BALP[-1])
                )
                dMxdt = gamma * (My * Bz - Mz * By) - Mx / self.T2_s
                dMydt = gamma * (Mz * Bx - Mx * Bz) - My / self.T2_s
                dMzdt = gamma * (Mx * By - My * Bx) - (Mz - M0inf) / self.T1_s

                d2Mxdt2 = (
                    gamma * (dMydt * Bz + My * dBzdt - dMzdt * By - Mz * dBydt)
                    - dMxdt / self.T2_s
                )
                d2Mydt2 = (
                    gamma * (dMzdt * Bx + Mz * dBxdt - dMxdt * Bz - Mx * dBzdt)
                    - dMydt / self.T2_s
                )
                d2Mzdt2 = (
                    gamma * (dMxdt * By + Mx * dBydt - dMydt * Bx - My * dBxdt)
                    - dMzdt / self.T1_s
                )

                Mx1 = Mx + dMxdt * timeStep + d2Mxdt2 / 2.0 * timeStep**2
                My1 = My + dMydt * timeStep + d2Mydt2 / 2.0 * timeStep**2
                Mz1 = Mz + dMzdt * timeStep + d2Mzdt2 / 2.0 * timeStep**2

                self.trjry[i + 1] = [Mx1, My1, Mz1]
                self.dMdt[i] = [dMxdt, dMydt, dMzdt]
                self.McrossB[i] = [
                    My * Bz - Mz * By,
                    Mz * Bx - Mx * Bz,
                    Mx * By - My * Bx,
                ]
                self.d2Mdt2[i] = [d2Mxdt2, d2Mydt2, d2Mzdt2]
                # [Mx0, My0, Mz0] = [Mx, My, Mz].copy()
                [Mx, My, Mz] = [Mx1, My1, Mz1].copy()

        # self.trjry = np.array(self.trjry)
        # self.dMdt = np.array(self.dMdt)
        # self.McrossB = np.array(self.McrossB)
        # self.d2Mdt2 = np.array(self.d2Mdt2)

        if verbose:
            check(self.trjry.shape)
        # del M0

    def monitorTrajectory(
        self,
        plotrate: float = None,  #
        verbose: bool = False,
    ):
        if plotrate is None:
            plotrate = self.simuRate_Hz

        if plotrate > self.simuRate_Hz:
            print(
                "WARNING: samprate > self.simurate. samprate will be decreased to simurate"
            )
            plotrate = self.simuRate_Hz
            plotintv = 1
        else:
            plotintv = int(1.0 * self.simuRate_Hz / plotrate)

        self.trjry_visual = self.trjry.copy()

        BALP_array_step = np.concatenate(
            (self.excField.B_vec, [self.excField.B_vec[-1]]), axis=0
        )
        timestamp_step = np.concatenate(
            (
                self.timeStamp,
                [self.timeStamp[-1] + self.timeStamp[-1] - self.timeStamp[-2]],
            ),
            axis=0,
        )
        fig = plt.figure(figsize=(15 * 0.8, 7 * 0.8), dpi=150)  #
        gs = gridspec.GridSpec(nrows=2, ncols=4)  #
        # fix the margins
        left = 0.056
        bottom = 0.1
        right = 0.985
        top = 0.924
        wspace = 0.313
        hspace = 0.127
        fig.subplots_adjust(
            left=left, top=top, right=right, bottom=bottom, wspace=wspace, hspace=hspace
        )

        BALPamp_ax = fig.add_subplot(gs[0, 0])
        Mxy_ax = fig.add_subplot(gs[0, 1], sharex=BALPamp_ax)
        Mz_ax = fig.add_subplot(gs[1, 1], sharex=BALPamp_ax)
        dMxydt_ax = fig.add_subplot(gs[0, 2], sharex=BALPamp_ax)
        dMzdt_ax = fig.add_subplot(gs[1, 2], sharex=BALPamp_ax)
        d2Mxydt_ax = fig.add_subplot(gs[0, 3], sharex=BALPamp_ax)
        d2Mzdt_ax = fig.add_subplot(gs[1, 3], sharex=BALPamp_ax)

        if self.excType == "RandomJump":
            lastnum = -2
        else:
            lastnum = -1
        BALPamp_ax.plot(
            timestamp_step[0:lastnum:plotintv],
            BALP_array_step[0:-1:plotintv, 0],
            label="$B_{x}$",
            color="tab:blue",
            alpha=0.7,
        )  # self.BALP_array[0:-1:plotintv, 0]
        BALPamp_ax.plot(
            timestamp_step[0:lastnum:plotintv],
            BALP_array_step[0:-1:plotintv, 1],
            label="$B_{y}$",
            color="tab:orange",
            alpha=0.7,
        )
        BALPamp_ax.plot(
            timestamp_step[0:lastnum:plotintv],
            BALP_array_step[0:-1:plotintv, 2],
            label="$B_{z}$",
            color="tab:green",
            alpha=0.7,
        )

        BALPamp_ax.set_ylabel("Magnetic field [T]")  # $B_\\mathrm{exc}$
        BALPamp_ax.set_xlabel("time [s]")
        BALPamp_ax.legend(loc="upper right")

        Mtabs = np.sqrt(
            self.trjry[0:-1:plotintv, 0] ** 2 + self.trjry[0:-1:plotintv, 1] ** 2
        )

        Mxy_ax.plot(
            timestamp_step[0:lastnum:plotintv],
            self.trjry[0:-1:plotintv, 0],
            label="$M_x$",
            color="tab:purple",
            alpha=1,
        )
        Mxy_ax.plot(
            timestamp_step[0:lastnum:plotintv],
            self.trjry[0:-1:plotintv, 1],
            label="$M_y$",
            color="tab:brown",
            alpha=1,
        )
        Mxy_ax.plot(
            timestamp_step[0:lastnum:plotintv],
            Mtabs,
            label="$|M_\\mathrm{transverse}|$",
            color="tab:orange",
            alpha=0.7,
            linestyle="--",
        )

        Mxy_ax.legend(loc="upper right")
        # Mxy_ax.set_xlabel("time [s]")
        Mxy_ax.set_ylabel("")
        Mxy_ax.grid()

        Mz_ax.plot(
            timestamp_step[0:lastnum:plotintv],
            self.trjry[0:-1:plotintv, 2],
            label="$M_z$",
            color="tab:pink",
        )
        Mz_ax.legend(loc="upper right")
        Mz_ax.grid()
        Mz_ax.set_xlabel("time [s]")
        Mz_ax.set_ylabel("")
        Mz_ax.set_ylim(0, 1.1)

        dMxydt_ax.plot(
            self.timeStamp[0:-1:plotintv],
            self.dMdt[0 : -1 : int(plotintv), 0],
            label="$d M_x / dt$",
            color="tab:gray",
            alpha=0.7,
        )
        dMxydt_ax.plot(
            self.timeStamp[0:-1:plotintv],
            self.dMdt[0 : -1 : int(plotintv), 1],
            label="$d M_y / dt$",
            color="tab:olive",
            alpha=0.7,
        )
        dMxydt_ax.legend(loc="upper right")
        dMxydt_ax.grid()
        # dMxydt_ax.set_xlabel('time [s]')
        dMxydt_ax.set_ylabel("")

        dMzdt_ax.plot(
            self.timeStamp[0:-1:plotintv],
            self.dMdt[0 : -1 : int(plotintv), 2],
            label="$d M_z / dt$",
            color="tab:cyan",
            alpha=1,
        )
        dMzdt_ax.legend(loc="upper right")
        dMzdt_ax.grid()
        dMzdt_ax.set_xlabel("time [s]")
        dMzdt_ax.set_ylabel("")

        d2Mxydt_ax.plot(
            self.timeStamp[0:-1:plotintv],
            self.d2Mdt2[0 : -1 : int(plotintv), 0],
            label="$d^2 M_x /d t^2$",
            color="tab:blue",
            alpha=0.7,
        )
        d2Mxydt_ax.plot(
            self.timeStamp[0:-1:plotintv],
            self.d2Mdt2[0 : -1 : int(plotintv), 1],
            label="$d^2 M_y /d t^2$",
            color="tab:cyan",
            alpha=0.7,
        )
        d2Mxydt_ax.legend(loc="upper right")
        d2Mxydt_ax.grid()
        # McrossBxy_ax.set_xlabel('time [s]')
        d2Mxydt_ax.set_ylabel("")

        d2Mzdt_ax.plot(
            self.timeStamp[0:-1:plotintv],
            self.d2Mdt2[0 : -1 : int(plotintv), 2],
            label="$d^2 M_z /d t^2$",
            color="tab:purple",
            alpha=1,
        )
        d2Mzdt_ax.legend(loc="upper right")
        d2Mzdt_ax.grid()
        d2Mzdt_ax.set_xlabel("time [s]")
        d2Mzdt_ax.set_ylabel("")

        fig.suptitle(f"T2={self.T2_s:.1g}s T1={self.T1_s:.1e}s")
        # gaNN={self.excField.gaNN:.0e} axion_nu={self.excField.nu:.1e}\nXe
        # print(f'TrajectoryMonitoring_gaNN={self.ALPwind.gaNN:.0e}_axion_nu={self.ALPwind.nu:.1e}_Xe_T2={self.T2:.1g}s_T1={self.T1:.1e}s')
        # plt.tight_layout()
        plt.show()

    def visualizeTrajectory3D(
        self,
        plotrate: float,  # [Hz]
        # rotframe=True,
        verbose=False,
    ):
        if plotrate is None:
            plotrate = self.simuRate_Hz

        if plotrate > self.simuRate_Hz:
            print(
                "WARNING: plotrate > self.simurate. plotrate will be decreased to simurate"
            )
            # warnings.warn('plotrate > self.simurate. plotrate will be decreased to simurate', DeprecationWarning)
            plotrate = self.simuRate_Hz
            plotintv = 1
        else:
            plotintv = int(1.0 * self.simuRate_Hz / plotrate)

        # 3D plot for magnetization vector
        fig = plt.figure(figsize=(6, 5), dpi=150)
        gs = gridspec.GridSpec(nrows=1, ncols=1)
        # fig.subplots_adjust(left=left, top=top, right=right,
        #                             bottom=bottom, wspace=wspace, hspace=hspace)
        # threeD_ax:plt.Axes = fig.add_subplot(gs[0, 0], projection="3d")
        threeD_ax: Axes3D = fig.add_subplot(gs[0, 0], projection="3d")

        # verts = []
        # verts.append(list(zip(frequencies, spectrum_arr[i])))
        # print('verts.shape ', len(verts), len(verts[0]))
        # popt_arr = np.array(popt_arr)
        # confi95_arr = np.array(confi95_arr)
        # print('popt_arr.shape ', popt_arr.shape)
        # print('confi95_arr.shape ', popt_arr.shape)
        # poly = PolyCollection(verts)  # , facecolors=[cc('r'), cc('g'), cc('b'), cc('y')]
        # poly.set_alpha(0.9)
        # threeD_ax.add_collection3d(poly, zs=time_arr, zdir='y')
        # threeD_ax.set_xlabel('absolute frequency / ' + specxunit)
        # threeD_ax.set_xlim3d(np.amin(frequencies), np.amax(frequencies))
        # threeD_ax.set_zlabel('Flux PSD / $10^{-9}$' + specyunit, rotation=180)
        # threeD_ax.set_zlim3d(np.amin(spectrum_arr), np.amax(spectrum_arr))
        threeD_ax.plot(
            xs=self.trjry_visual[0:-1:plotintv, 0],
            ys=self.trjry_visual[0:-1:plotintv, 1],
            zs=self.trjry_visual[0:-1:plotintv, 2],
            zdir="z",
        )

        threeD_ax.zaxis._axinfo["juggled"] = (1, 2, 0)
        # threeD_ax.set_ylabel('Time / min')
        # threeD_ax.set_ylim3d(np.amin(time_arr)-5, np.amax(time_arr)+5)
        threeD_ax.grid(False)
        threeD_ax.w_xaxis.set_pane_color((1, 1, 1, 0.0))
        threeD_ax.w_yaxis.set_pane_color((1, 1, 1, 0.0))
        threeD_ax.w_zaxis.set_pane_color((1, 1, 1, 0.0))

        r = self.init_M_amp
        u, v = np.mgrid[0 : 2 * np.pi : 30j, 0 : np.pi : 20j]
        x = np.cos(u) * np.sin(v)
        y = np.sin(u) * np.sin(v)
        z = np.cos(v)
        threeD_ax.plot_surface(x, y, z, cmap=plt.cm.YlGnBu_r, alpha=0.4)
        XYZlim = [-1.2, 1.2]
        threeD_ax.set_xlim3d(XYZlim)
        threeD_ax.set_ylim3d(XYZlim)
        threeD_ax.set_zlim3d(XYZlim)
        try:
            threeD_ax.set_aspect("equal")
        except NotImplementedError:
            pass
        threeD_ax.set_box_aspect((1, 1, 1))

        threeD_ax.xaxis.set_label_text("x")  #
        threeD_ax.yaxis.set_label_text("y")  #
        threeD_ax.zaxis.set_label_text("z")  #

        fig.suptitle(f"T2={self.T2_s:.1g}s T1={self.T1_s:.1e}s")
        # gaNN={self.excField.gaNN:.0e} axion_nu={self.excField.nu:.1e}\nXe
        # print(
        #     f"TrajectoryVisualization_gaNN={self.excField.gaNN:.0e}_axion_nu={self.excField.nu:.1e}_Xe_T2={self.T2:.1g}s_T1={self.T1:.1e}s"
        # )
        plt.tight_layout()
        plt.show()

    def statTrajectory(self, verbose=False):
        timestep = 1.0 / self.simuRate_Hz
        # xs=self.trjry_visual[0:-1:int(plotintv),0][0:plotlim], \
        # ys=self.trjry_visual[0:-1:int(plotintv),1][0:plotlim], \
        # zs=self.trjry_visual[0:-1:int(plotintv),0][0:plotlim]
        self.avgMxsq = np.mean(self.trjry[:, 0] ** 2, dtype=np.float64)
        self.avgMysq = np.mean(self.trjry[:, 1] ** 2, dtype=np.float64)
        self.avgMzsq = np.mean(self.trjry[:, 2] ** 2, dtype=np.float64)
        if verbose:
            check(self.avgMxsq)
            check(self.avgMysq)
            check(self.avgMzsq)
            check(np.sqrt(self.avgMxsq + self.avgMysq))

    def saveToH5(self, h5fpathandname=None, saveintv=1, verbose=False):  # int
        """ """
        # self.name = name
        # self.sample = sample
        # self.gyroratio=sample.gyroratio
        # self.station = station
        # self.init_time = init_time
        # self.init_magamp=init_magamp
        # self.init_magtheta = init_magtheta
        # self.init_magphi = init_magphi
        # self.init_phase = init_phase
        # self.B0z = B0z
        # self.simurate = simurate
        # self.simustep = 1.0/self.simurate
        # self.ALPwind = ALPwind
        if h5fpathandname[-3:] != ".h5":
            suffix = ".h5"
        else:
            suffix = ""
        h5f = h5py.File(h5fpathandname + suffix, "w")
        h5demod0 = h5f.create_group("NMRKineticSimu/demods/0")
        h5demod0.create_dataset(
            "demodfreq",
            data=np.array([abs(self.sample.gamma.value_in("Hz/T") * self.B0z_T / (2 * np.pi))]),
        )
        h5demod0.create_dataset(
            "samprate", data=np.array([self.simuRate_Hz / (1.0 * saveintv)])
        )
        h5demod0.create_dataset("filter_order", data=np.array([0], dtype=np.int64))
        h5demod0.create_dataset("filter_TC", data=np.array([0.0]))
        h5demod0.create_dataset("timestamp", data=np.array([0]))
        h5demod0.create_dataset("auxin0", data=np.array([0]))
        h5demod0.create_dataset("samplex", data=self.trjry[0:-1:saveintv, 0])
        h5demod0.create_dataset("sampley", data=self.trjry[0:-1:saveintv, 1])

        h5demod1 = h5f.create_group("NMRKineticSimu/demods/1")
        h5demod1.create_dataset("samplez", data=self.trjry[0:-1:saveintv, 2])

        h5axion = h5f.create_group("ALPwind")
        h5axion.create_dataset("name", data=[self.excField.name])
        h5axion.create_dataset("nu", data=[self.excField.nu])
        h5axion.create_dataset("gaNN", data=[self.excField.gaNN])
        h5axion.create_dataset("direction_solar", data=self.excField.direction_solar)
        h5axion.create_dataset("direction_earth", data=self.excField.direction_earth)

        h5station = h5f.create_group("Station")
        h5station.create_dataset("name", data=[self.station.name])
        h5station.create_dataset("latitude", data=[self.station.latitude])
        h5station.create_dataset("longitude", data=[self.station.longitude])
        h5station.create_dataset("NSsemisphere", data=[self.station.NSsemisphere])
        h5station.create_dataset("EWsemisphere", data=[self.station.EWsemisphere])
        h5station.create_dataset("elevation", data=[self.station.elevation])

        h5sample = h5f.create_group("Sample")
        h5sample.create_dataset("name", data=[self.sample.name])
        h5sample.create_dataset("gyroratio", data=[self.sample.gamma.value_in("Hz/T")])
        h5sample.create_dataset("T1", data=np.array([self.T1_s]))
        h5sample.create_dataset("T2", data=np.array([self.T2_s]))
        h5sample.create_dataset("pol", data=[self.sample.pol])
        h5sample.create_dataset("vol", data=[self.sample.vol])
        h5sample.create_dataset("mdm", data=[self.sample.mu])
        h5f.close()

    def analyzeTrajectory(
        self,
    ):
        type(DualChanSig)
        print(DualChanSig)

        self.trjryStream = DualChanSig(
            # name="Simulation data",
            filelist=[],
            verbose=True,
        )
        # self.trjryStream.attenuation = 0
        # self.trjryStream.filterstatus = "off"
        # self.trjryStream.filter_TC = 0.0
        # self.trjryStream.filter_order = 0
        self.trjryStream.demodfreq = self.demodFreq_Hz
        saveintv = 1
        self.trjryStream.samprate = self.simuRate_Hz / saveintv
        self.trjryStream.exptype = "Simulation"

        self.trjryStream.dataX = (
            1 * self.trjry[int(0 * self.simuRate_Hz) : -1 : saveintv, 0]
        )  # * \
        # np.cos(2 * np.pi * self.nu_rot * self.timestamp[0:-1:saveintv])
        self.trjryStream.dataY = (
            1 * self.trjry[int(0 * self.simuRate_Hz) : -1 : saveintv, 1]
        )  # * \
        # np.sin(2 * np.pi * self.nu_rot * self.timestamp[0:-1:saveintv])

        # self.liastream.dataX = 0.5 * 1 * \
        # 	np.cos(2 * np.pi * self.nu_rot * self.timestamp[0:-1:saveintv])
        # self.liastream.dataY = 0.5 * 1 * \
        # 	np.sin(2 * np.pi * self.nu_rot * self.timestamp[0:-1:saveintv])

        self.trjryStream.GetNoPulsePSD(
            windowfunction="rectangle",
            # decayfactor=-10,
            chunksize=None,  # sec
            analysisrange=[0, -1],
            getstd=False,
            stddev_range=None,
            # polycorrparas=[],
            # interestingfreq_list=[],
            selectshots=[],
            verbose=False,
        )
        self.trjryStream.FitPSD(
            fitfunction="Lorentzian",  # 'Lorentzian' 'dualLorentzian' 'tribLorentzian' 'Gaussian 'dualGaussian' 'auto' 'Polyeven'
            inputfitparas=["auto", "auto", "auto", "auto"],
            smooth=False,
            smoothlevel=10,
            fitrange=["auto", "auto"],
            alpha=0.05,
            getresidual=False,
            getchisq=False,
            verbose=False,
        )

    def analyzeB1(
        self,
    ):
        self.B1Stream = DualChanSig(
            name="Simulation data",
            # device="Simulation",
            # device_id="Simulation",
            filelist=[],
            verbose=True,
        )
        self.B1Stream.attenuation = 0
        self.B1Stream.filterstatus = "off"
        self.B1Stream.filter_TC = 0.0
        self.B1Stream.filter_order = 0
        self.B1Stream.demodfreq = self.demodFreq_Hz
        saveintv = 1
        self.B1Stream.samprate = self.simuRate_Hz / saveintv
        # check(self.timestamp.shape)
        # check(self.trjry[0:-1:saveintv, 0].shape)

        self.B1Stream.dataX = (
            1 * self.excField.B_vec[int(0 * self.simuRate_Hz) : -1 : saveintv, 0]
        )  # * \
        # np.cos(2 * np.pi * self.nu_rot * self.timestamp[0:-1:saveintv])
        self.B1Stream.dataY = (
            1 * self.excField.B_vec[int(0 * self.simuRate_Hz) : -1 : saveintv, 1]
        )  # * \
        # np.sin(2 * np.pi * self.nu_rot * self.timestamp[0:-1:saveintv])

        # self.B1stream.dataX = 0.5 * 1 * \
        # 	np.cos(2 * np.pi * self.nu_rot * self.timestamp[0:-1:saveintv])
        # self.B1stream.dataY = 0.5 * 1 * \
        # 	np.sin(2 * np.pi * self.nu_rot * self.timestamp[0:-1:saveintv])

        self.B1Stream.GetNoPulsePSD(
            windowfunction="rectangle",
            # decayfactor=-10,
            chunksize=None,  # sec
            analysisrange=[0, -1],
            getstd=False,
            stddev_range=None,
            # polycorrparas=[],
            # interestingfreq_list=[],
            selectshots=[],
            verbose=False,
        )
        self.B1Stream.FitPSD(
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

    def compareBandSig(self):
        self.analyzeTrajectory()

        specxaxis, ALP_signal_spec, specxunit, specyunit = self.trjryStream.GetSpectrum(
            showtimedomain=True,
            showfit=True,
            showresidual=False,
            showlegend=True,  # !!!!!show or not to show legend
            spectype="PSD",  # in 'PSD', 'ASD', 'FLuxPSD', 'FluxASD'
            ampunit="V",
            specxunit="Hz",  # 'Hz' 'kHz' 'MHz' 'GHz' 'ppm' 'ppb'
            # specxlim=[self.demodfreq - 0 , self.demodfreq + 20],
            # specylim=[0, 4e-23],
            specyscale="linear",  # 'log', 'linear'
            showstd=False,
            showplt_opt=False,
            return_opt=True,
        )

        specxaxis, BALP_spec, specxunit, specyunit = self.excField.plotField(
            demodfreq=self.demodFreq_Hz, samprate=self.simuRate_Hz, showplt_opt=False
        )

        fig = plt.figure(figsize=(6.0, 4.0), dpi=150)  # initialize a figure
        gs = gridspec.GridSpec(nrows=1, ncols=1)  # create grid for multiple figures

        ax00 = fig.add_subplot(gs[0, 0])
        ax00.plot(
            specxaxis,
            BALP_spec / np.amax(BALP_spec),
            label="BALP_spec",
            linestyle="-",
            zorder=1,
        )
        ax00.plot(
            specxaxis,
            ALP_signal_spec / np.amax(ALP_signal_spec),  #
            label="ALP_signal_spec",
            linestyle="--",
        )
        ax00.plot(
            specxaxis,
            self.trjryStream.fitcurves[0] / np.amax(self.trjryStream.fitcurves[0]),
            label=self.trjryStream.fitreport,
            linestyle="--",
        )
        check(self.trjryStream.popt[1])
        check(self.trjryStream.popt[2])
        # print('fit linewidth = ', self.trjryStream.popt[1])
        ax00.set_xlabel("frequency" + specxunit)
        ax00.set_ylabel("PSD")
        # ax00.set_xscale('log')
        # ax00.set_yscale('log')
        ax00.legend()
        ax00.set_xlim(self.demodFreq_Hz - 10, self.demodFreq_Hz + 10)
        # #############################################################################
        fig.suptitle("", wrap=True)
        # #############################################################################
        # # put figure index
        # letters = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)", "(g)", "(h)", "(i)"]
        # for i, ax in enumerate([ax00, ax10]):
        #     xleft, xright = ax00.get_xlim()
        #     ybottom, ytop = ax00.get_ylim()
        #     ax.text(x=xleft, y=ytop, s=letters[i], ha="right", va="bottom", color="blue")
        # # ha = 'left' or 'right'
        # # va = 'top' or 'bottom'
        # #############################################################################
        # # put a mark of script information on the figure
        # # Get the script name and path automatically
        # script_path = os.path.abspath(__file__)
        # # Add the annotation to the figure
        # plt.annotate(
        #     f"Generated by: {script_path}",
        #     xy=(0.02, 0.02),
        #     xycoords="figure fraction",
        #     fontsize=3,
        #     color="gray",
        # )
        # #############################################################################
        plt.tight_layout()
        # plt.savefig('example figure - one-column.png', transparent=False)
        plt.show()
