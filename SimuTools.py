# from turtle import color
# import warnings
# from turtle import color
from cycler import V
from functioncache import *
import numpy as np

# import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# from matplotlib.collections import PolyCollection
import matplotlib.ticker as mticker

# from numpy.core.fromnumeric import shape, transpose

import numba as nb
from math import sin, cos, sqrt
from scipy.stats import maxwell, rayleigh, uniform, norm, chi2, gamma, expon

import h5py

from DataAnalysis import LIASignal


def TTL(x: float | np.ndarray, start: float, stop: float) -> float:
    """
    Returns 1 if start <= x <= stop, else returns 0.

    Parameters:
    x : float or array-like
        The input value(s) where the function is evaluated.

    Returns:
    float or array-like
        1 if start <= x <= stop, else 0.
    """
    # if (start <= x) & (x <= stop):
    #     return 1.0
    # else:
    #     return 0
    return np.where((start <= x) & (x <= stop), 1.0, 0.0)


class Sample:
    def __init__(
        self,
        name=None,  # name of the atom/molecule
        gyroratio=None,  # [Hz/T]. Remember input it like 2 * np.pi * 11.777*10**6
        numofnuclei=None,  #
        tempunit="K",  # temperature scale
        boilpt=None,  # [K]
        meltpt=None,  # [K]
        density_liquid=None,  # [g/cm^3]
        density_gas=None,  # [g/cm^3] at STP
        density_solid=None,  # [g/cm^3]
        molarmass=None,  # [g/mol]
        spindenisty_liquid=None,  # [mol/cm^3]
        spindenisty_gas=None,  # [g/cm^3] at STP
        spindenisty_solid=None,  # [mol/cm^3]
        shareofpeaks=None,  # array or list.
        T2=None,  # [s]
        T1=None,  # [s]
        pol=np.NaN,
        vol=np.NaN,
        mdm=np.NaN,
        verbose=False,
    ):
        """
        create a sample with certain properities.

        numofnuclei
        boilpt
        meltpt
        freezpt
        density_liquid
        density_gas
        density_solid
        molarmass
        spindenisty_liquid
        spindenisty_gas
        spindenisty_solid
        shareofpeaks



        Wikipedia: Standard temperature and pressure
        https://en.wikipedia.org/wiki/Standard_temperature_and_pressure
        In chemistry, IUPAC changed its definition of standard temperature and pressure in 1982:[1][2]

        Until 1982, STP was defined as a temperature of 273.15 K (0 °C, 32 °F) and an absolute pressure
        of exactly 1 atm (101.325 kPa).
        Since 1982, STP has been defined as a temperature of 273.15 K (0 °C, 32 °F) and an absolute
        pressure of exactly 105 Pa (100 kPa, 1 bar).
        STP should not be confused with the standard state commonly used in thermodynamic evaluations
        of the Gibbs energy of a reaction.

        NIST uses a temperature of 20 °C (293.15 K, 68 °F) and an absolute pressure of 1 atm
        (14.696 psi, 101.325 kPa).[3] This standard is also called normal temperature and pressure
        (abbreviated as NTP). However, a common temperature and pressure in use by NIST for
        thermodynamic experiments is 298.15 K (25°C, 77°F) and 1 bar (14.5038 psi, 100 kPa).[4][5] NIST
        also uses "15 °C (60 °F)" for the temperature compensation of refined petroleum products,
        despite noting that these two values are not exactly consistent with each other.[6]
        """
        self.name = name
        self.gyroratio = gyroratio
        zerocelsius = 273.15  # [K]
        self.numofnuclei = numofnuclei
        if tempunit in ["K", "k", "Kelvin", "kelvin"]:
            self.boilpt = boilpt
            self.meltpt = meltpt
        elif tempunit in ["C", "c", "Celsius", "celsius"]:
            print("WARNING: Converting to Kelvin temperature scale...")
            if boilpt is not None:
                self.boilpt = boilpt + zerocelsius
            else:
                self.boilpt = boilpt
            if meltpt is not None:
                self.meltpt = meltpt + zerocelsius
            else:
                self.meltpt = meltpt

        self.density_liquid = density_liquid
        self.density_gas = density_gas
        self.density_solid = density_solid
        self.molarmass = molarmass
        self.spindenisty_liquid = spindenisty_liquid
        self.spindenisty_gas = spindenisty_gas
        self.spindenisty_solid = spindenisty_solid
        # if shareofpeaks is None:
        #     self.shareofpeaks = shareofpeaks
        # elif type(shareofpeaks) is np.ndarray:
        #     if type(shareofpeaks[0]) is np.float32 or type(shareofpeaks[0]) is np.float64:
        #         self.shareofpeaks = shareofpeaks
        #     elif type(shareofpeaks[0]) is np.int32 or type(shareofpeaks[0]) is np.int64:
        #         self.shareofpeaks = shareofpeaks.astype(np.float64)
        #     else:
        #         raise TypeError('shareofpeaks[0] is not float nor int. ')
        # elif shareofpeaks is list:
        #     if type(shareofpeaks[0]) is float or type(shareofpeaks[0]) is int:
        #         self.shareofpeaks = np.array(shareofpeaks,dtype=np.float64)
        #     else:
        #         raise TypeError('shareofpeaks[0] is not float nor int. ')

        if self.molarmass is not None:
            if self.spindenisty_liquid is None and self.density_liquid is not None:
                self.spindenisty_liquid = (
                    self.numofnuclei * self.density_liquid / self.molarmass
                )
            elif self.spindenisty_liquid is not None and self.density_liquid is None:
                self.density_liquid = (
                    self.spindenisty_liquid * self.molarmass / self.numofnuclei
                )

            if self.spindenisty_gas is None and self.density_gas is not None:
                self.spindenisty_gas = (
                    self.numofnuclei * self.density_gas / self.molarmass
                )
            elif self.spindenisty_gas is not None and self.density_gas is None:
                self.density_gas = (
                    self.spindenisty_gas * self.molarmass / self.numofnuclei
                )

            if self.spindenisty_solid is None and self.density_solid is not None:
                self.spindenisty_solid = (
                    self.numofnuclei * self.density_solid / self.molarmass
                )
            elif self.spindenisty_solid is not None and self.density_solid is None:
                self.density_solid = (
                    self.spindenisty_solid * self.molarmass / self.numofnuclei
                )

        self.T2 = T2  # [s]
        self.T1 = T1  # [s]
        self.pol = pol
        self.vol = vol
        self.mdm = mdm


# dictofsamples = {'Methanol':[4, 0.792, 32.04, 3/4],
# 'Ethanol':[6, 0.78945, 46.069, 1/2],
# '129Xe':[.26401, 2.942, 131.2930, 1],
# '131Xe':[.21232, 2.942, 131.2930, 1],
# 'TMS':[12, 0.648, 88.225, 12/12]}

liquid_Xe129 = Sample(
    name="Liquid Xe-129",  # name of the atom/molecule
    gyroratio=2
    * np.pi
    * (-11.777)
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
    T2=None,  # [s]
    T1=1000,  # [s]
    pol=0.5,
    verbose=False,
)
TestSample10MHzT = Sample(
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
    T2=None,  # [s]
    T1=1000,  # [s]
    pol=1,
    verbose=False,
)
Methanol = Sample(
    name="12C Methanol",  # name of the atom/molecule
    gyroratio=2
    * np.pi
    * 42.577478518
    * 10**6,  # [Hz/T]. Remember input it like 2 * np.pi * 11.777*10**6
    numofnuclei=4,  #
    tempunit="K",  # temperature scale
    boilpt=337.8,  # [K]
    meltpt=175.6,  # [K]
    density_liquid=0.792,  # [g/cm^3] at boiling point
    density_gas=None,  # [g/cm^3] at STP
    density_solid=None,  # [g/cm^3]
    molarmass=32.04,  # [g/mol]
    spindenisty_liquid=None,  # [mol/cm^3]
    spindenisty_gas=None,  # [g/cm^3] at STP
    spindenisty_solid=None,  # [mol/cm^3]
    shareofpeaks=[3.0 / 4, 1.0 / 4],  # array or list.
    pol=1.76876e-7,
    verbose=False,
)
Ethanol = Sample(
    name="Ethanol",  # name of the atom/molecule
    gyroratio=2
    * np.pi
    * 42.577478518
    * 10**6,  # [Hz/T]. Remember input it like 2 * np.pi * 11.777*10**6
    numofnuclei=6,  #
    tempunit="K",  # temperature scale
    boilpt=351.38,  # [K]
    meltpt=159.01,  # [K]
    density_liquid=0.78945,  # [g/cm^3] at boiling point
    density_gas=None,  # [g/cm^3] at STP
    density_solid=None,  # [g/cm^3]
    molarmass=46.069,  # [g/mol]
    spindenisty_liquid=None,  # [mol/cm^3]
    spindenisty_gas=None,  # [g/cm^3] at STP
    spindenisty_solid=None,  # [mol/cm^3]
    shareofpeaks=[3 / 6.0, 2.0 / 6, 1.0 / 6],  # array or list.
    T2=None,  # [s]
    T1=None,  # [s]
    pol=1.76876e-7,
    verbose=False,
)
# emppty example
# molecule = Sample(
#         name='',  # name of the atom/molecule
#         gyroratio=2 * np.pi * 42.577478518*10**6,  # [Hz/T]. Remember input it like 2 * np.pi * 11.777*10**6
#         numofnuclei=None,  #
#         tempunit='K',  # temperature scale
#         boilpt=None,  # [K]
#         meltpt=None,  # [K]
#         density_liquid=None,  # [g/cm^3] at boiling point
#         density_gas=None,  # [g/cm^3] at STP
#         density_solid=None,  # [g/cm^3]
#         molarmass=None,  # [g/mol]
#         spindenisty_liquid=None,  # [mol/cm^3]
#         spindenisty_gas=None,  # [g/cm^3] at STP
#         spindenisty_solid=None,  # [mol/cm^3]
#         shareofpeaks=None,  # array or list.
#         T2_liquid=None,  # [s]
#         T2_gas=None,  # [s] at STP
#         T2_solid=None,  # [s]
#         T1_liquid=1000,  # [s]
#         T1_gas=3*24*3600,  # [s] at STP
#         T1_solid=None,  # [s]
#         verbose=False
#         )


class MagField:
    def __init__(
        self,
        name="B field",
    ):
        self.name = name
        self.nu = None

    def setPulse(
        self,
        timeStamp: np.ndarray,
        B1: float,  # amplitude of the excitation pulse in [T]
        nu_rot: float,
        init_phase: float,
        direction: np.ndarray,  #  = np.array([1, 0, 0])
        duty_func,
        verbose: bool = False,
    ):
        """
        generate a pulse in the rotating frame
        """
        Bx_envelope = 1.0 / 2 * B1 * duty_func(timeStamp)
        Bx_envelope = np.multiply(
            Bx_envelope, np.cos(2 * np.pi * nu_rot * timeStamp + init_phase)
        )
        Bx = np.outer(Bx_envelope, np.array([1, 0, 0]))

        By_envelope = 1.0 / 2 * B1 * duty_func(timeStamp)
        By_envelope = np.multiply(
            By_envelope, np.sin(2 * np.pi * nu_rot * timeStamp + init_phase)
        )
        By = np.outer(By_envelope, np.array([0, 1, 0]))

        dBxdt_envelope = 1.0 / 2 * B1 * duty_func(timeStamp)
        dBxdt_envelope = np.multiply(
            dBxdt_envelope,
            -2 * np.pi * nu_rot * np.sin(2 * np.pi * nu_rot * timeStamp + init_phase),
        )
        dBxdt = np.outer(dBxdt_envelope, np.array([1, 0, 0]))

        dBydt_envelope = 1.0 / 2 * B1 * duty_func(timeStamp)
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

    def setALP_Field(
        self,
        method: str,  # 'inverse-FFT' 'time-interfer'
        timeStamp: np.ndarray,
        Brms: float,  # RMS amplitude of the pseudo-magnetic field in [T]
        nu_a: float,  # frequency in the rotating frame
        # direction: np.ndarray,  #  = np.array([1, 0, 0])
        demodfreq:float,
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
                nu_a=nu_a + demodfreq,
                nu=frequencies + demodfreq,
                case="grad_perp",
                alpha=0.0,
            )

            rvs_amp = expon.rvs(loc=0.0, scale=1.0, size=timeLen)
            # rvs_amp = 1.0
            rvs_phase = np.exp(1j * uniform.rvs(loc=0, scale=2 * np.pi, size=timeLen))
            # rvs_phase = 1.0

            ax_sq_lineshape = lineshape * rvs_amp
            # ax_sq_lineshape = lineshape
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

            for i in np.arange(first_positive_index, timeLen, dtype=int):
                nu_rot = frequencies[i]
                ax_amp = ax_lineshape[i]
                # random phase
                init_phase = uniform.rvs(loc=0, scale=2 * np.pi, size=1)
                By_init_phase = uniform.rvs(loc=0, scale=2 * np.pi, size=1)
                # # fixed phase
                # Bx_init_phase = 0
                # By_init_phase = 0

                # Bx
                Bx_amp += (
                    0.5
                    * Brms
                    * ax_amp
                    * np.cos(2 * np.pi * nu_rot * timeStamp + init_phase)
                )
                # By
                By_amp += (
                    0.5
                    * Brms
                    * ax_amp
                    * np.sin(2 * np.pi * nu_rot * timeStamp + init_phase)
                )
                # dBx / dt
                dBxdt_amp += (
                    0.5
                    * Brms
                    * ax_amp
                    * (2 * np.pi * nu_rot)
                    * np.cos(2 * np.pi * nu_rot * timeStamp + init_phase)
                )
                # dBy / dt
                dBydt_amp += (
                    0.5
                    * Brms
                    * ax_amp
                    * (-2 * np.pi * nu_rot)
                    * np.sin(2 * np.pi * nu_rot * timeStamp + init_phase)
                )
            
            Bx = np.outer(Bx_amp, np.array([1, 0, 0]))
            By = np.outer(By_amp, np.array([0, 1, 0]))
            dBxdt = np.outer(dBxdt_amp, np.array([1, 0, 0]))
            dBydt = np.outer(dBydt_amp, np.array([0, 1, 0]))

            self.B_vec = Bx + By
            self.dBdt_vec = dBxdt + dBydt

            if makeplot:
                self.B_Stream = LIASignal(
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
                self.B_Stream.dmodfreq = demodfreq
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
                    specxlim=[nu_a + demodfreq - 5, nu_a + demodfreq + 20],
                    # specylim=[0, 4e-23],
                    specyscale="linear",  # 'log', 'linear'
                    showstd=False,
                    showplt_opt=True,
                    return_opt=True,
                )

        if method == "inverse-FFT":
            pass
        elif method == "time-interfer":
            setALP_Field_timeIntf()
        else:
            raise ValueError("method not found")

    def plotMagField(self, ax: plt.Axes):
        return


class Simulation:
    # NMR simulation in the rotating frame
    def __init__(
        self,
        name="liquid Xe",
        sample: Sample = None,  # class Sample
        station=None,  # refer to class Station
        init_time=0.0,  # [s]. When at time = 0, the ALP wind comes in the direction of (theta, phi) = (90°+23°26′, 0)
        init_mag_amp=1,  # initial magnetization vector amplitude
        init_M_theta=0.0,  # [rad]
        init_M_phi=0.0,  # [rad]
        init_phase=0.0 * np.pi / 180,  # [rad]
        demodfreq: float = 1e6,
        B0z=0.0,  # [T]  B0 in the laboratory frame.
        simuRate=1e6,  # simulation rate in [Hz]. simulation step = 1 / simurate
        duration: float = 1,
        excField=None,  # class AxionWind
        verbose=True,
    ):
        """
        initialize NMR simulation
        """
        self.name = name
        self.sample = sample
        self.gyroratio = sample.gyroratio

        self.station = station

        self.init_time = init_time

        self.init_M_amp = init_mag_amp
        self.init_M_theta = init_M_theta
        self.init_M_phi = init_M_phi

        self.init_phase = init_phase
        self.B0z = B0z
        self.simuRate = simuRate
        self.timeStep = (
            1.0 / self.simuRate
        )  # the key parameter in setting simulation timing
        self.duration = duration
        self.timeStamp = np.arange(int(self.duration * self.simuRate)) * self.timeStep
        # self.timestamp_t
        self.numOfSimuSteps: int = len(self.timeStamp) - 1

        self.excField: MagField = excField
        # self.simutimestep = self.simustep * self.ALPwind.cohT
        # self.simutimerate = 1.0 / self.simutimestep
        self.demodfreq = demodfreq
        self.nu_rot = (
            abs(self.sample.gyroratio * self.B0z / (2 * np.pi)) - self.demodfreq
        )  # frequency difference between the excitation field and the Larmor frequency
        # nu_rot is the Larmor frequency of the magnetization in the rotating frame

        # check(self.nu_rot)
        # check(abs(self.sample.gyroratio*self.B0z/(2*np.pi)))
        # check(abs(self.ALPwind.nu))
        self.T2 = 1.0 * sample.T2
        self.T1 = 1.0 * sample.T1

        if self.simuRate < 10 * abs(self.nu_rot):
            print("WARNING: self.simurate < 10*self.nu_rot")
            check(self.simuRate)
            check(self.nu_rot)

        # move it somewhere later
        # if self.simuRate < 10 * abs(
        #     self.sample.gyroratio * self.excField.BALP / (2 * np.pi)
        # ):
        #     print(
        #         "WARNING: self.simurate < 10 * abs(self.gyroratio*self.BALP/(2*np.pi))"
        #     )

        # self.va = 300 * 1e3

        if verbose:
            print(f"Larmor frequency: {self.B0z*self.sample.gyroratio/(2*np.pi):e} Hz")
            # print(f"ALP compton frequency: {self.excField.nu:e} Hz")
            print(f"simulation rate: {self.simuRate:e} Hz")

        # if abs(
        #     self.B0z * self.sample.gyroratio / (2 * np.pi) - self.excField.nu
        # ) / self.excField.nu > 20 * 10 ** (-6):
        #     print("WARNING: NMR frequency is > 20 ppm away from excitation frequency")
        #     print(f"Larmor frequency: {self.B0z*self.sample.gyroratio/(2*np.pi):e} Hz")
        #     print(f"ALP compton frequency: {self.excField.nu:e} Hz")
        if self.T2 > self.T1:
            print("WARNING: T2 is larger than T1")
            # warnings.warn("T2 is larger than T1", DeprecationWarning)

    def RandomJump(
        self,
        numofcohT=None,  # float. number of coherent period for simulation
        verbose=False,
    ):
        """
        Generate simulation parameters by 'RandomJump' method.

        Parameters
        ----------
        numofcohT : float

        Number of coherent period for simulation. Can be integer or float values. Default in None.

        verbose : bool

        """
        numofsampling = int(np.ceil(numofcohT))
        check(numofsampling)
        BALPamp_array = statRayleigh(
            sigma=self.excField.BALP,
            num=numofsampling,
            showplt=verbose,
            verbose=verbose,
        )
        theta_array = (
            statUni2Pi(num=numofsampling, showplt=verbose, verbose=verbose) / 2.0
        )
        phi_array = statUni2Pi(num=numofsampling, showplt=verbose, verbose=verbose)
        phase0_array = statUni2Pi(num=numofsampling, showplt=verbose, verbose=verbose)
        BALP_list = []

        BALPamp_array[0] = self.excField.BALP
        check(BALPamp_array)
        theta_array[0] = 90.0 * np.pi / 180.0
        phi_array[0] = 0.0 * np.pi / 180.0
        phase0_array[0] = 0

        numofsimustep_perperiod = int(self.simuRate * self.excField.cohT)
        timestamp = np.linspace(
            start=0,
            stop=numofsampling * self.excField.cohT,
            num=int(np.ceil(self.simuRate * numofsampling * self.excField.cohT)),
        )
        for i in range(numofsampling - 1):
            BALP = 0.5 * BALPamp_array[i]
            theta = theta_array[i]
            phi = phi_array[i]
            Bx = BALP * np.sin(theta) * np.cos(phi)
            By = BALP * np.sin(theta) * np.sin(phi)
            Bz = BALP * np.cos(theta)
            phase0 = phase0_array[i]
            for j in range(numofsimustep_perperiod):
                BALP_list.append([Bx, By, Bz, phase0].copy())  # room for modification?
        for i in [-1]:
            BALP = 0.5 * BALPamp_array[i]
            theta = theta_array[i]
            phi = phi_array[i]
            Bx = (
                BALP * np.sin(theta) * np.cos(phi) * np.cos(phi)
            )  # *np.cos(2*np.pi*self.nu_rot*timestamp)
            By = (
                BALP * np.sin(theta) * np.sin(phi) * np.sin(phi)
            )  # *np.sin(2*np.pi*self.nu_rot*timestamp)
            Bz = BALP * np.cos(theta)
            phase0 = phase0_array[i]
            numofstep_lastperiod = int(
                self.simuRate * self.excField.cohT * (numofcohT - numofsampling + 1)
            )
            check(numofstep_lastperiod)
            for j in range(numofstep_lastperiod):
                BALP_list.append(
                    [Bx, By, Bz, phase0].copy()
                )  # self.BALP_array [Bx, By, Bz, phase0]
        del BALPamp_array, theta_array, phi_array, phase0_array
        self.Bexc_t_vec = np.array(BALP_list)  # self.BALP_array [Bx, By, Bz, phase0]
        self.dBexc_dt_vec = np.zeros(self.Bexc_t_vec.shape)
        # check(self.BALP_array)
        self.timeStamp = np.arange(len(self.Bexc_t_vec) + 1) * self.timeStep

    @nb.jit(
        [
            "int16(int64, int64, float64[:], \
        float64, float64[:], float64[:], \
        float64[:], float64[:,:], float64[:,:],     \
        float64[:], float64, float64, \
        float64, float64)"
        ],
        nopython=True,
    )
    def ThermalLightLoop(
        numofsimustep,  # int
        numofALPparticle,  # int
        random_phase,  # float[:]
        ALP_B,  # float
        ALP_nu_rot,  # float[:]
        ALP_phase,  # float[:]
        ALP_vtova_arr,  # float[:]
        BALP_arr,  # float[:,:]
        dBALPdt_arr,  # float[:,:]
        ALPwind_direction_earth,  # float[:]
        init_time,  # float
        simustep,  # float
        station_theta,  # float
        station_phi,  # float
    ):

        # BALP_list = np.array([])
        for i in range(numofsimustep):
            # if verbose and i%10000 == 0:
            #     check(i)
            # decide BALP amplitude and phase0 from 2D random walk and decoherence due to 2 particle collision in each step
            # update ALP particles' phase array
            (
                ALP_phase[(2 * i) % (2 * numofALPparticle)],
                ALP_phase[(2 * i + 1) % (2 * numofALPparticle)],
            ) = (random_phase[2 * i], random_phase[2 * i + 1])
            # BALP = abs(rw)  # ALPwind_BALP *
            # phase0 = np.angle(rw)
            BALP = np.sum(
                ALP_B
                * ALP_vtova_arr
                * np.sin(
                    2 * np.pi * ALP_nu_rot * (init_time + i * simustep) + ALP_phase
                )
            ) / sqrt(2.0 * numofALPparticle)
            dBALPdt = np.sum(
                ALP_B
                * ALP_vtova_arr
                * 2
                * np.pi
                * ALP_nu_rot
                * np.cos(
                    2 * np.pi * ALP_nu_rot * (init_time + i * simustep) + ALP_phase
                )
            ) / sqrt(2.0 * numofALPparticle)
            # decide the direciton of B_ALP from the experiment time and motion of celestial bodies
            theta_e = ALPwind_direction_earth[1]  #
            phi_e = ALPwind_direction_earth[2]  #
            theta_s = station_theta  # theta_station
            phi_s = (
                init_time + i * simustep
            ) * 2 * 3.141592 / 86164.0 + station_phi  # phi_station

            x = sin(theta_e) * cos(theta_s) * cos(phi_e - phi_s) - cos(theta_e) * sin(
                theta_s
            )
            y = sin(theta_e) * sin(phi_e - phi_s)
            z = sin(theta_e) * sin(theta_s) * cos(phi_e - phi_s) + cos(theta_e) * cos(
                theta_s
            )
            Bx, By, Bz = 0.5 * BALP * np.array([x, y, z])
            dBxdt, dBydt, dBzdt = (
                0.5 * dBALPdt * np.array([x, y, z])
            )  # to be improved!!!!!!
            BALP_arr[i] = [Bx, By, Bz]  # , phase0 to be improved?
            dBALPdt_arr[i] = [dBxdt, dBydt, dBzdt]
            # BALP_modu_list.append(direction_lab)
        # BALP_array = np.array(BALP_list)
        # return BALP_array
        return 0

    def ThermalLight(
        self,
        numofcohT=None,  # float. number of coherent period for simulation
        usenumba=True,
        verbose=False,
    ):
        """
        Generate parameters by 'thermal light' method.

        Parameters
        ----------
        numofcohT : float

        Number of coherent period for simulation. Can be integer or float values. Default in None.

        verbose : bool

        """
        numofsimustep = int(self.simuRate * self.excField.cohT * numofcohT)
        # initialize an array of particles
        # rw2D = []
        # numofALPparticle = int(self.simurate*self.ALPwind.cohT)
        self.numofALPparticle = int(self.simuRate * self.excField.cohT)
        if self.numofALPparticle < 1000:
            print("WARNING: self.numofALPparticle < 1000")
        ALP_phase = statUni2Pi(
            num=2 * self.numofALPparticle, showplt=False, verbose=False
        )
        sin_arr = np.sin(ALP_phase)
        cos_arr = np.cos(ALP_phase)
        random_phase = uniform.rvs(loc=0, scale=2 * np.pi, size=2 * numofsimustep)

        self.ALP_B = (
            self.excField.BALP
        )  # why? * (1 + self.ALPwind.Gamma * np.random.standard_cauchy(size=2 * self.numofALPparticle))
        ALP_nu = self.excField.nu * (1 + (self.ALP_B / self.excField.BALP * 1e-3) ** 2)
        ALP_nu_rot = abs(ALP_nu) - abs(self.sample.gyroratio * self.B0z / (2 * np.pi))
        self.va = 220 * 1e3  # 220 km/s
        self.speedtova = maxwell.rvs(size=2 * self.numofALPparticle)

        plt.rc("font", size=16)
        if verbose:
            fig = plt.figure(figsize=(8 * 0.6, 6 * 0.6), dpi=150)  #
            gs = gridspec.GridSpec(nrows=1, ncols=1)  #
            # fig.subplots_adjust(left=left_spc, top=top_spc, right=right_spc,
            #                     bottom=bottom_spc, wspace=xgrid_spc, hspace=ygrid_spc)
            ax = fig.add_subplot(gs[0, 0])
            ax.hist(self.speedtova, bins=100, density=True, color="tab:green")
            ax.set_xlabel("$|v|/v_0$")
            ax.set_ylabel("Probability density")
            # plt.title('Speed distribution')
            plt.tight_layout()
            plt.show()
            # check(np.amin(self.speedtova))

        self.nu_a_arr = self.excField.nu * (1.0 + (self.speedtova * self.va / 3e8) ** 2)
        # check(np.amin(self.nu_a_arr))
        if verbose:
            fig = plt.figure(figsize=(8 * 0.6, 6 * 0.6), dpi=150)  #
            gs = gridspec.GridSpec(nrows=1, ncols=1)  #
            # fig.subplots_adjust(left=left_spc, top=top_spc, right=right_spc,
            #                     bottom=bottom_spc, wspace=xgrid_spc, hspace=ygrid_spc)
            ax = fig.add_subplot(gs[0, 0])
            ax.hist(
                1e6 * (self.nu_a_arr / self.excField.nu - 1),
                bins=100,
                density=True,
                color="tab:purple",
            )
            ax.set_xlabel("$(\\nu/\\nu_a-1) \\times 10^6$")

            ax.set_ylabel("Probability density")
            ax.set_xlim(-0.5, 10.5)  # [0, 1, 2, 3, 4, 5, 6, 7,8 , 9, 10]
            ax.set_xticks([0, 2, 4, 6, 8, 10])  # [0, 1, 2, 3, 4, 5, 6, 7,8 , 9, 10]
            # plt.title('Speed distribution')
            plt.tight_layout()
            plt.show()

            # plt.hist(1e6 * (self.nu_a_arr/self.ALPwind.nu-1), bins=100, density=True)
            # plt.xlabel('$(\\nu/\\nu_a-1) \\times 10^6$')
            # plt.ylabel('Distribution density')
            # plt.title('Frequency distribution')
            # plt.show()

        ALP_nu_rot_arr = self.nu_a_arr - abs(
            self.sample.gyroratio * self.B0z / (2 * np.pi)
        )
        # plt.hist(ALP_B/self.ALPwind.BALP, bins=1000)
        # plt.show()

        # for i in range(int(self.simurate*self.ALPwind.cohT)):
        #     phase = uniform.rvs(loc=0, scale=2*np.pi,size=numofstep)
        #     sin_arr = np.sin(phase)
        #     cos_arr = np.cos(phase)
        #     rw2D.append(np.sum(cos_arr) + 1j*np.sum(sin_arr))
        # rw2D = self.ALPwind.BALP*np.array(rw2D, dtype=np.complex64)/np.sqrt(self.numofALPparticle/2.)
        rw = (np.sum(cos_arr) + 1j * np.sum(sin_arr)) / np.sqrt(
            self.numofALPparticle / 2.0
        )

        # check(numofsimustep)
        self.Bexc_t_vec = np.zeros((numofsimustep, 3))
        self.dBexc_dt_vec = np.zeros((numofsimustep, 3))
        # loop
        if usenumba:
            # check(nb.typeof(numofsimustep))
            # check(nb.typeof(self.numofALPparticle))
            # check(nb.typeof(self.ALP_B))
            # check(nb.typeof(ALP_phase))
            # check(nb.typeof(rw))
            # check(nb.typeof(self.BALP_array))
            # check(nb.typeof(self.ALPwind.direction_earth))
            # check(nb.typeof(self.ALPwind.BALP))
            # check(nb.typeof(self.init_time))
            # check(nb.typeof(self.simustep))
            # check(nb.typeof(self.station.theta))
            # check(nb.typeof(self.station.phi))
            # check(nb.typeof(self.nu_rot))

            # numofsimustep,  # int
            # numofALPparticle,  # int
            # random_phase,  # float[:]

            # ALP_B,  # float
            # ALP_nu_rot,  # float[:]
            # ALP_phase,  # float[:]

            # ALP_vtova_arr,  # float[:]
            # BALP_arr,  # float[:,:]
            # dBALPdt_arr,  # float[:,:]

            # ALPwind_direction_earth,  # float[:]
            # init_time,  # float
            # simustep,  # float

            # station_theta,  # float
            # station_phi,  # float
            Simulation.ThermalLightLoop(
                numofsimustep=numofsimustep,  # int
                numofALPparticle=self.numofALPparticle,  # int
                random_phase=random_phase,  # float[:]
                ALP_B=self.ALP_B,  # float
                ALP_nu_rot=ALP_nu_rot_arr,  # float[:]
                ALP_phase=ALP_phase,  # float[:]
                ALP_vtova_arr=self.speedtova,  # float[:]
                BALP_arr=self.Bexc_t_vec,  # float[:,:]
                dBALPdt_arr=self.dBexc_dt_vec,  # float[:,:]
                ALPwind_direction_earth=self.excField.direction_earth,  # float[:]
                init_time=self.init_time,  # float
                simustep=self.timeStep,  # float
                station_theta=self.station.theta,  # float
                station_phi=self.station.phi,  # float
            )

        else:
            for i in range(numofsimustep):
                # if verbose and i%10000 == 0:
                #     check(i)
                # decide BALP amplitude and phase0 from 2D random walk and decoherence due to 2 particle collision in each step
                # tic = time.perf_counter()
                phase_2i, phase_2ip1 = (
                    ALP_phase[(2 * i) % (2 * self.numofALPparticle)],
                    ALP_phase[(2 * i + 1) % (2 * self.numofALPparticle)],
                )
                rw -= (
                    np.cos(phase_2i)
                    + 1j * np.sin(phase_2i)
                    + np.cos(phase_2ip1)
                    + 1j * np.sin(phase_2ip1)
                ) / np.sqrt(self.numofALPparticle / 2.0)

                phase_2i_new, phase_2ip1_new = (
                    random_phase[2 * i],
                    random_phase[2 * i + 1],
                )
                rw += (
                    np.cos(phase_2i_new)
                    + 1j * np.sin(phase_2i_new)
                    + np.cos(phase_2ip1_new)
                    + 1j * np.sin(phase_2ip1_new)
                ) / np.sqrt(2.0 * self.numofALPparticle)
                (
                    ALP_phase[(2 * i) % (2 * self.numofALPparticle)],
                    ALP_phase[(2 * i + 1) % (2 * self.numofALPparticle)],
                ) = (phase_2i_new, phase_2ip1_new)

                BALP = self.excField.BALP * np.abs(rw)
                phase0 = np.angle(rw)
                # decide the direciton of B_ALP from the experiment time and motion of celestial bodies
                # toc = time.perf_counter()
                # newphase_generation += toc-tic
                # tic = time.perf_counter()
                direction_lab = Npole2station(
                    theta_e=self.excField.direction_earth[1],  #
                    phi_e=self.excField.direction_earth[2],  #
                    theta_s=self.station.theta,  # theta_station
                    phi_s=(self.init_time + i * self.timeStep) * 2 * np.pi / 86164.0
                    + self.station.phi,  # phi_station
                    verbose=False,
                )
                Bx, By, Bz = BALP * direction_lab
                self.Bexc_t_vec[i] = [Bx, By, Bz, phase0]
                # BALP_modu_list.append(direction_lab)
                # toc = time.perf_counter()
                # newBALP_generation += toc-tic
        # looptime = newphase_generation + newBALP_generation
        # if verbose:
        #     check(newphase_generation/looptime)
        self.timeStamp = np.arange(len(self.Bexc_t_vec)) * self.timeStep
        check(self.excField.BALP * abs(self.sample.gyroratio))
        # check(self.BALP_array[:, 0])
        self.BALPsq_arr = (
            self.Bexc_t_vec[:, 0] ** 2
            + self.Bexc_t_vec[:, 1] ** 2
            + self.Bexc_t_vec[:, 2] ** 2
        )
        check(np.mean(np.sqrt(self.BALPsq_arr)) * abs(self.sample.gyroratio))
        check(np.sqrt(np.mean(self.BALPsq_arr)) * abs(self.sample.gyroratio))
        if verbose:
            fig = plt.figure(figsize=(8 * 0.6, 6 * 0.6), dpi=150)  #
            gs = gridspec.GridSpec(nrows=1, ncols=1)  #
            # fig.subplots_adjust(left=left_spc, top=top_spc, right=right_spc,
            #                     bottom=bottom_spc, wspace=xgrid_spc, hspace=ygrid_spc)
            ax = fig.add_subplot(gs[0, 0])
            ax.plot(
                self.timeStamp,
                2 * self.Bexc_t_vec[:, 0] / self.excField.BALP,
                color="tab:brown",
            )
            ax.set_xlabel("time [s]")
            ax.set_ylabel("$B_{a,t} / B_{a} $")
            plt.tight_layout()
            plt.show()
        # plt.hist(np.sqrt(self.BALPsq_arr)/self.ALPwind.BALP, bins=30)
        # plt.title('BALP_array/ALPwind.BALP')
        # plt.show()
        if verbose:
            fig = plt.figure(figsize=(8 * 0.6, 6 * 0.6), dpi=150)  #
            gs = gridspec.GridSpec(nrows=1, ncols=1)  #
            # fig.subplots_adjust(left=left_spc, top=top_spc, right=right_spc,
            #                     bottom=bottom_spc, wspace=xgrid_spc, hspace=ygrid_spc)
            ax = fig.add_subplot(gs[0, 0])
            ax.hist(
                2 * np.sqrt(self.BALPsq_arr) / self.excField.BALP,
                bins=50,
                density=True,
                color="blue",
            )
            ax.set_xticks([0, 1, 2, 3, 4])
            ax.set_xlabel("$ |B_{a,t} / B_{a}| $")
            ax.set_ylabel("Probability density")
            plt.tight_layout()
            plt.show()

    def InfCoherence(
        self,
        numofcohT=None,  # float. number of coherent period for simulation
        verbose=False,
    ):
        """
        Generate parameters by 'thermal light' method.

        Parameters
        ----------
        numofcohT : float

        Number of coherent period for simulation. Can be integer or float values. Default in None.

        verbose : bool

        """
        numofsimustep = int(self.simuRate * self.excField.cohT * numofcohT)
        # initialize an array of particles
        # rw2D = []
        # numofALPparticle = int(self.simurate*self.ALPwind.cohT)
        self.numofALPparticle = int(1234567)
        ALP_phase = statUni2Pi(
            num=2 * self.numofALPparticle, showplt=False, verbose=False
        )
        sin_arr = np.sin(ALP_phase)
        cos_arr = np.cos(ALP_phase)
        rw = (np.sum(cos_arr) + 1j * np.sum(sin_arr)) / np.sqrt(
            self.numofALPparticle / 2.0
        )
        BALP = self.excField.BALP * np.abs(rw)
        phase0 = np.angle(rw)
        if verbose:
            check(numofsimustep)
        BALP_list = []
        BALP_modu_list = []
        # newphase_generation = 0
        # newBALP_generation = 0
        # loop
        for i in range(numofsimustep):
            # if verbose and i%10000 == 0:
            #     check(i)
            # decide the direciton of B_ALP from the experiment time and motion of celestial bodies
            # toc = time.perf_counter()
            # newphase_generation += toc-tic
            # tic = time.perf_counter()
            direction_lab = Npole2station(
                theta_e=self.excField.direction_earth[1],  #
                phi_e=self.excField.direction_earth[2],  #
                theta_s=self.station.theta,  # theta_station
                phi_s=(self.init_time + i * self.timeStep) * 2 * np.pi / 86164.0
                + self.station.phi,  # phi_station
                verbose=False,
            )
            Bx, By, Bz = BALP * direction_lab
            BALP_list.append([Bx, By, Bz, phase0].copy())
            BALP_modu_list.append(direction_lab)
            # toc = time.perf_counter()
            # newBALP_generation += toc-tic
        # looptime = newphase_generation + newBALP_generation
        # if verbose:
        #     check(newphase_generation/looptime)
        self.Bexc_t_vec = np.array(BALP_list)  # [Bx, By, Bz, phase0]
        self.BALP_modu_array = np.array(BALP_modu_list) * self.excField.BALP
        self.timeStamp = np.arange(len(self.Bexc_t_vec)) * self.timeStep

    def GenerateParam(
        self,
        numofcohT=None,  # float. number of coherent period for simulation
        excType=None,  # 'inf coherence' 'ThermalLight'
        showplt=False,  # whether to plot B_ALP
        plotrate=None,
        verbose=False,
    ):
        """
        Generate parameters for simulation.

        Parameters
        ----------
        numofcohT : float

        Number of coherent period for simulation. Can be integer or float values. Default in None.

        excType : string

        The type of excitation for generating simulation parameters.

        'RandomJump' - The amplitude, phase, direction of B_1 field will jump to new random values.

                The amplitude is sampled from Rayleigh distribution --> see functioncache.statRayleigh()

                phase, direction (theta and phi in spherical coordinates) obey U[0, 2 pi), U[0, pi) and U[0, 2 pi).

                theta->azimuthal angle, phi->polar angle.

                refer to [1] -> 5. Sensitivity scaling with averaging time.

        'thermal light' - Gradually change B_1 by the period of 1 coherent time.

                The amplitude, phase, direction of B_1 field obey same distributions as in 'RandomJump' method.

                However, there's no sudden jump. The decoherent progress happens gradually over time.

                Refer to [2] and [3] for more details in this 'thermal light source' like ensemble.

        verbose : bool

        Reference
        ---------
        [1] Budker, D., Graham, P. W., Ledbetter, M., Rajendran, S. & Sushkov, A. O.
            Proposal for a cosmic axion spin precession experiment (CASPEr). Phys. Rev.
            X 4, 021030 (2014).

        [2] Loudon, R. The Quantum Theory of Light 2nd edn (Oxford University Press, 1983).

        [3] Dmitry Budker and Alexander O. Sushkov,  Physics on your feet: Berkeley graduate exam questions, DOI: 10.1080/00107514.2016.1156750

        """
        if self.excField.cohT is None:
            raise ValueError("self.ALPwind.cohT is None")
        if numofcohT is None:
            raise TypeError("numofcohT is None")
        # if numofcohT < 1:
        #     raise ValueError('numofcohT < 1')
        self.numofcohT = numofcohT * 1.0
        self.method_dict = {
            "RandomJump": Simulation.RandomJump,
            "ThermalLight": Simulation.ThermalLight,
            "InfCoherence": Simulation.InfCoherence,
        }
        if excType not in self.method_dict.keys():
            raise KeyError("method not in self.method_dict.keys()")
        else:
            self.excType = excType

        self.method_dict[excType](
            self,
            numofcohT=numofcohT,  # float. number of coherent period for simulation
            verbose=verbose,
        )
        if showplt and excType == "InfCoherence":
            if plotrate > self.simuRate:
                print(
                    "WARNING: plotrate > self.simurate. plotrate will be decreased to simurate"
                )
                plotrate = self.simuRate
                plotintv = 1
            else:
                plotintv = int(1.0 * self.simuRate / plotrate)
            # self.BALP_array = np.array(BALP_list)  # [Bx, By, Bz, phase0]
            # self.timestamp = np.arange(len(self.BALP_array)+1)*self.simustep

            fig = plt.figure(figsize=(4 * 1.0, 3 * 1.0), dpi=150)  #
            gs = gridspec.GridSpec(nrows=1, ncols=1)  #
            # fig.subplots_adjust( left=left_spc, top=top_spc, right=right_spc,bottom=bottom_spc,
            # wspace=0.1, hspace=0.01)
            # BALPamp_ax = fig.add_subplot(gs[0,0])
            # BALPamp_ax.plot(self.timestamp[0:-1:plotintv], self.BALP_array[0:-1:plotintv, 3], label='$B_{ALP}$', color='tab:red', alpha=0.9)
            # BALPamp_ax.legend(loc='upper right')
            # BALPamp_ax.set_ylabel('ALP B field / T')
            # BALPamp_ax.set_xlabel('time [s]')

            BALPxyz_ax = fig.add_subplot(gs[0, 0])
            BALPxyz_ax.plot(
                self.timeStamp[0:-1:plotintv],
                self.BALP_modu_array[0:-1:plotintv, 0],
                label="ALP $B_{x}$",
                color="tab:blue",
                alpha=0.7,
            )  # self.BALP_array[0:-1:plotintv, 0]
            BALPxyz_ax.plot(
                self.timeStamp[0:-1:plotintv],
                self.BALP_modu_array[0:-1:plotintv, 1],
                label="ALP $B_{y}$",
                color="tab:orange",
                alpha=0.7,
            )
            BALPxyz_ax.plot(
                self.timeStamp[0:-1:plotintv],
                self.BALP_modu_array[0:-1:plotintv, 2],
                label="ALP $B_{z}$",
                color="tab:green",
                alpha=0.7,
            )
            BALPxyz_ax.set_ylabel("ALP B field / T")
            BALPxyz_ax.set_xlabel("Time / hour")
            BALPxyz_ax.legend(loc="upper right")
            formatter = mticker.FuncFormatter(lambda y, _: f"{y/3600:.0f}")
            BALPxyz_ax.xaxis.set_major_formatter(formatter)
            plt.tight_layout()
            plt.show()
        elif showplt and excType != "InfCoherence":
            if plotrate > self.simuRate:
                print(
                    "WARNING: plotrate > self.simurate. plotrate will be decreased to simurate"
                )
                plotrate = self.simuRate
                plotintv = 1
            else:
                plotintv = int(1.0 * self.simuRate / plotrate)
            # self.BALP_array = np.array(BALP_list)  # [Bx, By, Bz, phase0]
            # self.timestamp = np.arange(len(self.BALP_array)+1)*self.simustep

            fig = plt.figure(figsize=(7 * 1.0, 3 * 1.0), dpi=150)  #
            gs = gridspec.GridSpec(nrows=1, ncols=2)  #
            # fig.subplots_adjust( left=left_spc, top=top_spc, right=right_spc,bottom=bottom_spc,
            # wspace=0.1, hspace=0.01)
            # BALPamp_ax = fig.add_subplot(gs[0,0])
            # BALPamp_ax.plot(self.timestamp[0:-1:plotintv], self.BALP_array[0:-1:plotintv, 3], label='$B_{ALP}$', color='tab:red', alpha=0.9)
            # BALPamp_ax.legend(loc='upper right')
            # BALPamp_ax.set_ylabel('ALP B field / T')
            # BALPamp_ax.set_xlabel('time [s]')

            BALPxyz_ax = fig.add_subplot(gs[0, 0])
            BALPxyz_ax.plot(
                self.timeStamp[0:-1:plotintv],
                self.Bexc_t_vec[0:-1:plotintv, 0],
                label="ALP $B_{x}$",
                color="tab:blue",
                alpha=0.7,
            )  # self.BALP_array[0:-1:plotintv, 0]
            BALPxyz_ax.plot(
                self.timeStamp[0:-1:plotintv],
                self.Bexc_t_vec[0:-1:plotintv, 1],
                label="ALP $B_{y}$",
                color="tab:orange",
                alpha=0.7,
            )
            BALPxyz_ax.plot(
                self.timeStamp[0:-1:plotintv],
                self.Bexc_t_vec[0:-1:plotintv, 2],
                label="ALP $B_{z}$",
                color="tab:green",
                alpha=0.7,
            )
            BALPxyz_ax.set_ylabel("ALP B field / T")
            BALPxyz_ax.set_xlabel("Time / hour")
            BALPxyz_ax.legend(loc="upper right")
            formatter_s2h = mticker.FuncFormatter(lambda x, _: f"{x/3600:.1f}")
            BALPxyz_ax.xaxis.set_major_formatter(formatter_s2h)

            BALPphase0_ax = fig.add_subplot(gs[0, 1])
            BALPphase0_ax.plot(
                self.timeStamp[0:-1:plotintv],
                self.Bexc_t_vec[0:-1:plotintv, 3],
                label="ALP $\Phi_{0}$",
                color="tab:cyan",
                alpha=1.0,
            )  # self.BALP_array[0:-1:plotintv, 0]
            BALPphase0_ax.set_ylabel("Phase0/$\pi$ ")
            BALPphase0_ax.set_xlabel("Time / hour")
            BALPphase0_ax.xaxis.set_major_formatter(formatter_s2h)
            # formatter_rad2pi = mticker.FuncFormatter(lambda y, _: f'{y/np.pi:.1f}')
            # BALPphase0_ax.yaxis.set_major_formatter(formatter_rad2pi)
            BALPphase0_ax.set_ylim(-1 * np.pi, 1 * np.pi)
            BALPphase0_ax.set_yticks([-1 * np.pi, 0.0, 1 * np.pi])
            BALPphase0_ax.set_yticklabels(["-$\pi$", "0", "$\pi$"])
            plt.tight_layout()
            plt.show()

    def generatePulseExcitation(
        self,
        pulseDur: float = 100e-6,
        tipAngle: float = np.pi / 2,
        direction: np.ndarray = np.array([1, 0, 0]),
        showplt: bool = False,  # whether to plot B_ALP
        plotrate: float = None,
        verbose: bool = False,
    ):
        self.excType = "pulse"
        B1 = 2 * tipAngle / (self.sample.gyroratio * pulseDur)
        duty_func = partial(TTL, start=0, stop=pulseDur)

        self.excField.setPulse(
            timeStamp=self.timeStamp,
            B1=B1,  # amplitude of the excitation pulse in [T]
            nu_rot=self.excField.nu - self.demodfreq,  # Hz
            init_phase=0,
            direction=direction,
            duty_func=duty_func,
            verbose=False,
        )

        # check(duty_func(pulseDur / 2))

    # @nb.jit
    @nb.jit(
        [
            "void(float64[:,:], float64[:,:], \
        float64, float64, float64, float64, float64, float64, float64, float64, float64, \
        float64[:,:], float64[:,:], float64[:,:], float64[:,:])"
        ],
        nopython=True,
    )
    def GenerateTrajectoryLoop(
        B_t,
        dBdt,
        Mx,
        My,
        Mz,
        nu_rot,
        gyroratio,
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
            [dBxdt, dBydt, dBzdt] = dBdt[i][
                0:3
            ]  # (-1.0)*BALP[0:3] * 2*np.pi * nu_rot * np.sin(2*np.pi * nu_rot * (i) * timestep + BALP[-1])
            # if i == 0 :
            #     check(BALP)
            #     check(BALP[0:3])
            #     check([Bx, By, Bz])
            #     check(self.nu_rot)
            dMxdt = gyroratio * (My * Bz - Mz * By) - Mx / T2
            dMydt = gyroratio * (Mz * Bx - Mx * Bz) - My / T2
            dMzdt = gyroratio * (Mx * By - My * Bx) - (Mz - M0inf) / T1

            d2Mxdt2 = (
                gyroratio * (dMydt * Bz + My * dBzdt - dMzdt * By - Mz * dBydt)
                - dMxdt / T2
            )
            d2Mydt2 = (
                gyroratio * (dMzdt * Bx + Mz * dBxdt - dMxdt * Bz - Mx * dBzdt)
                - dMydt / T2
            )
            d2Mzdt2 = (
                gyroratio * (dMxdt * By + Mx * dBydt - dMydt * Bx - My * dBxdt)
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

    def GenerateTrajectory(self, usenumba=True, verbose=False):
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
        # check(M0inf)
        self.trjry[0] = vecM0
        # check(self.gyroratio*self.B0z/(2*np.pi) - self.ALPwind.nu)
        timeStep = self.timeStep
        gyroratio = self.sample.gyroratio
        B0z_rot_amp = self.B0z - self.demodfreq / (self.sample.gyroratio / (2 * np.pi))
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
            Simulation.GenerateTrajectoryLoop(
                B_t=self.excField.B_vec + B0_rot,
                dBdt=self.excField.dBdt_vec,
                Mx=Mx,
                My=My,
                Mz=Mz,
                nu_rot=self.nu_rot,
                gyroratio=self.sample.gyroratio,
                timeStep=self.timeStep,
                M0inf=M0inf,
                T2=self.T2,
                T1=self.T1,
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
                    2 * np.pi * self.nu_rot * (i) * timeStep + BALP[-1]
                )
                [dBxdt, dBydt, dBzdt] = (
                    (-1.0)
                    * BALP[0:3]
                    * 2
                    * np.pi
                    * self.nu_rot
                    * np.sin(2 * np.pi * self.nu_rot * (i) * timeStep + BALP[-1])
                )
                # if i == 0 :
                #     check(BALP)
                #     check(BALP[0:3])
                #     check([Bx, By, Bz])
                #     check(self.nu_rot)
                dMxdt = gyroratio * (My * Bz - Mz * By) - Mx / self.T2
                dMydt = gyroratio * (Mz * Bx - Mx * Bz) - My / self.T2
                dMzdt = gyroratio * (Mx * By - My * Bx) - (Mz - M0inf) / self.T1

                d2Mxdt2 = (
                    gyroratio * (dMydt * Bz + My * dBzdt - dMzdt * By - Mz * dBydt)
                    - dMxdt / self.T2
                )
                d2Mydt2 = (
                    gyroratio * (dMzdt * Bx + Mz * dBxdt - dMxdt * Bz - Mx * dBzdt)
                    - dMydt / self.T2
                )
                d2Mzdt2 = (
                    gyroratio * (dMxdt * By + Mx * dBydt - dMydt * Bx - My * dBxdt)
                    - dMzdt / self.T1
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

    def MonitorTrajectory(
        self,
        plotrate=10**3,  # per simulation period
        # rotframe=True,
        verbose=False,
    ):
        if plotrate > self.simuRate:
            print(
                "WARNING: samprate > self.simurate. samprate will be decreased to simurate"
            )
            plotrate = self.simuRate
            plotintv = 1
        else:
            plotintv = int(1.0 * self.simuRate / plotrate)
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
        fig = plt.figure(figsize=(16, 8), dpi=150)  #
        gs = gridspec.GridSpec(nrows=2, ncols=4)  #
        fig.subplots_adjust(  # left=left_spc, top=top_spc, right=right_spc,bottom=bottom_spc,
            wspace=0.1, hspace=0.01
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
        # Mxy_ax.plot(timestamp_step[0:-1:plotintv], 1, \
        #     label='prediction for |Mt|', color='tab:green', alpha=0.7)
        # np.savetxt(f'\\\\desktop-3ge6tor/d/Yu0702/casper-gradient-code/Supplementary/20220430 Scan Strategy/timestamp_step_T2s_{self.T2}.txt', timestamp_step[0:lastnum:plotintv])
        # np.savetxt(f'\\\\desktop-3ge6tor/d/Yu0702/casper-gradient-code/Supplementary/20220430 Scan Strategy/Mtabs_T2s_{self.T2}.txt', Mtabs)

        Mxy_ax.legend(loc="upper right")
        Mxy_ax.set_xlabel("time [s]")
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
        # Mz_ax.set_xlabel('time [s]')
        Mz_ax.set_ylabel("")
        Mz_ax.set_ylim(0,1)

        dMxydt_ax.plot(
            self.timeStamp[0:-1:plotintv],
            self.dMdt[0 : -1 : int(plotintv), 0],
            label="$d M_x / d_t$",
            color="tab:gray",
            alpha=0.7,
        )
        dMxydt_ax.plot(
            self.timeStamp[0:-1:plotintv],
            self.dMdt[0 : -1 : int(plotintv), 1],
            label="$d M_y / d_t$",
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
            label="$d M_z / d_t$",
            color="tab:cyan",
            alpha=1,
        )
        dMzdt_ax.legend(loc="upper right")
        dMzdt_ax.grid()
        dMzdt_ax.set_xlabel("time [s]")
        dMzdt_ax.set_ylabel("")

        # McrossBxy_ax = fig.add_subplot(gs[0,3])
        # McrossBxy_ax.plot(timestamp_step[0:-1:plotintv], self.McrossB[0:-1:int(plotintv),0], label='McrossBx', color='tab:blue',alpha=0.7)
        # McrossBxy_ax.plot(timestamp_step[0:-1:plotintv], self.McrossB[0:-1:int(plotintv),1], label='McrossBy', color='tab:cyan',alpha=0.7)
        # McrossBxy_ax.legend(loc='upper right')
        # McrossBxy_ax.grid()
        # # McrossBxy_ax.set_xlabel('time [s]')
        # McrossBxy_ax.set_ylabel('')
        # McrossBz_ax = fig.add_subplot(gs[1,3])
        # McrossBz_ax.plot(timestamp_step[0:-1:plotintv], self.McrossB[0:-1:int(plotintv),2], label='McrossBz', color='tab:purple',alpha=1)
        # McrossBz_ax.legend(loc='upper right')
        # McrossBz_ax.grid()
        # McrossBz_ax.set_xlabel('time [s]')
        # McrossBz_ax.set_ylabel('')

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

        fig.suptitle(f"T2={self.T2:.1g}s T1={self.T1:.1e}s")
        # gaNN={self.excField.gaNN:.0e} axion_nu={self.excField.nu:.1e}\nXe
        # print(f'TrajectoryMonitoring_gaNN={self.ALPwind.gaNN:.0e}_axion_nu={self.ALPwind.nu:.1e}_Xe_T2={self.T2:.1g}s_T1={self.T1:.1e}s')
        plt.tight_layout()
        plt.show()

    def VisualizeTrajectory3D(
        self,
        plotrate=10**3,  # [Hz]
        # rotframe=True,
        verbose=False,
    ):
        if plotrate > self.simuRate:
            print(
                "WARNING: plotrate > self.simurate. plotrate will be decreased to simurate"
            )
            # warnings.warn('plotrate > self.simurate. plotrate will be decreased to simurate', DeprecationWarning)
            plotrate = self.simuRate
            plotintv = 1
        else:
            plotintv = int(1.0 * self.simuRate / plotrate)

        # 3D plot for magnetization vector
        fig = plt.figure(figsize=(6, 5), dpi=150)
        gs = gridspec.GridSpec(nrows=1, ncols=1)
        # fig.subplots_adjust(left=left, top=top, right=right,
        #                             bottom=bottom, wspace=wspace, hspace=hspace)
        threeD_ax = fig.add_subplot(gs[0, 0], projection="3d")
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
        fig.suptitle(f"T2={self.T2:.1g}s T1={self.T1:.1e}s")
        # gaNN={self.excField.gaNN:.0e} axion_nu={self.excField.nu:.1e}\nXe
        # print(
        #     f"TrajectoryVisualization_gaNN={self.excField.gaNN:.0e}_axion_nu={self.excField.nu:.1e}_Xe_T2={self.T2:.1g}s_T1={self.T1:.1e}s"
        # )
        plt.tight_layout()
        plt.show()

    def StatTrajectory(self, verbose=False):
        timestep = 1.0 / self.simuRate
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

    def SaveTrajectory(self, h5fpathandname=None, saveintv=1, verbose=False):  # int
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
        h5demod0 = h5f.create_group(f"NMRKineticSimu/demods/0")
        h5demod0.create_dataset(
            "dmodfreq",
            data=np.array([abs(self.sample.gyroratio * self.B0z / (2 * np.pi))]),
        )
        h5demod0.create_dataset(
            "samprate", data=np.array([self.simuRate / (1.0 * saveintv)])
        )
        h5demod0.create_dataset("filter_order", data=np.array([0], dtype=np.int64))
        h5demod0.create_dataset("filter_TC", data=np.array([0.0]))
        h5demod0.create_dataset("timestamp", data=np.array([0]))
        h5demod0.create_dataset("auxin0", data=np.array([0]))
        h5demod0.create_dataset("samplex", data=self.trjry[0:-1:saveintv, 0])
        h5demod0.create_dataset("sampley", data=self.trjry[0:-1:saveintv, 1])

        h5demod1 = h5f.create_group(f"NMRKineticSimu/demods/1")
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
        h5sample.create_dataset("gyroratio", data=[self.sample.gyroratio])
        h5sample.create_dataset("T1", data=np.array([self.T1]))
        h5sample.create_dataset("T2", data=np.array([self.T2]))
        h5sample.create_dataset("pol", data=[self.sample.pol])
        h5sample.create_dataset("vol", data=[self.sample.vol])
        h5sample.create_dataset("mdm", data=[self.sample.mdm])
        h5f.close()

    def analyzeTrajectory(
        self,
    ):
        self.trjryStream = LIASignal(
            name="Simulation data",
            device="Simulation",
            device_id="Simulation",
            filelist=[],
            verbose=True,
        )
        self.trjryStream.attenuation = 0
        self.trjryStream.filterstatus = "off"
        self.trjryStream.filter_TC = 0.0
        self.trjryStream.filter_order = 0
        self.trjryStream.dmodfreq = self.demodfreq
        saveintv = 1
        self.trjryStream.samprate = self.simuRate / saveintv
        self.trjryStream.exptype = 'Simulation'
        # check(self.timestamp.shape)
        # check(self.trjry[0:-1:saveintv, 0].shape)

        self.trjryStream.dataX = (
            1 * self.trjry[int(0 * self.simuRate) : -1 : saveintv, 0]
        )  # * \
        # np.cos(2 * np.pi * self.nu_rot * self.timestamp[0:-1:saveintv])
        self.trjryStream.dataY = (
            1 * self.trjry[int(0 * self.simuRate) : -1 : saveintv, 1]
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
        self.B1Stream = LIASignal(
            name="Simulation data",
            device="Simulation",
            device_id="Simulation",
            filelist=[],
            verbose=True,
        )
        self.B1Stream.attenuation = 0
        self.B1Stream.filterstatus = "off"
        self.B1Stream.filter_TC = 0.0
        self.B1Stream.filter_order = 0
        self.B1Stream.dmodfreq = self.demodfreq
        saveintv = 1
        self.B1Stream.samprate = self.simuRate / saveintv
        # check(self.timestamp.shape)
        # check(self.trjry[0:-1:saveintv, 0].shape)

        self.B1Stream.dataX = (
            1 * self.excField.B_vec[int(0 * self.simuRate) : -1 : saveintv, 0]
        )  # * \
        # np.cos(2 * np.pi * self.nu_rot * self.timestamp[0:-1:saveintv])
        self.B1Stream.dataY = (
            1 * self.excField.B_vec[int(0 * self.simuRate) : -1 : saveintv, 1]
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
