import inspect
import re

import numpy as np
from Envelope import *
from math import sin, sqrt, pi
from typing import Callable

from scipy.integrate import quad

import matplotlib.pyplot as plt

from Envelope import PhysicalQuantity


def check(arg):
    """
    Print information of input arg

    Example
    ------
    import numpy as np

    a = np.zeros((2, 4))

    check(a)

    a+=1

    check(a)

    check(len(a))

    TERMINAL OUTPUT:

    d:\Yu0702\casper-gradient-code\\testofcheckpoint.py @45 a : ndarray(array([[0., 0., 0., 0.], [0., 0., 0., 0.]])) [shape=(2, 4)]

    d:\Yu0702\casper-gradient-code\\testofcheckpoint.py @47 a : ndarray(array([[1., 1., 1., 1.], [1., 1., 1., 1.]])) [shape=(2, 4)]

    d:\Yu0702\casper-gradient-code\\testofcheckpoint.py @48 len(a) : int(2)

    d:\Yu0702\casper-gradient-code\\testofcheckpoint.py @49 a.shape : tuple((2, 4)) [len=2]


    Copyright info:
    ------
    Adopted from https://gist.github.com/HaleTom/125f0c0b0a1fb4fbf4311e6aa763844b

    Author: Tom Hale

    Original comment: Print the line and filename, function call, the class, str representation and some other info
                    Inspired by https://stackoverflow.com/a/8856387/5353461


    """
    frame = inspect.currentframe()
    callerframeinfo = inspect.getframeinfo(frame.f_back)
    try:
        context = inspect.getframeinfo(frame.f_back).code_context
        caller_lines = "".join([line.strip() for line in context])
        m = re.search(r"check\s*\((.+?)\)$", caller_lines)
        if m:
            caller_lines = m.group(1)
            position = (
                str(callerframeinfo.filename) + " line " + str(callerframeinfo.lineno)
            )  # print path
            # position = " line " + str(callerframeinfo.lineno)  # do no print path

            # Add additional info such as array shape or string length
            additional = ""
            if hasattr(arg, "shape"):
                additional += "[shape={}]".format(arg.shape)
            elif hasattr(arg, "__len__"):  # shape includes length information
                additional += "[len={}]".format(len(arg))

            # Use str() representation if it is printable
            str_arg = str(arg)
            str_arg = str_arg if str_arg.isprintable() else repr(arg)

            print(position, "" + caller_lines + " : ", end="")
            print(arg.__class__.__name__ + "(" + str_arg + ")", additional)
        else:
            print("check: couldn't find caller context")
    finally:
        del frame
        del callerframeinfo


def Lorentzian(x, center, FWHM, area: float = 1.0, offset: float = 0.0):
    """
    Return the value of the Lorentzian function
        offset + 0.5*FWHM*area / (np.pi * ( (x-center)**2 + (0.5*FWHM)**2 )      )

                           FWHM A
        offset + ───────────────────────
                  2π ((x-c)^2+(FWHM/2)^2 )

    Parameters
    ----------

    x : scalar or array_like
        argument of the Lorentzian function
    center : scalar
        the position of the Lorentzian peak
    FWHM : scalar
        full width of half maximum (FWHM) / linewidth of the Lorentzian peak
    area : scalar
        area under the Lorentzian curve (without taking offset into consideration)
    offset : scalar
        offset for the curve


    Returns
    -------
    the value of the Lorentzian function : ndarray or scalar

    Examples
    --------
    >>>

    Reference
    ----------
    Null

    """
    return offset + 0.5 * abs(FWHM) * area / (
        np.pi * ((x - center) ** 2 + (0.5 * FWHM) ** 2)
    )


def get_FWHMin(x, y):
    """
    Calculate the Full Width at Half Maximum (FWHM) of a dip.

    Parameters:
        x (array-like): The x-values of the curve.
        y (array-like): The y-values of the curve.

    Returns:
        float: The FWHM of the curve.
    """
    # Ensure inputs are numpy arrays
    x = np.array(x)
    y = np.array(y)

    # Find the maximum value of y and its half-maximum
    y_min = np.amin(y)
    Twice_min = y_min * 2.0

    # Find indices where y crosses the half-maximum
    # check(np.where(y <= Twice_min))
    indices = np.where(y <= Twice_min)[0]
    if len(indices) < 2:
        raise ValueError(
            "Cannot calculate FWHM: The curve does not have two points crossing the half-maximum."
        )

    # Extract the first and last indices crossing the half-maximum
    left_index = indices[0]
    right_index = indices[-1]

    # # Interpolate to find more precise crossing points
    # x_left = np.interp(Twice_min, [y[left_index - 1], y[left_index]], [x[left_index - 1], x[left_index]])
    # x_right = np.interp(Twice_min, [y[right_index], y[right_index + 1]], [x[right_index], x[right_index + 1]])
    x_left = x[left_index]
    x_right = x[right_index]

    # Calculate FWHM
    FWHMin = abs(x_right - x_left)

    return FWHMin


def freq2mass(
    freq: PhysicalQuantity,
):
    return (h_Planck * freq / c**2).convert_to("eV/c**2")


def mass2freq(
    mass: PhysicalQuantity,
):
    return (mass * c**2 / h_Planck).convert_to("Hz")


def energy2freq(
    energy: PhysicalQuantity,
):
    return (energy / h_Planck).convert_to("Hz")


class LF_2025:
    # ask me to write comment if you need
    # Yuzhe Zhang yuhzhang@uni-mainz.de
    def __init__(
        self,
        name="CASPEr-G-LF new projection",
        B0_max=PhysicalQuantity(0.1, "tesla"),
        rho_E_DM=PhysicalQuantity(0.3, "GeV / cm**3"),
        va=PhysicalQuantity(220, "km / s"),
    ):
        self.name = name
        self.B0_max = B0_max
        self.rho_E_DM = rho_E_DM
        self.rho_M_DM = rho_E_DM / c**2
        self.Q_a = PhysicalQuantity(10**6, "")
        self.nu_lowlimit = PhysicalQuantity(1, "kHz")
        self.va = va

    def sampleMethanol(
        self,
    ):
        # Methanol properites
        rho_M_Methanol = PhysicalQuantity(0.792, "g / cm**3 ")
        molmassMethanol = PhysicalQuantity(32.04, "g / mol")
        rho_N_MethanolProton = (
            PhysicalQuantity(4.0, "") * rho_M_Methanol / molmassMethanol * mol_to_N
        )

        self.sample = "Methanol"

        self.ns = rho_N_MethanolProton.convert_to("1 / cm**3")
        self.ns_SPN = PhysicalQuantity(sqrt(self.ns.value), " 1 / cm**3 ")
        # check(ns)
        # self.T2 = PhysicalQuantity(1, 's')
        self.gamma = gamma_p
        self.nu_uplimit = self.gamma * self.B0_max / (2 * pi)
        self.mu = mu_p

    def sampleLXe129(self, abundance: PhysicalQuantity = PhysicalQuantity(100, "%")):
        # Xe properites
        rho_M_LXe = PhysicalQuantity(3.1, "g / cm**3 ")
        molmassLXe = PhysicalQuantity(131.29, "g / mol")
        rho_N_LXe129 = abundance * 1 * rho_M_LXe / molmassLXe * mol_to_N
        # check(abundance.convert_to('%'))
        # print(rhoN_MethanolProton.convert_to("1 / cm ** 3"))

        self.sample = "LXe129"
        self.ns = rho_N_LXe129.convert_to(" 1 / cm**3 ")
        self.ns_SPN = PhysicalQuantity(sqrt(self.ns.value), " 1 / cm**3 ")
        # check(ns)
        self.gamma = ((gamma_Xe129) ** 2) ** (1 / 2)
        # check(self.gamma)
        self.nu_uplimit = self.gamma * self.B0_max / (2 * pi)
        self.mu = mu_Xe129
        self.T2 = PhysicalQuantity(1000, "s")

    def sampleLXe129_approx(
        self, abundance: PhysicalQuantity = PhysicalQuantity(100, "%")
    ):
        # Xe properites
        # rho_M_LXe = PhysicalQuantity(3.1, "g / cm**3 ")
        # molmassLXe = PhysicalQuantity(131.29, "g / mol")
        # rho_N_LXe129 = abundance * 1 * rho_M_LXe / molmassLXe * mol_to_N
        # check(abundance.convert_to('%'))
        # print(rhoN_MethanolProton.convert_to("1 / cm ** 3"))

        self.sample = "LXe129"
        self.ns = PhysicalQuantity(1e22, " 1 / cm**3 ")
        self.ns_SPN = PhysicalQuantity(sqrt(self.ns.value), " 1 / cm**3 ")
        # check(ns)
        self.gamma = ((gamma_Xe129) ** 2) ** (1 / 2)
        # check(self.gamma)
        self.nu_uplimit = self.gamma * self.B0_max / (2 * pi)
        self.mu = mu_Xe129
        self.T2 = PhysicalQuantity(1000, "s")

    def GetT2(
        self,
        nu: PhysicalQuantity,
    ):
        nu = nu.convert_to("Hz")
        if self.sample == "Methanol":
            return PhysicalQuantity(1, "s")
        elif self.sample == "LXe129":
            return PhysicalQuantity(1000, "s")

    def NMR_lw_Hz(
        self,
        NMR_lw_ppm: PhysicalQuantity,
        nu: PhysicalQuantity,
    ):
        return (nu * NMR_lw_ppm).convert_to("Hz")

    def GetTdelta(
        self,
        NMR_lw_ppm: PhysicalQuantity,
        nu: PhysicalQuantity,
    ):
        return (1 / (pi * self.NMR_lw_Hz(NMR_lw_ppm, nu))).convert_to("s")

    def GetT2star(
        self,
        NMR_lw_ppm: PhysicalQuantity,
        nu: PhysicalQuantity,
    ):
        T2star = (self.GetT2(nu) ** (-1) + self.GetTdelta(NMR_lw_ppm, nu) ** (-1)) ** (
            -1
        )
        return T2star.convert_to("s")

    def getThermalPol(
        self,
        # nu_L:PhysicalQuantity,
        B_pol: PhysicalQuantity,
        Temp: PhysicalQuantity,
    ):
        pol = hbar * self.gamma * B_pol / (2 * k * Temp)
        # pol = hbar * (2 * pi * nu_L) / ( 2 * k * Temp)
        # self.p = p.convert_to('')
        pol = pol.convert_to("")
        # check(pol)
        return pol

    def getM0(
        self,
        pol,
        ns,
    ):
        """
        compute magnetization M0
        """
        M0 = (self.mu * pol * ns).convert_to("A/m")
        # self.M0_SPN = (mu_p * ns_SPN).convert_to("A/m")
        return M0

    def getPhi_pick(
        self,
        M0: PhysicalQuantity,
        gV: PhysicalQuantity = PhysicalQuantity(
            37.0, "1/m"
        ),  # estimated from a  cylindrical sample (R=4 mm, H=22.53 mm) coupling to the gradiometer
        Vol: PhysicalQuantity = PhysicalQuantity(pi * 4.0**2 * 24, "mm**3"),
    ):
        """
        get the flux in the pickup coil (gradiometer)
        """
        Phi_pick = gV * mu_0 * M0 * Vol
        Phi_pick = Phi_pick.convert_to("Phi_0")
        return Phi_pick

    def getM2Pick(
        self,
        gV: PhysicalQuantity = PhysicalQuantity(
            37.0, "1/m"
        ),  # estimated from a  cylindrical sample (R=4 mm, H=22.53 mm) coupling to the gradiometer
        Vol: PhysicalQuantity = PhysicalQuantity(pi * 4.0**2 * 24, "mm**3"),
    ):
        """ """
        return gV

    def getPick2In(
        self,
        # defaults are properties of SQUID C649_G12 for DM measurement 2022.12.14 and 12.23
        Lin=PhysicalQuantity(400, "nH"),
        Lpick=PhysicalQuantity(553, "nH"),
        Min=PhysicalQuantity(1 / 0.5194, "Phi_0/microA"),
    ):
        """
        input coupling
        considering pickup-SQUID coupling
        """
        # check(Mf.convert_to('Phi_0 / mA'))
        # check(Min.convert_to('nH'))

        # coupling between Phi_pick and Phi_in
        pick2in = Min / (Lpick + Lin)
        pick2in = pick2in.convert_to("")
        # check(pick2in)
        return pick2in

    def getIn2Vf(
        self,
        # defaults are properties of SQUID C649_G12 for DM measurement 2022.12.14 and 12.23
        Lin=PhysicalQuantity(400, "nH"),
        Lpick=PhysicalQuantity(553, "nH"),
        Mf=PhysicalQuantity(1 / 43.803, "Phi_0 / microA"),
        Min=PhysicalQuantity(1 / 0.5194, "Phi_0/microA"),
        Rf=PhysicalQuantity(3, "kiloohm"),
    ):
        """
        input coupling
        considering pickup-SQUID coupling
        """
        # check(Mf.convert_to('Phi_0 / mA'))
        # check(Min.convert_to('nH'))

        # coupling between V_f and Phi_in
        in2Vf = Rf / Mf * Min / (Lpick + Lin)
        in2Vf = in2Vf.convert_to("V/Phi_0")

        return in2Vf

    # def useLF_Magnet(
    #     self,
    # ):
    #     self.B0_max = PhysicalQuantity(0.1, "tesla")

    # def useHF_Magnet(
    #     self,
    # ):
    #     self.B0_max = PhysicalQuantity(14.1, "tesla")

    def useShims(
        self,
        NMR_lw_ppm: PhysicalQuantity,
    ):
        self.NMR_lw_ppm = NMR_lw_ppm

    def getOmega_a(self, alpha=PhysicalQuantity(pi / 2, "rad")):
        """
        ALP field Rabi frequency Omega_a at gaNN = self.gaNN_base
        """

        alpha = alpha.convert_to("rad")
        # self.gaNN_base = PhysicalQuantity(1, "GeV**(-1)")
        self.gaNN_base = PhysicalQuantity(1.1e-13, "eV**(-1)")

        Omega_a = (
            1
            / 2.0
            * self.gaNN_base
            * (2.0 * hbar * c * self.rho_E_DM) ** (1 / 2)
            * self.va
            * sin(alpha.value)
        )
        Omega_a = Omega_a.convert_to("Hz")
        # check(gaNN)
        # check(sin(alpha.value))
        # check(Omega_a / (2 * pi))
        return Omega_a

    def getTipAngle(
        self,
        Omega_a: PhysicalQuantity,
        T2: PhysicalQuantity,
        tau_a: PhysicalQuantity,
    ):
        angle = Omega_a * T2 * (tau_a / (T2 + tau_a)) ** 0.5
        angle = angle.convert_to("rad")
        # check(sin(angle.value))
        return angle

    def getSpecFac(
        self,
        nu_L: PhysicalQuantity,
        nu_a: PhysicalQuantity,
        T2: PhysicalQuantity,
        Tdelta: PhysicalQuantity,
        tau_a: PhysicalQuantity,
    ):
        Delta2 = (tau_a) ** (-1)
        Delta3 = (T2) ** (-1) + (Tdelta) ** (-1)
        # u = (
        #     Delta2
        #     / Delta3
        #     * (PhysicalQuantity(1, "") + (nu_L - nu_a) ** 2 / (Delta3 / 2) ** 2) ** (-1)
        # )
        u = Delta2 / Delta3
        u = u.convert_to("")
        return u

    def getPSDnoise_SQUID(self, nu: PhysicalQuantity):
        noise = PhysicalQuantity(1e-12, "Phi_0 ** 2 / Hz")
        return noise

    def getPower_sig(
        self,
        gin: PhysicalQuantity,
        M0: PhysicalQuantity,
        vol: PhysicalQuantity,
        u: PhysicalQuantity,
        tipAngle: PhysicalQuantity,
    ) -> PhysicalQuantity:
        Phi_in_pito2 = gin * mu_0 * M0 * vol
        power_sig = 1.0 / 2 * (u * Phi_in_pito2) ** 2.0 * tipAngle**2
        power_sig = power_sig.convert_to("Phi_0**2")
        return power_sig

    def getPower_SPN(
        self,
        gin: PhysicalQuantity,
        M_SPN: PhysicalQuantity,
        vol: PhysicalQuantity,
    ) -> PhysicalQuantity:
        Phi_in_SPN = gin * mu_0 * M_SPN * vol
        power_SPN = 1.0 / 2 * (Phi_in_SPN) ** 2.0
        power_SPN = power_SPN.convert_to("Phi_0**2")
        return power_SPN.convert_to("Phi_0**2")

    def getPSD_SPN(
        self, power: PhysicalQuantity, Delta: PhysicalQuantity
    ) -> PhysicalQuantity:
        PSD_SPN = power / Delta
        PSD_SPN = PSD_SPN.convert_to("Phi_0**2 / Hz")
        return PSD_SPN

    def getPowerNoise_MF(
        self,
        PSDnoise_SQUID: PhysicalQuantity,
        power_SPN: PhysicalQuantity,
        # nu:PhysicalQuantity,
        ALP_lw_Hz: PhysicalQuantity,
        Tmeas: PhysicalQuantity,
        T2star: PhysicalQuantity,
        verbose: bool = False,
    ) -> PhysicalQuantity:
        """
        Noise in the spectrum after matched filtering
        """
        # assume long measurement time
        assert (Tmeas - 1 / ALP_lw_Hz).convert_to("s").value >= 0
        PSDnoise_SPN = self.getPSD_SPN(power=power_SPN, Delta=1 / (pi * T2star))
        PSDnoise = (PSDnoise_SQUID**2 + PSDnoise_SPN**2) ** 0.5
        Navg = (ALP_lw_Hz * Tmeas).convert_to("")
        RBW = 1 / Tmeas
        if (Tmeas - T2star).value < 0:
            # measurement time is shorter than T2star
            # RBW boarder than
            powerNoise_MF = RBW * (PSDnoise_SQUID**2 + (power_SPN / RBW) ** 2) ** 0.5
        elif Navg.value < 1:
            # measurement time is longer than T2star
            # but shorter than 1 / ALP_lw_Hz
            # matched filtering cannot be applied
            powerNoise_MF = RBW * PSDnoise
        else:
            # measurement time is longer than 1 / ALP_lw_Hz
            # matched filtering can be applied
            powerNoise_MF = PSDnoise * PhysicalQuantity(1, "Hz") / Navg**0.5

        powerNoise_MF = powerNoise_MF.convert_to("Phi_0**2")

        if verbose:
            check((Tmeas - T2star).convert_to("s"))
            check(Navg)
            check(powerNoise_MF)

        return powerNoise_MF

    def getEfficPow(
        self,
        # nu:PhysicalQuantity,
        RBW_Hz: PhysicalQuantity,  # resolution bandwidth
        ALP_lw_Hz: PhysicalQuantity,
        SG_effic=PhysicalQuantity(100, "%"),
    ):
        """
        analysis efficiency for signal power
        """
        assert RBW_Hz.value > 0
        assert RBW_Hz.value > 0
        assert SG_effic.value > 0
        # ALP_lw_Hz = ALP_lw_Hz.convert_to("Hz")
        # RBW_Hz = RBW_Hz.convert_to("Hz")
        # check(ALP_lw_Hz.value)
        # check(RBW_Hz.value)
        effici = PhysicalQuantity(100, "%")
        effici *= SG_effic
        if (RBW_Hz - ALP_lw_Hz).value > 0:
            effici *= PhysicalQuantity(1, "") - 0.5 * ALP_lw_Hz / RBW_Hz

        effici = effici.convert_to("")
        return effici

    def getThermMethanol_Sensi(
        self,
        freq_list: list[PhysicalQuantity],  # frequencies [Hz]
        Tmeas_list: list[PhysicalQuantity],
        verbose: bool = False,
    ) -> list[PhysicalQuantity]:
        """
        gaNN sensitivity with a thermally-polarized methanol sample

        compute SNR for gaNN = 1 GeV^-1, so as to estimate the gaNN limit
        """
        # if np.amin(freq)<1e3 or np.amax(freq) > 4.3

        # get self.sample, self.ns, self.ns_SPN
        self.sampleMethanol()  #

        # set inhomogeneity
        self.useShims(PhysicalQuantity(10, "ppm"))

        # temperature
        temp = PhysicalQuantity(273.15 - 90, "K")

        # sensitivity at nu_a
        def sensi(
            nu_a: PhysicalQuantity,  # frequency [Hz]
            Tmeas: PhysicalQuantity,  # frequencies [Hz]
            # eta:PhysicalQuantity
        ) -> PhysicalQuantity:
            ALP_lw_Hz = nu_a / self.Q_a
            # check(ALP_lw_Hz)
            tau_a = (1 / (pi * ALP_lw_Hz)).convert_to("s")

            nu_a = nu_a.convert_to("Hz")
            self.nu_uplimit = self.nu_uplimit.convert_to("Hz")
            nu_L = nu_a if (nu_a - self.nu_uplimit).value <= 0 else self.nu_uplimit

            T2 = self.GetT2(nu_L)
            # check(T2)
            Tdelta = self.GetTdelta(NMR_lw_ppm=self.NMR_lw_ppm, nu=nu_L)
            # check(Tdelta)
            T2star = ((T2) ** (-1) + (Tdelta) ** (-1)) ** (-1)

            # Phi pi/2

            gin = self.getPick2In() * self.getM2Pick()
            pol = self.getThermalPol(B_pol=2 * pi * nu_L / self.gamma, Temp=temp)
            u = self.getSpecFac(
                nu_L=nu_a, nu_a=nu_a, T2=self.GetT2(nu_L), Tdelta=Tdelta, tau_a=tau_a
            )
            tipAngle = self.getTipAngle(Omega_a=self.getOmega_a(), T2=T2, tau_a=tau_a)
            vol = PhysicalQuantity(1.2, "cm**3")
            power_sig = self.getPower_sig(
                gin=gin,
                M0=self.getM0(pol, ns=self.ns),
                vol=vol,
                u=u,
                tipAngle=self.getTipAngle(
                    Omega_a=self.getOmega_a(), T2=T2, tau_a=tau_a
                ),
            )
            check(power_sig)

            power_SPN = self.getPower_SPN(
                gin=gin,
                M_SPN=self.getM0(pol=PhysicalQuantity(1, ""), ns=self.ns_SPN),
                vol=vol,
            )
            PSD_SPN = self.getPSD_SPN(power=power_SPN, Delta=1 / (pi * T2star))
            # check(nu_L)
            # check((SPN_PSDnoise**0.5).convert_to('microPhi_0 / Hz**0.5'))

            eta = self.getEfficPow(RBW_Hz=1 / Tmeas, ALP_lw_Hz=ALP_lw_Hz)

            # check(ALP_lw_Hz)
            powerNoise_MF = self.getPowerNoise_MF(
                # PSDnoise_SQUID=self.getPSDnoise_SQUID(nu=nu_a),
                PSDnoise_SQUID=PhysicalQuantity(4e-11, "Phi_0**2/Hz"),
                power_SPN=power_SPN,
                ALP_lw_Hz=ALP_lw_Hz,
                Tmeas=Tmeas,
                T2star=T2star,
                verbose=True,
            )

            glim = (5 * powerNoise_MF / (eta * power_sig)) ** 0.5 * self.gaNN_base
            # check((powerNoise_MF))
            # check(eta)
            # check(P_perp)
            glim = glim.convert_to("GeV**(-1)")
            if verbose:
                print("**************************")
                check(nu_a)
                check(tau_a)
                check(T2)
                check(T2star)
                check(Tmeas)
                check(gin)
                check(self.getPick2In())
                check(self.getM2Pick())
                check(pol)
                check(u)
                check(tipAngle)
                check(eta)
                check(powerNoise_MF)
                check(glim)
            return glim

        glim_list = []
        for i, freq in enumerate(freq_list):
            glim_list.append(sensi(freq, Tmeas_list[i]))

        return glim_list

    def plotThermMethanol_Sensi(
        self,
        ax: plt.Axes,
        freq_list: list[PhysicalQuantity],  # frequencies [Hz]
        Tmeas_list: list[PhysicalQuantity],
        verbose: bool = False,
    ):
        """
        gaNN sensitivity with a thermally-polarized methanol

        compute SNR for gaNN = 1 GeV^-1, so as to estimate the gaNN limit
        """
        glim_list = self.getThermMethanol_Sensi(freq_list, Tmeas_list, verbose=verbose)
        glim_vals = []
        freq_vals = []
        for glim in glim_list:
            # assert isinstance(glim, PhysicalQuantity), "glim is not a PhysicalQuantity"
            glim = glim.convert_to("GeV**(-1)")
            glim_vals.append(glim.value)
        for freq in freq_list:
            # assert isinstance(glim, PhysicalQuantity), "glim is not a PhysicalQuantity"
            freq = freq.convert_to("Hz")
            freq_vals.append(freq.value)

        glim_vals = np.array(glim_vals)
        freq_vals = np.array(freq_vals)

        # check(np.amin(freq))  # 1348562.406998
        # freq0 = 1348560
        top = 1e-0
        ax.fill_between(
            x=freq_vals,
            y1=glim_vals,
            y2=top,
            color="tab:green",
            edgecolor="k",
            linewidth=0.5,
            alpha=0.5,
            zorder=2,
        )

        # ax.set_xlim(0, 53)
        # ax.set_ylim(1e-10, 1e-2)

        ax.set_xscale("log")
        # ax.set_yscale('linear')
        # ax.set_xscale('log')
        ax.set_yscale("log")

        ax.set_xticks([1e3, 1e4, 1e5, 1e6, 4.3e6])
        # xticklabels = []
        # for freq in freq_vals:
        #     xticklabels.append(f'{}')
        ax.set_xticklabels(["1 kHz", "10 kHz", "100 kHz", "1 MHz", "4.3 MHz"])
        # ax.set_yticks([1e-2, 1e-4, 1e-6, 1e-8, 1e-10])
        # ax.set_yticklabels([1, 10, 100, 1000])
        # ax.set_ylim(bottom=0.1)
        # ax.legend(loc='upper right')
        ax.grid(True)

        ax.set_xlabel("Frequency")
        ax.set_ylabel("$|g_\\mathrm{aNN}| [\\mathrm{GeV}^{-1}]$ ", color="k")

        return

    def getXe129_Sensi_Phase1(
        self,
        freq_list: list[PhysicalQuantity],  # frequencies [Hz]
        Tmeas_list: list[PhysicalQuantity],
        verbose: bool = False,
    ) -> list[PhysicalQuantity]:
        """
        gaNN sensitivity with a thermally-polarized methanol

        compute SNR for gaNN = 1 GeV^-1, so as to estimate the gaNN limit
        """
        # if np.amin(freq)<1e3 or np.amax(freq) > 4.3

        # get self.sample, self.ns, self.ns_SPN
        self.sampleLXe129()  #
        vol = PhysicalQuantity(1.2, "cm**3")

        # set inhomogeneity
        self.useShims(PhysicalQuantity(2, "ppm"))

        # temperature
        # temp = PhysicalQuantity(273.15 - 90, "K")

        # sensitivity at nu_a
        def sensi(
            nu_a: PhysicalQuantity,  # frequency [Hz]
            Tmeas: PhysicalQuantity,  # frequencies [Hz]
            # eta:PhysicalQuantity
        ) -> PhysicalQuantity:
            ALP_lw_Hz = nu_a / self.Q_a
            # check(ALP_lw_Hz)
            tau_a = 1 / (pi * ALP_lw_Hz)

            # nu_a = nu_a.convert_to("Hz")
            # self.freq_uplimit = self.freq_uplimit.convert_to("Hz")
            nu_L = nu_a if (nu_a - self.nu_uplimit).value <= 0 else self.nu_uplimit

            T2 = self.GetT2(nu_L)
            Tdelta = self.GetTdelta(NMR_lw_ppm=self.NMR_lw_ppm, nu=nu_L)
            T2star = (T2 ** (-1) + Tdelta ** (-1)) ** (-1)
            # print("**************************")
            # check(nu_a)
            # check(T2)
            # check(T2star)
            # check(Tmeas)
            # Phi pi/2

            gin = self.getPick2In() * self.getM2Pick()
            pol = PhysicalQuantity(0.1, "")
            u = self.getSpecFac(
                nu_L=nu_a, nu_a=nu_a, T2=self.GetT2(nu_L), Tdelta=Tdelta, tau_a=tau_a
            )

            power_sig = self.getPower_sig(
                gin=gin,
                M0=self.getM0(pol, ns=self.ns),
                vol=vol,
                u=u,
                tipAngle=self.getTipAngle(
                    Omega_a=self.getOmega_a(), T2=T2, tau_a=tau_a
                ),
            )

            power_SPN = self.getPower_SPN(
                gin=gin,
                M_SPN=self.getM0(pol=PhysicalQuantity(1, ""), ns=self.ns_SPN),
                vol=vol,
            )
            PSD_SPN = self.getPSD_SPN(power=power_SPN, Delta=1 / (pi * T2star))
            # check((SPN_PSDnoise**0.5).convert_to('microPhi_0 / Hz**0.5'))

            eta = self.getEfficPow(RBW_Hz=1 / Tmeas, ALP_lw_Hz=ALP_lw_Hz)
            # check(ALP_lw_Hz)
            powerNoise_MF = self.getPowerNoise_MF(
                PSDnoise_SQUID=self.getPSDnoise_SQUID(nu=nu_a),
                power_SPN=power_SPN,
                ALP_lw_Hz=ALP_lw_Hz,
                Tmeas=Tmeas,
                T2star=T2star,
            )

            glim = (5 * powerNoise_MF / (eta * power_sig)) ** 0.5 * self.gaNN_base
            # check((powerNoise_MF))
            # check(eta)
            # check(P_perp)
            glim = glim.convert_to("GeV**(-1)")
            if verbose:
                check(glim)
            return glim

        glim_list = []
        for i, freq in enumerate(freq_list):
            glim_list.append(sensi(freq, Tmeas_list[i]))

        return glim_list

    def plotXe129_Sensi_Phase1(
        self,
        ax: plt.Axes,
        freq_list: list[PhysicalQuantity],  # frequencies [Hz]
        Tmeas_list: list[PhysicalQuantity],
        verbose: bool = False,
    ):
        """
        gaNN sensitivity with a thermally-polarized methanol

        compute SNR for gaNN = 1 GeV^-1, so as to estimate the gaNN limit
        """
        glim_list = self.getXe129_Sensi_Phase1(freq_list, Tmeas_list, verbose=verbose)
        glim_vals = []
        freq_vals = []
        for glim in glim_list:
            # assert isinstance(glim, PhysicalQuantity), "glim is not a PhysicalQuantity"
            glim = glim.convert_to("GeV**(-1)")
            glim_vals.append(glim.value)
        for freq in freq_list:
            # assert isinstance(glim, PhysicalQuantity), "glim is not a PhysicalQuantity"
            freq = freq.convert_to("Hz")
            freq_vals.append(freq.value)

        glim_vals = np.array(glim_vals)
        freq_vals = np.array(freq_vals)

        # ax01 = fig.add_subplot(gs[0, 1])  #
        # ax10 = fig.add_subplot(gs[1, 0])  #
        # ax11 = fig.add_subplot(gs[1, 1])  #
        # check(np.amin(freq))  # 1348562.406998
        # freq0 = 1348560
        ax.plot(
            freq_vals,
            glim_vals,
            linestyle="--",
            color="tab:green",
            label="updated CASPEr-G-LF limit",
        )
        top = 1e-0
        ax.fill_between(
            x=freq_vals,
            y1=glim_vals,
            y2=top,
            color="tab:green",
            # edgecolor="tab:green",
            # linestyle='--',
            # linewidth=2,
            alpha=0.5,
            # label='updated CASPEr-G-LF limit',
            zorder=2,
        )
        # ax.set_xlim(0, 53)
        # ax.set_ylim(1e-10, 1e-2)

        ax.set_xscale("log")
        # ax.set_yscale('linear')
        # ax.set_xscale('log')
        ax.set_yscale("log")

        # ax.set_xticks(freq_vals[[0, 1, 2, 4]])
        # ax.set_xticklabels(["1 kHz", "10 kHz", "100 kHz", "1.2 MHz"])
        # ax.set_yticks([1e-2, 1e-4, 1e-6, 1e-8, 1e-10])
        # ax.set_yticklabels([1, 10, 100, 1000])
        # ax.set_ylim(bottom=0.1)
        # ax.legend(loc='upper right')
        ax.grid(True)

        ax.set_xlabel("Frequency")
        ax.set_ylabel("$|g_\\mathrm{aNN}| [\\mathrm{GeV}^{-1}]$ ", color="k")
        return freq_vals, glim_vals

    def getXe129_Sensi_Phase2(
        self,
        freq_list: list[PhysicalQuantity],  # frequencies [Hz]
        Tmeas_list: list[PhysicalQuantity],
        verbose: bool = False,
    ) -> list[PhysicalQuantity]:
        """
        gaNN sensitivity with a thermally-polarized methanol

        compute SNR for gaNN = 1 GeV^-1, so as to estimate the gaNN limit
        """
        # if np.amin(freq)<1e3 or np.amax(freq) > 4.3
        # self.useLF_Magnet()

        # get self.sample, self.ns, self.ns_SPN
        self.sampleLXe129_approx()  #
        vol = PhysicalQuantity(10, "cm**3")
        pol = PhysicalQuantity(0.05, "")

        # set inhomogeneity
        self.useShims(PhysicalQuantity(2, "ppm"))

        # sensitivity at nu_a
        def sensi(
            nu_a: PhysicalQuantity,  # frequency [Hz]
            Tmeas: PhysicalQuantity,  # frequencies [Hz]
            # eta:PhysicalQuantity
        ) -> PhysicalQuantity:
            ALP_lw_Hz = nu_a / self.Q_a
            # check(ALP_lw_Hz)
            tau_a = (1 / (pi * ALP_lw_Hz)).convert_to("s")

            # nu_a = nu_a.convert_to("Hz")
            # self.freq_uplimit = self.freq_uplimit.convert_to("Hz")
            nu_L = nu_a if (nu_a - self.nu_uplimit).value <= 0 else self.nu_uplimit

            T2 = self.GetT2(nu_L)
            Tdelta = self.GetTdelta(NMR_lw_ppm=self.NMR_lw_ppm, nu=nu_L)
            # check(self.NMR_lw_ppm)
            # check(nu_L)
            T2star = (T2 ** (-1) + Tdelta ** (-1)) ** (-1)
            # print("**************************")
            # check(nu_a)
            # check(T2)
            # check(T2star)
            # check(Tmeas)
            # Phi pi/2

            gin = self.getPick2In() * self.getM2Pick()
            # check(self.getPick2In())
            u = self.getSpecFac(
                nu_L=nu_a, nu_a=nu_a, T2=self.GetT2(nu_L), Tdelta=Tdelta, tau_a=tau_a
            )
            check(u)
            M0 = self.getM0(pol, ns=self.ns)
            tipAngle = self.getTipAngle(Omega_a=self.getOmega_a(), T2=T2, tau_a=tau_a)
            # check(M0 * sin(tipAngle.value))

            power_sig = self.getPower_sig(
                gin=gin,
                M0=self.getM0(pol, ns=self.ns),
                vol=vol,
                u=u,
                tipAngle=self.getTipAngle(
                    Omega_a=self.getOmega_a(), T2=T2star, tau_a=tau_a
                ),
            )

            power_SPN = self.getPower_SPN(
                gin=gin,
                M_SPN=self.getM0(pol=PhysicalQuantity(1, ""), ns=self.ns_SPN),
                vol=vol,
            )
            PSD_SPN = self.getPSD_SPN(power=power_SPN, Delta=1 / (pi * T2star))
            # check((SPN_PSDnoise**0.5).convert_to('microPhi_0 / Hz**0.5'))

            eta = self.getEfficPow(RBW_Hz=1 / Tmeas, ALP_lw_Hz=ALP_lw_Hz)
            # check(ALP_lw_Hz)
            powerNoise_MF = self.getPowerNoise_MF(
                PSDnoise_SQUID=self.getPSDnoise_SQUID(nu=nu_a),
                power_SPN=power_SPN,
                ALP_lw_Hz=ALP_lw_Hz,
                Tmeas=Tmeas,
                T2star=T2star,
            )

            glim = (5 * powerNoise_MF / (eta * power_sig)) ** 0.5 * self.gaNN_base
            # check((powerNoise_MF))
            # check(eta)
            # check(P_perp)
            glim = glim.convert_to("GeV**(-1)")
            if verbose:

                M_Transver_T2 = (M0 * sin(tipAngle.value)).convert_to("ampere / meter")
                M_Transver_T2star = (
                    M0
                    * self.getTipAngle(
                        Omega_a=self.getOmega_a(), T2=T2star, tau_a=tau_a
                    )
                ).convert_to("ampere / meter")

                sample_R = PhysicalQuantity(4, "mm")
                sample_H = PhysicalQuantity(24, "mm")
                sample_area = 2 * pi * sample_R * sample_H
                sample_vol = pi * sample_R**2 * sample_H
                flux_sample = M_Transver_T2 * mu_0 * sample_area
                # flux_sample = flux_sample.convert_to('Phi_0')
                ratio = self.getM2Pick() * sample_vol / (sample_area)
                ratio = ratio.convert_to("")
                # check(ratio)
                T2_T2star_ratio = (T2 / T2star).convert_to("")
                self.rho_E_DM = self.rho_E_DM.convert_to("eV/m**3")

                print(
                    f"Transver magnetization (T2*) = {M_Transver_T2star.value:.2e}",
                    M_Transver_T2star.unit,
                )
                print(
                    f"Transver magnetization (T2) = {M_Transver_T2.value:.2e}",
                    M_Transver_T2.unit,
                )
                print(f"F0 = {nu_a.value:.2e}", nu_a.unit)
                print(f"T2 = {T2.value}", T2.unit)
                print(f"T2* = {T2star.value:.2e}", T2star.unit)
                print(
                    f"T2/T2* ratio = {T2_T2star_ratio.value:.2e}", T2_T2star_ratio.unit
                )
                print(f"M0 = {M0.value:.2e}", M0.unit)
                print(f"Volume = {vol.value:.2e}", vol.unit)
                print(
                    f"Coupling gaNN = {self.gaNN_base.value:.2e}", self.gaNN_base.unit
                )
                print(f"DM density = {self.rho_E_DM.value:.2e}", self.rho_E_DM.unit)
                print(f"DM rms velocity = {self.va.value:.2e}", self.va.unit)
                print(
                    f"Rabi frequency = {self.getOmega_a().value:.2e}",
                    self.getOmega_a().unit,
                )
                print(f"Axion Q factor = {self.Q_a.value:.2e}", self.Q_a.unit)
                print(f"axion coherence time = {tau_a.value:.2e}", tau_a.unit)
                # print(f' = {.value:.2e}', .unit)
                check(glim)
            return glim

        glim_list = []
        for i, freq in enumerate(freq_list):
            glim_list.append(sensi(freq, Tmeas_list[i]))
        # check(vol)
        # check(pol)
        # check(self.ns)
        # check(self.rho_E_DM)
        return glim_list

    def plotXe129_Sensi_Phase2(
        self,
        ax: plt.Axes,
        freq_list: list[PhysicalQuantity],  # frequencies [Hz]
        Tmeas_list: list[PhysicalQuantity],
        verbose: bool = False,
    ):
        """
        gaNN sensitivity with a thermally-polarized methanol

        compute SNR for gaNN = 1 GeV^-1, so as to estimate the gaNN limit
        """
        glim_list = self.getXe129_Sensi_Phase2(freq_list, Tmeas_list, verbose=verbose)
        glim_vals = []
        freq_vals = []
        for glim in glim_list:
            # assert isinstance(glim, PhysicalQuantity), "glim is not a PhysicalQuantity"
            glim = glim.convert_to("GeV**(-1)")
            glim_vals.append(glim.value)
        for freq in freq_list:
            # assert isinstance(glim, PhysicalQuantity), "glim is not a PhysicalQuantity"
            freq = freq.convert_to("Hz")
            freq_vals.append(freq.value)

        glim_vals = np.array(glim_vals)
        freq_vals = np.array(freq_vals)

        # ax01 = fig.add_subplot(gs[0, 1])  #
        # ax10 = fig.add_subplot(gs[1, 0])  #
        # ax11 = fig.add_subplot(gs[1, 1])  #
        # check(np.amin(freq))  # 1348562.406998
        # freq0 = 1348560
        ax.plot(
            freq_vals,
            glim_vals,
            linestyle="--",
            color="tab:green",
            label="updated CASPEr-G limit",
        )
        top = 1e-0
        ax.fill_between(
            x=freq_vals,
            y1=glim_vals,
            y2=top,
            color="tab:green",
            # edgecolor="tab:green",
            # linestyle='--',
            # linewidth=2,
            alpha=0.5,
            # label='updated CASPEr-G-LF limit',
            zorder=2,
        )
        # ax.set_xlim(0, 53)
        # ax.set_ylim(1e-10, 1e-2)

        ax.set_xscale("log")
        # ax.set_yscale('linear')
        # ax.set_xscale('log')
        ax.set_yscale("log")

        # ax.set_xticks(freq_vals)
        # ax.set_xticklabels(["1 kHz", "10 kHz", "100 kHz", "1.2 MHz"])
        # ax.set_yticks([1e-2, 1e-4, 1e-6, 1e-8, 1e-10])
        # ax.set_yticklabels([1, 10, 100, 1000])
        # ax.set_ylim(bottom=0.1)
        # ax.legend(loc='upper right')
        ax.grid(True)

        ax.set_xlabel("Frequency")
        ax.set_ylabel("$|g_\\mathrm{aNN}| [\\mathrm{GeV}^{-1}]$ ", color="k")

    def plot2017OverviewLimit(self, ax: plt.Axes, verbose: bool = False):
        filepath = r"limit_data\AxionNeutron\Projections\CASPEr_wind.txt"
        masses, limits = np.loadtxt(filepath, unpack=True)
        freq_vals = []
        for mass in masses:
            freq_vals.append(energy2freq(PhysicalQuantity(mass, "eV")).value)
        ax.plot(
            freq_vals,
            limits,
            linestyle="--",
            color="tab:red",
            label="CASPEr-G limit from 2018",
        )
        top = 1e-0
        ax.fill_between(
            x=freq_vals,
            y1=limits,
            y2=top,
            color="tab:red",
            edgecolor="tab:green",
            # linestyle='--',
            linewidth=0,
            alpha=0.3,
            # label='updated CASPEr-G-LF limit',
            zorder=2,
        )

    def measTime1tau_a(
        self,
        freq: PhysicalQuantity,
    ) -> PhysicalQuantity:
        return (self.Q_a / freq).convert_to("s")

    def measTime100tau_a(
        self,
        freq: PhysicalQuantity,
    ) -> PhysicalQuantity:
        return (100 * self.Q_a / freq).convert_to("s")

    def getTotalScanTime(
        self,
        freq_start: PhysicalQuantity,
        freq_stop: PhysicalQuantity,
        func_measTime: Callable[
            [PhysicalQuantity], PhysicalQuantity
        ],  # measurement time a some frequency
    ) -> PhysicalQuantity:

        def func_measTime_s(
            # func_measTime: Callable[[PhysicalQuantity], PhysicalQuantity],
            freq: float,
        ):
            measTime = func_measTime(PhysicalQuantity(freq, "Hz")).convert_to("s")
            return measTime.value

        # def getTau_a(freq: PhysicalQuantity, Q_a: PhysicalQuantity):
        #     return (Q_a / freq).convert_to('s')

        result, error = quad(
            func_measTime_s,
            freq_start.convert_to("Hz").value,
            freq_stop.convert_to("Hz").value,
        )
        check(result)
        check(error)
        totalScanTime = PhysicalQuantity(result, "s").convert_to("year")
        check(totalScanTime)
        return totalScanTime

    # def plotSensi(ax:plt.Axes, x, y, ytop):
