##################################################
#
##################################################


import numpy as np
from typing import Optional
from src.Sample import Sample

from src.Envelope import (
    PhysicalQuantity,
    _safe_convert,
    gamma_p,
    hbar,
    kB,
    mu_0,
    c,
)
import h5py
from src.functioncache import PhysicalObject

class SQUID(PhysicalObject):
    def __init__(
        self,
        name=None,
        Lin: Optional[PhysicalQuantity] = None,  # inoput coil inductance
        Min: Optional[PhysicalQuantity] = None,  # mutual inductance
        Mf: Optional[PhysicalQuantity] = None,
        Rf: Optional[PhysicalQuantity] = None,
        attenuation: Optional[PhysicalQuantity] = None,
    ):  # in Ohm
        """
        """
        self.name = name
        self.Lin = Lin
        self.Min = Min
        self.Mf = Mf
        self.Rf = Rf
        self.attenuation = attenuation
        # Specify common units for automatic conversion
        self.physicalQuantities = {
            "Lin": "nH",
            "Min": "Phi_0/microA",
            "Mf": "Phi_0/microA",
            "Rf": "ohm",
            "attenuation": "dB"
        }   
        # make sure that we use common units for quantities
        self.useCommonUnits()

class Pickup(PhysicalObject):
    def __init__(
        self,
        name=None,
        Lcoil: Optional[PhysicalQuantity] = None,  # coil inductance
        gV: Optional[PhysicalQuantity] = None,  # sample-to-pickup coupling strength
        vol: Optional[PhysicalQuantity] = None,  # volume of the pickup
        verbose: bool = False,
    ):  # in Ohm
        """
        name : str
            name of the
        """
        self.name = name
        self.Lcoil = Lcoil
        self.gV = gV
        self.vol = vol
        # Specify common units for automatic conversion
        self.physicalQuantities = {
            "Lcoil": "nH",
            "gV": "1/meter",
            "vol": "cm**3"
        }
        # make sure that we use common units for quantities
        self.useCommonUnits()



class Magnet(PhysicalObject):
    def __init__(
        self,
        name=None,
        B0: Optional[PhysicalQuantity] = None,
        lw: Optional[PhysicalQuantity] = None,
        verbose: bool = False,
    ):  # in Ohm
        """
        name : str
            name of the SQUID. default to 'PhiC6L1W'. 'PhiC73L1' is the other option
        """
        self.name = name
        self.B0 = B0
        self.lw = lw
        # Specify common units for automatic conversion
        self.physicalQuantities = {
            "B0": "T",
            "lw": ""
        }
        # make sure that we use common units for quantities
        self.useCommonUnits()



class LockinAmplifier(PhysicalObject):
    def __init__(
        self,
        name=None,
        demodFreq: Optional[PhysicalQuantity] = None,
        sampRate: Optional[PhysicalQuantity] = None,
        DTRC_TC: Optional[PhysicalQuantity] = None,
        DTRC_order: Optional[PhysicalQuantity] = None,
        # : Optional[PhysicalQuantity] = None,
        # : Optional[PhysicalQuantity] = None,
        verbose: bool = False,
    ):  # in Ohm
        """
        name : str
            name of the SQUID. default to 'PhiC6L1W'. 'PhiC73L1' is the other option
        """
        self.name = name
        self.demodFreq = demodFreq
        self.sampRate = sampRate
        self.DTRC_TC = DTRC_TC
        self.DTRC_order = DTRC_order
        self.physicalQuantities = {
            "demodFreq": "Hz",
            "sampRate": "Hz",
            "DTRC_TC": "s",
            "DTRC_order": ""
        }
        # make sure that we use common units for quantities
        self.useCommonUnits()


C649_O12 = SQUID(
    name="C649_O12 in Nb capsule S219, channel 2",
    Lin=PhysicalQuantity(400, "nH"),  # inoput coil inductance
    Min=PhysicalQuantity(1 / 0.525, "Phi_0/microA"),  # mutual inductance
    Mf=PhysicalQuantity(1 / 44.16, "Phi_0/microA"),
    Rf=PhysicalQuantity(3, "kiloohm"),
    attenuation=None,
)


gradiometer = Pickup(
    name="(old) gradiometer on PEEK",
    Lcoil=PhysicalQuantity(400, "nH"),
    gV=PhysicalQuantity(37.0, "1/m"),  # sample-to-pickup coupling strength
    # assume cylindrical sample (R=4 mm, H=22.53 mm) coupling to the gradiometer
    vol=PhysicalQuantity(np.pi * 14**2 * 22.53, "mm**3"),
)

LFmagnet = Magnet(
    name="LF magnet",
    B0=PhysicalQuantity(0.1, "T"),
    lw=PhysicalQuantity(10, "ppm"),
)

Halbach = Magnet(name=None, lw=None, B0=PhysicalQuantity(1.0, "T"))

LIA = LockinAmplifier(
    name="virtual LIA",
    demodFreq=PhysicalQuantity(1.0, "MHz"),
    sampRate=None,
    DTRC_TC=PhysicalQuantity(1.0, "s"),
    DTRC_order=PhysicalQuantity(0, ""),
    verbose=False,
)


class CASPEr_LF:
    def __init__(
        self,
        name=None,
        sample: Sample = None,
        pickup: Pickup = None,
        SQUID: SQUID = None,
        magnet_pol: Magnet = None,
        magnet_det: Magnet = None,
        verbose: bool = False,
    ):  # in Ohm
        """
        name : str
            name of the SQUID. default to 'PhiC6L1W'. 'PhiC73L1' is the other option
        """
        self.name = name
        self.sample = sample
        self.pickup = pickup
        self.SQUID = SQUID
        self.magnet_pol = magnet_pol
        self.magnet_det = magnet_det

    def getPhi_pick(
        self,
        M0: PhysicalQuantity,
    ):
        """
        get the flux in the pickup coil (gradiometer)
        """
        Phi_pick = self.pickup.gV * mu_0 * M0 * self.sample.vol
        Phi_pick = Phi_pick.convert_to("Phi_0")
        return Phi_pick

    def getThermalPol(
        self,
        B_pol: PhysicalQuantity | None = None,
        temp: PhysicalQuantity | None = None,
    ) -> float:
        """
        Compute the thermal polarization of the sample.

        The polarization is given by tanh(ħ * γ * B_pol / (2 k_B T)).

        Parameters
        ----------
        B_pol : PhysicalQuantity, optional
            Polarization magnetic field. If not provided, inferred from
            self.magnet_pol.B0 or self.magnet_det.B0.
        temp : PhysicalQuantity, optional
            Sample temperature. If not provided, inferred from self.sample.temp.

        Returns
        -------
        float
            Thermal polarization (dimensionless, between -1 and 1).
        """

        # Determine magnetic field
        if B_pol is None:
            B_pol = getattr(self.magnet_pol, "B0", None) or getattr(
                self.magnet_det, "B0", None
            )
        if B_pol is None:
            raise ValueError(
                "Polarization magnetic field (B_pol) not specified or unavailable."
            )

        # Determine temperature
        if temp is None:
            temp = getattr(self.sample, "temp", None)
        if temp is None:
            raise ValueError("Sample temperature (temp) not specified or unavailable.")

        # Compute polarization
        ratio = (hbar * gamma_p * B_pol) / (2 * kB * temp)
        pol = np.tanh(ratio.convert_to(""))  # ensure dimensionless before tanh

        return pol

    def getM0(
        self,
        pol,
        ns,
    ):
        """
        compute magnetization M0
        """
        M0 = (self.sample.mu * pol * ns).convert_to("A/m")
        # self.M0_SPN = (mu_p * ns_SPN).convert_to("A/m")
        return M0

    def getPhi_pick(
        self,
        M0: PhysicalQuantity,
        gV: PhysicalQuantity = PhysicalQuantity(
            37.0, "1/m"
        ),  # estimated from a  cylindrical sample (R=4 mm, H=22.53 mm) coupling to the gradiometer
        Vol: PhysicalQuantity = PhysicalQuantity(np.pi * 4.0**2 * 24, "mm**3"),
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
        Vol: PhysicalQuantity = PhysicalQuantity(np.pi * 4.0**2 * 24, "mm**3"),
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

    def getOmega_a(self, alpha=PhysicalQuantity(np.pi / 2, "rad")):
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
            * np.sin(alpha.value)
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
        PSDnoise_SPN = self.getPSD_SPN(power=power_SPN, Delta=1 / (np.pi * T2star))
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

        # if verbose:
        #     check((Tmeas - T2star).convert_to("s"))
        #     check(Navg)
        #     check(powerNoise_MF)

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
