##################################################
#
##################################################


import numpy as np

from Sample import Sample

from Envelope import (
    PhysicalQuantity,
    gamma_Xe129,
    gamma_p,
    mu_p,
    mu_Xe129,
    hbar,
    k,
    mu_0,
)


class SQUID:
    def __init__(
        self,
        name=None,
        Lin: PhysicalQuantity = None,  # inoput coil inductance
        Min: PhysicalQuantity = None,  # mutual inductance
        Mf: PhysicalQuantity = None,
        Rf: PhysicalQuantity = None,
        attenuation: PhysicalQuantity = None,
    ):  # in Ohm
        """
        name : str
            name of the SQUID. default to 'PhiC6L1W'. 'PhiC73L1' is the other option
        Mf : float
            feedback sensitivity which can be found in the SQUID specifications.
            For the SQUID on channel 3 we usually use before 2024-05, M_f = 1 / (44.12e-6) \Phi_0 / A
            For the new SQUID on channel 2 we installed on 2024-05 (Sensor ID: C649_O12), M_f = 1 / (44.16e-6) \Phi_0 / A
        Rf : float
            feedback resistance in Ohm
        attenuation : float
            in dB
        """
        self.name = name
        self.Mf = Mf
        self.Rf = Rf
        self.attenuation = attenuation


SQD = SQUID(
    name=None,
    Lin=PhysicalQuantity(400, "nH"),  # inoput coil inductance
    Min=PhysicalQuantity(1 / 0.5194, "Phi_0/microA"),  # mutual inductance
    Mf=None,
    Rf=None,
    attenuation=None,
)


class Pickup:
    def __init__(
        self,
        name=None,
        gV: PhysicalQuantity = PhysicalQuantity(
            37.0, "1/m"
        ),  # sample-to-pickup coupling strength
        Lcoil: PhysicalQuantity = None,  # coil inductance
        vol: PhysicalQuantity = None,  # volume of the pickup
        verbose: bool = False,
    ):  # in Ohm
        """
        name : str
            name of the
        """
        self.name = name
        self.gV = gV
        self.vol = vol


gradiometer = Pickup(
    name="gradiometer on PEEK",
    gV=PhysicalQuantity(37.0, "1/m"),  # sample-to-pickup coupling strength
    # assume cylindrical sample (R=4 mm, H=22.53 mm) coupling to the gradiometer
    Lcoil=PhysicalQuantity(400, "nH"),
    vol=PhysicalQuantity(np.pi * 14**2 * 22.53, "mm**3"),
)


class Magnet:
    def __init__(
        self,
        name=None,
        lw_ppm: PhysicalQuantity = None,
        verbose: bool = False,
    ):  # in Ohm
        """
        name : str
            name of the SQUID. default to 'PhiC6L1W'. 'PhiC73L1' is the other option
        """
        self.name = name
        self.lw_ppm = lw_ppm


SCmagnet = Magnet(
    name=None,
    lw_ppm=PhysicalQuantity(10, "ppm"),
)


class CASPErLF:
    def __init__(
        self,
        name=None,
        sample: Sample = None,
        pickup: Pickup = None,
        SQUID: SQUID = None,
        magnet: Magnet = None,
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
        self.magnet = magnet

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
