import numpy as np


from Envelope import PhysicalQuantity


class Sample:
    """
    Describe the sample used in experiments.
    Only consider samples in one phase.
    """

    def __init__(
        self,
        name=None,  # name of the sample
        gamma: PhysicalQuantity = None,  # [Hz/T]. Remember input it like 2 * np.pi * 11.777*10**6
        massDensity: PhysicalQuantity = None,  # [g/cm^3] at STP
        molarMass: PhysicalQuantity = None,  # [g/mol]
        numOfSpins: PhysicalQuantity = None,  # [per molecule]
        T2: PhysicalQuantity = None,  # [s]
        T1: PhysicalQuantity = None,  # [s]
        # pol=np.NaN,
        vol: PhysicalQuantity = np.NaN,
        # boilPt: PhysicalQuantity = None,  #
        # meltPt: PhysicalQuantity = None,  #
        # spindenisty_liquid=None,  # [mol/cm^3]
        # spindenisty_gas=None,  # [g/cm^3] at STP
        # spindenisty_solid=None,  # [mol/cm^3]
        # shareofpeaks=None,  # array or list.
        # temp: PhysicalQuantity = None,
        magMom: PhysicalQuantity = np.NaN,  # magnetic dipole moment
        verbose: bool = False,
    ):
        """

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
        self.gamma = gamma

        self.massDensity = massDensity
        self.molarMass = molarMass
        self.numOfSpins = numOfSpins

        assert self.molarMass is not None
        self.spinNumDensity = self.numOfSpins * self.massDensity / self.molarMass
        # raise ValueError("self.molarMass is None. ")

        self.T2 = T2  # [s]
        self.T1 = T1  # [s]
        self.vol = vol
        # self.boilPt = boilPt
        # self.meltPt = meltPt
        self.mdm = magMom


liquid_Xe129 = Sample(
    name="Liquid Xe-129",  # name of the sample
    gamma=2 * np.pi * (-11.777) * 10**6,  # [Hz/T]. Remember input it with 2 * np.pi
    numofnuclei=1,  #
    tempunit="K",  # temperature scale
    boilpt=165.051,  # [K]
    meltpt=161.40,  # [K]
    massDensity=2.942,  # [g/cm^3] at boiling point
    density_gas=5.894 * 10**3,  # [g/cm^3] at STP
    density_solid=None,  # [g/cm^3]
    molarMass=131.2930,  # [g/mol]
    spindenisty_liquid=None,  # [mol/cm^3]
    spindenisty_gas=None,  # [g/cm^3] at STP
    spindenisty_solid=None,  # [mol/cm^3]
    shareofpeaks=[1.0],  # array or list.
    T2=None,  # [s]
    T1=1000,  # [s]
    pol=0.5,
    verbose=False,
)

Methanol = Sample(
    name="C-12 Methanol",  # name of the atom/molecule
    gamma=2
    * np.pi
    * 42.577478518
    * 10**6,  # [Hz/T]. Remember input it like 2 * np.pi * 11.777*10**6
    numofnuclei=4,  #
    tempunit="K",  # temperature scale
    boilpt=337.8,  # [K]
    meltpt=175.6,  # [K]
    massDensity=0.792,  # [g/cm^3] at boiling point
    density_gas=None,  # [g/cm^3] at STP
    density_solid=None,  # [g/cm^3]
    molarMass=32.04,  # [g/mol]
    spindenisty_liquid=None,  # [mol/cm^3]
    spindenisty_gas=None,  # [g/cm^3] at STP
    spindenisty_solid=None,  # [mol/cm^3]
    shareofpeaks=[3.0 / 4, 1.0 / 4],  # array or list.
    pol=1.76876e-7,
    verbose=False,
)
Ethanol = Sample(
    name="Ethanol",  # name of the atom/molecule
    gamma=2
    * np.pi
    * 42.577478518
    * 10**6,  # [Hz/T]. Remember input it like 2 * np.pi * 11.777*10**6
    numofnuclei=6,  #
    tempunit="K",  # temperature scale
    boilpt=351.38,  # [K]
    meltpt=159.01,  # [K]
    massDensity=0.78945,  # [g/cm^3] at boiling point
    density_gas=None,  # [g/cm^3] at STP
    density_solid=None,  # [g/cm^3]
    molarMass=46.069,  # [g/mol]
    spindenisty_liquid=None,  # [mol/cm^3]
    spindenisty_gas=None,  # [g/cm^3] at STP
    spindenisty_solid=None,  # [mol/cm^3]
    shareofpeaks=[3 / 6.0, 2.0 / 6, 1.0 / 6],  # array or list.
    T2=None,  # [s]
    T1=None,  # [s]
    pol=1.76876e-7,
    verbose=False,
)
