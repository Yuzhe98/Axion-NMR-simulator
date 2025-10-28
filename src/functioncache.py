import inspect  # for check()
import re  # for check()
import time
import datetime
import sys
import math

# from turtle import colorpartial
import matplotlib
import matplotlib.axes
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
from matplotlib.patches import FancyArrowPatch

# from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d

# from astropy.time import Time
from datetime import datetime
from tqdm import tqdm
import random

from functools import partial

import scipy.stats as stats
from scipy.stats import rayleigh, uniform, norm, chi2, gamma

# import numba as nb
# from numba import jit, vectorize

# import pyautogui

# from DataAnalysis import LIASignal, SQUID
import pandas as pd

# from evidently.report import Report
# from evidently.metric_preset import DataDriftPreset


def GiveDateandTime():
    timestr = time.strftime("%Y%m%d_%H%M%S")
    # timestr = 'session_'+timestr
    return timestr


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
            )

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


def poly1(x, C0, C1):
    return C0 + C1 * x


def poly2(x, C0, C1, C2):
    return C0 + C1 * x + C2 * x**2


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


def estimateLorzfit(
    datax=None, datay=None, smooth=False, smoothlevel=1, debug=False, verbose=False
):
    """
    Return the estimated parameters as the initial guess for Lorentzian curve fitting

    Parameters
    ----------

    datax : array_like
        The independent variable where the data is measured. Should usually be an M-length sequence.
    datay : array_like
        The dependent data, a length M array - nominally f(xdata, ...).
    smooth : bool
        Smooth the data before estimating to avoid fitting to sharp noise peak. Smoothing is done by averaging
    smoothlevel : int
        2*smoothlevel+1 will be averaged to obtain 1 point
    debug : bool
        To demostrate the effect of smoothing. Otherwise it should be set to False.
    verbose : bool
        choose True to display processing information


    Returns
    -------
    the value of the Lorentzian function : ndarray or scalar

    Examples
    --------
    >>>

    Reference
    ---------
        scipy.optimize.curve_fit
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html

    """
    # check the length of datax and datay
    if len(datax) != len(datay):
        raise ValueError("len(datax) != len(datay)")

    if smooth and debug:
        datay_smooth10 = datay.copy()
        datay_smooth20 = datay.copy()
        datay_smooth30 = datay.copy()
        smoothlevel = 0
        fig = plt.figure(figsize=(8, 10))  #
        gs = gridspec.GridSpec(nrows=4, ncols=1)  #
        smooth0_ax = fig.add_subplot(gs[0, 0])
        smooth0_ax.plot(datax, datay, label="smooth level %d" % smoothlevel)

        smoothlevel = 10
        for i in range(smoothlevel, len(datay) - smoothlevel):
            datay_smooth10[i] = np.average(datay[i - smoothlevel : i + smoothlevel + 1])
        smooth10_ax = fig.add_subplot(gs[1, 0])
        smooth10_ax.plot(datax, datay_smooth10, label="smooth level %d" % smoothlevel)

        smoothlevel = 20
        for i in range(smoothlevel, len(datay) - smoothlevel):
            datay_smooth20[i] = np.average(datay[i - smoothlevel : i + smoothlevel + 1])
        smooth20_ax = fig.add_subplot(gs[2, 0])
        smooth20_ax.plot(datax, datay_smooth20, label="smooth level %d" % smoothlevel)

        smoothlevel = 30
        for i in range(smoothlevel, len(datay) - smoothlevel):
            datay_smooth30[i] = np.average(datay[i - smoothlevel : i + smoothlevel + 1])
        smooth30_ax = fig.add_subplot(gs[3, 0])
        smooth30_ax.plot(datax, datay_smooth30, label="smooth level %d" % smoothlevel)

        smooth0_ax.legend()
        smooth10_ax.legend()
        smooth20_ax.legend()
        smooth30_ax.legend()
        plt.tight_layout()
        plt.show()

    datay_smoothed = datay.copy()
    if smooth:
        for i in range(smoothlevel, len(datay) - smoothlevel):
            datay_smoothed[i] = np.average(datay[i - smoothlevel : i + smoothlevel + 1])
        for i in range(smoothlevel):
            datay_smoothed[i] = datay_smoothed[smoothlevel]
            datay_smoothed[-i] = datay_smoothed[-smoothlevel]
        if verbose:
            plt.figure()
            plt.plot(datax, datay, label="no smoothing")
            plt.plot(datax, datay_smoothed, label="after smoothing")
            plt.legend()
            plt.grid()
            plt.show()
    centerindex = np.argmax(datay_smoothed)
    center = datax[np.argmax(datay_smoothed)]
    # if centerindex<5 or abs(len(datay)-centerindex)<5:
    #     #print('signal peak is too close to the edge of range')
    #     raise ValueError('signal peak is too close to the edge of range')
    # amp = np.amax(datay)
    amp = datay[np.argmax(datay_smoothed)]
    amp_smoothed = np.amax(datay_smoothed)

    HMindexleft = 0
    HMindexright = len(datay) - 1
    HMindex = np.flatnonzero(
        (
            (datay_smoothed[1:] > amp_smoothed / 2.0)
            & (datay_smoothed[:-1] < amp_smoothed / 2.0)
        )
        | (
            (datay_smoothed[1:] < amp_smoothed / 2.0)
            & (datay_smoothed[:-1] > amp_smoothed / 2.0)
        )
    )
    # if verbose:
    #     print('HMindex.shape ', HMindex.shape)
    #     print('centerindex ', centerindex)
    #     print('HMindex \n', HMindex)
    for index in HMindex:
        if index <= centerindex and HMindexleft <= index:
            HMindexleft = index
        if index >= centerindex and index <= HMindexright:
            HMindexright = index
    if HMindexleft >= 1:
        HMindexleft -= 1
    if HMindexright <= (len(datax) - 2):
        HMindexright += 1
    gamma = np.average(
        [
            abs(datax[HMindexright] - datax[HMindexleft]),
            2 * abs(datax[centerindex] - datax[HMindexleft]),
            2 * abs(datax[centerindex] - datax[HMindexright]),
        ]
    )

    area = amp * np.pi * gamma / 2.0

    offset = 0
    if verbose:
        print(
            "estimateLorzfit [center, gamma, area, offset] ",
            [center, gamma, area, offset],
        )
    return [center, gamma, area, offset]


def dualLorentzian(
    x,
    center0,  # 0
    FWHM0,  # 1
    area0,  # 2
    center1,  # 3
    FWHM1,  # 4
    area1,  # 5
    offset,  # 6
):
    """
    Return the value of the dual-Lorentzian function

                        Γ0 A0                       Γ1 A1
        offset + ─────────────────────── + ───────────────────────
                  2(π(x-c0)^2+(Γ0/2)^2)     2(π(x-c1)^2+(Γ1/2)^2)

    Parameters
    ----------

    x : scalar or array_like
        argument of the Lorentzian function
    center0 and center1 : scalar
        the positions of two Lorentzian peaks
    gamma0 and gamma1 : scalar
        linewidth / full width of half maximum (FWHM) of Lorentzian peaks
    area0 and area1 : scalar
        areas under the Lorentzian curve (without taking offset into consideration)
    offset : scalar
        offset for the curve

    Returns
    -------
    the value of the Lorentzian function : ndarray or scalar

    Examples
    --------
    >>>

    Reference
    ---------
    Null

    """
    return (
        offset
        + 0.5 * abs(FWHM0) * area0 / (np.pi * ((x - center0) ** 2 + (0.5 * FWHM0) ** 2))
        + 0.5 * abs(FWHM1) * area1 / (np.pi * ((x - center1) ** 2 + (0.5 * FWHM1) ** 2))
    )


def estimatedualLorzfit(
    datax=None, datay=None, smooth=False, smoothlevel=1, debug=False, verbose=False
):
    """
    Return the estimated parameters as the initial guess for dual-Lorentzian curve fitting

    Parameters
    ----------

    datax : array_like
        The independent variable where the data is measured. Should usually be an M-length sequence.
    datay : array_like
        The dependent data, a length M array - nominally f(xdata, ...).
    smooth : bool
        Smooth the data before estimating to avoid fitting to sharp noise peak. Smoothing is done by averaging
    smoothlevel : int
        2*smoothlevel+1 will be averaged to obtain 1 point
    debug : bool
        To demostrate the effect of smoothing. Otherwise it should be set to False.
    verbose : bool
        choose True to display processing information


    Returns
    -------
    7 estimated parameters of the dual-Lorentzian function : ndarray or scalar

    Examples
    --------
    >>>

    Reference
    ---------
        scipy.optimize.curve_fit
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html

    """
    if len(datax) != len(datay):
        raise ValueError("len(datax) != len(datay)")

    if smooth and debug:
        datay_smooth10 = datay.copy()
        datay_smooth20 = datay.copy()
        datay_smooth30 = datay.copy()
        smoothlevel = 0
        fig = plt.figure(figsize=(8, 10))  #
        gs = gridspec.GridSpec(nrows=4, ncols=1)  #
        smooth0_ax = fig.add_subplot(gs[0, 0])
        smooth0_ax.plot(datax, datay, label="smooth level %d" % smoothlevel)

        smoothlevel = 10
        for i in range(smoothlevel, len(datay) - smoothlevel):
            datay_smooth10[i] = np.average(datay[i - smoothlevel : i + smoothlevel + 1])
        smooth10_ax = fig.add_subplot(gs[1, 0])
        smooth10_ax.plot(datax, datay_smooth10, label="smooth level %d" % smoothlevel)

        smoothlevel = 20
        for i in range(smoothlevel, len(datay) - smoothlevel):
            datay_smooth20[i] = np.average(datay[i - smoothlevel : i + smoothlevel + 1])
        smooth20_ax = fig.add_subplot(gs[2, 0])
        smooth20_ax.plot(datax, datay_smooth20, label="smooth level %d" % smoothlevel)

        smoothlevel = 30
        for i in range(smoothlevel, len(datay) - smoothlevel):
            datay_smooth30[i] = np.average(datay[i - smoothlevel : i + smoothlevel + 1])
        smooth30_ax = fig.add_subplot(gs[3, 0])
        smooth30_ax.plot(datax, datay_smooth30, label="smooth level %d" % smoothlevel)

        smooth0_ax.legend()
        smooth10_ax.legend()
        smooth20_ax.legend()
        smooth30_ax.legend()
        plt.tight_layout()
        plt.show()

    paras0 = estimateLorzfit(
        datax=datax,
        datay=datay,
        smooth=smooth,
        smoothlevel=smoothlevel,
        debug=debug,
        verbose=verbose,
    )
    paras1 = estimateLorzfit(
        datax=datax,
        datay=datay - Lorentzian(datax, paras0[0], paras0[1], paras0[2], paras0[3]),
        smooth=smooth,
        smoothlevel=smoothlevel,
        debug=debug,
        verbose=verbose,
    )
    return [
        paras0[0],
        paras0[1],
        paras0[2],
        paras1[0],
        paras1[1],
        paras1[2],
        paras0[3] + paras1[3],
    ]


def tribLorentzian(
    x,
    center0,  # 0
    gamma0,  # 1
    area0,  # 2
    center1,  # 3
    gamma1,  # 4
    area1,  # 5
    center2,  # 6
    gamma2,  # 7
    area2,  # 8
    offset,  # 9
):
    """
    Return the value of the trible-Lorentzian function

                        Γ0 A0                       Γ1 A1
        offset + ─────────────────────── + ───────────────────────
                  2(π(x-c0)^2+(Γ0/2)^2)     2(π(x-c1)^2+(Γ1/2)^2)

    Parameters
    ----------

    x : scalar or array_like
        argument of the Lorentzian function
    center0 and center1 : scalar
        the positions of two Lorentzian peaks
    gamma0 and gamma1 : scalar
        linewidth / full width of half maximum (FWHM) of Lorentzian peaks
    area0 and area1 : scalar
        areas under the Lorentzian curve (without taking offset into consideration)
    offset : scalar
        offset for the curve

    Returns
    -------
    the value of the Lorentzian function : ndarray or scalar

    Examples
    --------
    >>>

    Reference
    ---------
    Null

    """
    return (
        offset
        + 0.5 * gamma0 * area0 / (np.pi * ((x - center0) ** 2 + (0.5 * gamma0) ** 2))
        + 0.5 * gamma1 * area1 / (np.pi * ((x - center1) ** 2 + (0.5 * gamma1) ** 2))
        + 0.5 * gamma2 * area2 / (np.pi * ((x - center2) ** 2 + (0.5 * gamma2) ** 2))
    )


def estimatetribLorzfit(
    datax=None, datay=None, smooth=False, smoothlevel=1, debug=False, verbose=False
):
    """
    Return the estimated parameters as the initial guess for dual-Lorentzian curve fitting

    Parameters
    ----------

    datax : array_like
        The independent variable where the data is measured. Should usually be an M-length sequence.
    datay : array_like
        The dependent data, a length M array - nominally f(xdata, ...).
    smooth : bool
        Smooth the data before estimating to avoid fitting to sharp noise peak. Smoothing is done by averaging
    smoothlevel : int
        2*smoothlevel+1 will be averaged to obtain 1 point
    debug : bool
        To demostrate the effect of smoothing. Otherwise it should be set to False.
    verbose : bool
        choose True to display processing information


    Returns
    -------
    7 estimated parameters of the dual-Lorentzian function : ndarray or scalar

    Examples
    --------
    >>>

    Reference
    ---------
        scipy.optimize.curve_fit
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html

    """
    if len(datax) != len(datay):
        raise ValueError("len(datax) != len(datay)")

    if smooth and debug:
        datay_smooth10 = datay.copy()
        datay_smooth20 = datay.copy()
        datay_smooth30 = datay.copy()
        smoothlevel = 0
        fig = plt.figure(figsize=(8, 10))  #
        gs = gridspec.GridSpec(nrows=4, ncols=1)  #
        smooth0_ax = fig.add_subplot(gs[0, 0])
        smooth0_ax.plot(datax, datay, label="smooth level %d" % smoothlevel)

        smoothlevel = 10
        for i in range(smoothlevel, len(datay) - smoothlevel):
            datay_smooth10[i] = np.average(datay[i - smoothlevel : i + smoothlevel + 1])
        smooth10_ax = fig.add_subplot(gs[1, 0])
        smooth10_ax.plot(datax, datay_smooth10, label="smooth level %d" % smoothlevel)

        smoothlevel = 20
        for i in range(smoothlevel, len(datay) - smoothlevel):
            datay_smooth20[i] = np.average(datay[i - smoothlevel : i + smoothlevel + 1])
        smooth20_ax = fig.add_subplot(gs[2, 0])
        smooth20_ax.plot(datax, datay_smooth20, label="smooth level %d" % smoothlevel)

        smoothlevel = 30
        for i in range(smoothlevel, len(datay) - smoothlevel):
            datay_smooth30[i] = np.average(datay[i - smoothlevel : i + smoothlevel + 1])
        smooth30_ax = fig.add_subplot(gs[3, 0])
        smooth30_ax.plot(datax, datay_smooth30, label="smooth level %d" % smoothlevel)

        smooth0_ax.legend()
        smooth10_ax.legend()
        smooth20_ax.legend()
        smooth30_ax.legend()
        plt.tight_layout()
        plt.show()

    paras0 = estimateLorzfit(
        datax=datax,
        datay=datay,
        smooth=smooth,
        smoothlevel=smoothlevel,
        debug=debug,
        verbose=verbose,
    )
    paras1 = estimateLorzfit(
        datax=datax,
        datay=datay - Lorentzian(datax, paras0[0], paras0[1], paras0[2], paras0[3]),
        smooth=smooth,
        smoothlevel=smoothlevel,
        debug=debug,
        verbose=verbose,
    )
    paras2 = estimateLorzfit(
        datax=datax,
        datay=datay
        - Lorentzian(datax, paras0[0], paras0[1], paras0[2], paras0[3])
        - Lorentzian(datax, paras1[0], paras1[1], paras1[2], paras1[3]),
        smooth=smooth,
        smoothlevel=smoothlevel,
        debug=debug,
        verbose=verbose,
    )
    return [
        paras0[0],
        paras0[1],
        paras0[2],
        paras1[0],
        paras1[1],
        paras1[2],
        paras2[0],
        paras2[1],
        paras2[2],
        paras0[3] + paras1[3],
    ]


def Gaussian(x, center, sigma, area, offset):
    """
    Return the value of the Gaussian function

                           area                 1  (x-center)^2
        offset + ─────────────────────── exp(- ─── ──────────────)
                    sigma * sqrt(2 Pi)          2     sigma^2

    Parameters
    ----------

    x : scalar or array_like
        argument of the Lorentzian function
    center : scalar
        the position of the Lorentzian peak
    sigma : scalar
        variance of Gaussian function. FWHM = 2.35482 sigma, FWTM = 4.29193 sigma
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
    ---------
    https://en.wikipedia.org/wiki/Gaussian_function

    """
    return offset + area / (sigma * np.sqrt(2 * np.pi)) * np.exp(
        -0.5 * (x - center) ** 2 / sigma**2
    )


def estimateGaussfit(
    datax=None, datay=None, smooth=False, smoothlevel=1, debug=False, verbose=False
):
    fitparas = estimateLorzfit(
        datax=datax,
        datay=datay,
        smooth=smooth,
        smoothlevel=smoothlevel,
        debug=debug,
        verbose=verbose,
    )
    fitparas[1] /= 2.35482
    return fitparas


def dualGaussian(
    x,
    center0,  # 0
    sigma0,  # 1
    area0,  # 2
    center1,  # 3
    sigma1,  # 4
    area1,  # 5
    offset,  # 6
):
    """
    Return the value of the Gaussian function

                           area0                 1 (x-center0)^2              area1                 1  (x-center1)^2
        offset + ─────────────────────── exp(- ─── ──────────────) + ─────────────────────── exp(- ─── ──────────────)
                    sigma0 * sqrt(2 Pi)          2    sigma0^2         sigma1 * sqrt(2 Pi)          2    sigma1^2

    Parameters
    ----------

    x : scalar or array_like
        argument of the Lorentzian function
    center0 and center1 : scalar
        the position of the Lorentzian peak
    sigma0 and sigma1 : scalar
        variance of Gaussian function. FWHM = 2.35482 sigma, FWTM = 4.29193 sigma
    area0 and area1 : scalar
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
    ---------
    https://en.wikipedia.org/wiki/Gaussian_function

    """
    return (
        offset
        + area0
        / (sigma0 * np.sqrt(2 * np.pi))
        * np.exp(-0.5 * (x - center0) ** 2 / sigma0**2)
        + +area1
        / (sigma1 * np.sqrt(2 * np.pi))
        * np.exp(-0.5 * (x - center1) ** 2 / sigma1**2)
    )


def estimatedualGaussFit(
    datax=None, datay=None, smooth=False, smoothlevel=1, debug=False, verbose=False
):
    fitparas = estimatedualLorzfit(
        datax=datax,
        datay=datay,
        smooth=smooth,
        smoothlevel=smoothlevel,
        debug=debug,
        verbose=verbose,
    )
    fitparas[1] /= 2.35482
    fitparas[4] /= 2.35482
    return fitparas


def PolyEven(x, C0, C2, C4, C6, C8, C10, center, verbose=False):
    """
    Return the value of the polynomial
        C0 + C2 * (x-center)^2 + C4 * (x-center)^4 + C6 * (x-center)^6 + C8 * (x-center)^8 + C10 * (x-center)^10

    Parameters
    ----------
    x : scalar or array_like
        argument of the polynomial


    C0, C2, C4, C6, C8  : scalar

    verbose : bool
        the option for displaying assistive information

    Returns
    -------
    the value of the polynomial : scalar or array_like

    Examples
    --------
    >>>

    Reference
    ---------
    Null

    """
    return (
        C0
        + C2 * (x - center) ** 2
        + C4 * (x - center) ** 4
        + C6 * (x - center) ** 6
        + C8 * (x - center) ** 8
        + C10 * (x - center) ** 10
    )


def estimatePolyEvenfit(datax=None, datay=None, verbose=False):
    """
    Return the estimated parameters as the initial guess for even-oder polynomial curve fitting

    Parameters
    ----------

    datax : array_like
        The independent variable where the data is measured. Should usually be an M-length sequence.
    datay : array_like
        The dependent data, a length M array - nominally f(xdata, ...).
    verbose : bool
        choose True to display processing information


    Returns
    -------
    the value of the even-oder polynomial function : ndarray or scalar

    Examples
    --------
    >>>

    Reference
    ---------
        scipy.optimize.curve_fit
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html

    """
    center = datax[len(datax) // 2]
    C0 = datay[center]
    C2 = (datay[-1] - C0) / (datax[-1] - datax[center]) ** 2
    C4 = 0
    C6 = 0
    C8 = 0
    C10 = 0
    return [C0, C2, C4, C6, C8, C10, center]


def ExpCos(
    t=None,
    Amp=None,
    T2=None,
    nu=None,
    phi0=None,
    offset=None,
):
    """
    Exponentially decay cos wave
    s=A*exp(-t/T2)*sin(2*pi*nu*t+phi0)+offset
    """
    return (
        Amp * np.exp(-(t - t[0]) / T2) * np.cos(2 * np.pi * nu * (t - t[0]) + phi0)
        + offset
    )


def estimateExpCos(
    t=None,
    s=None,  # signal
    Lorpopt=None,
    dmodfreq=None,
):
    # ExpCos1(
    #     t=None,  #
    #     Amp=None,  # 0
    #     T2=None,  # 1
    #     nu=None,  # 2
    #     phi0=None,  # 3
    #     offset=None,  # 4
    #     verbose=False,  # 5
    # )
    Amp = max(np.amax(np.real(s)), np.amax(np.imag(s)))
    T2 = 1 / (np.pi * Lorpopt[1])
    nu = abs(Lorpopt[0] - dmodfreq)
    phi0 = 0
    offset = 0
    return [Amp, T2, nu, phi0, offset]


def ExpCosiSin(
    t=None,
    Amp=None,
    T2=None,
    nu=None,
    phi0=None,
    offsetx=None,
    offsety=None,
):
    """
    Exponentially decay cos wave
    s=A*exp(-t/T2)*sin(2*pi*nu*t+phi0)+offset
    """
    return (
        Amp
        * np.exp(-(t - t[0]) / T2)
        * (
            np.cos(2 * np.pi * nu * (t - t[0]) + phi0)
            + 1j * np.sin(2 * np.pi * nu * (t - t[0]) + phi0)
        )
        + offsetx
        + 1j * offsety
    )


def ExpCosiSinResidual(
    params,
    t,
    s,
    # Amp,
    # T2,
    # nu,
    # phi0,
    # offsetx,
    # offsety,
):
    """
    Exponentially decay cos wave
    s=A*exp(-t/T2)*sin(2*pi*nu*t+phi0)+offset
    """
    Amp, T2, nu, phi0, offsetx, offsety = (
        params[0],
        params[1],
        params[2],
        params[3],
        params[4],
        params[5],
    )
    return np.abs(
        s
        - Amp
        * np.exp(-(t - t[0]) / T2)
        * (
            np.cos(2 * np.pi * nu * (t - t[0]) + phi0)
            + 1j * np.sin(2 * np.pi * nu * (t - t[0]) + phi0)
        )
        - offsetx
        - 1j * offsety
    )


def estimateExpCosiSin(
    t=None,
    s=None,
    Lorpopt=None,
    dmodfreq=None,
):
    # ExpCos1(
    #     t=None,  #
    #     Amp=None,  # 0
    #     T2=None,  # 1
    #     nu=None,  # 2
    #     phi0=None,  # 3
    #     offset=None,  # 4
    #     verbose=False,  # 5
    # )
    Amp = max(np.amax(np.real(s)), np.amax(np.imag(s)))
    T2 = 1 / (np.pi * Lorpopt[1])
    nu = abs(Lorpopt[0] - dmodfreq)
    phi0 = 0
    offsetx = 0
    offsety = 0
    return [Amp, T2, nu, phi0, offsetx, offsety]


def dualExpCos(
    t=None,
    Amp_0=None,
    T2_0=None,
    nu_0=None,
    phi0_0=None,
    Amp_1=None,
    T2_1=None,
    nu_1=None,
    phi0_1=None,
    offset=None,
    verbose=False,
):
    """
    Two Exponentially decay cos waves

    """
    return (
        Amp_0
        * np.exp(-(t - t[0]) / T2_0)
        * np.cos(2 * np.pi * nu_0 * (t - t[0]) + phi0_0)
        + Amp_1
        * np.exp(-(t - t[0]) / T2_1)
        * np.cos(2 * np.pi * nu_1 * (t - t[0]) + phi0_1)
        + offset
    )


def estimatedualExpCos(
    t=None,
    s=None,
    Lorpopt=None,
    dmodfreq=None,
):
    # ExpSin1(
    #     t=None,  #
    #     Amp=None,  # 0
    #     T2=None,  # 1
    #     nu=None,  # 2
    #     phi0=None,  # 3
    #     offset=None,  # 4
    #     verbose=False,  # 5
    # )
    Amp = max(np.amax(np.real(s)), np.amax(np.imag(s)))
    T2 = 1 / (np.pi * Lorpopt[1])
    nu = abs(Lorpopt[0] - dmodfreq)
    phi0 = 0
    offset = 0
    return [Amp / 2, T2, nu, phi0, Amp / 2, T2, abs(nu - 3), phi0, offset]


def tribExpCos(
    t=None,
    Amp_0=None,
    T2_0=None,
    nu_0=None,
    phi0_0=None,
    Amp_1=None,
    T2_1=None,
    nu_1=None,
    phi0_1=None,
    Amp_2=None,
    T2_2=None,
    nu_2=None,
    phi0_2=None,
    offset=None,
    verbose=False,
):
    """
    Three Exponentially decay sin waves

    """
    return (
        Amp_0 * np.exp(-t / T2_0) * np.sin(2 * np.pi * nu_0 * t + phi0_0)
        + Amp_1 * np.exp(-t / T2_1) * np.sin(2 * np.pi * nu_1 * t + phi0_1)
        + Amp_2 * np.exp(-t / T2_2) * np.sin(2 * np.pi * nu_2 * t + phi0_2)
        + offset
    )


def expdecaywindow(num: int, decayfactor: float = 0.003, verbose=False):
    """
    Returns a exponentially-decaying array for the windowing

    Parameters
    ----------
    num : int
        Signal from lock-in amplifier and also the processed data
    decayfactor : float
        factor for deciding the decaying rate.
        optimal decayfactor should be chosen as 1/(T2star * samprate)
    verbose : bool
        Choose True to display processing information. Defaults to False.

    Returns
    -------
    An array for windowing

    [exp(-df*0), exp(-df*1), ... , exp(-df*(num-1))]

    df stands for decayfactor

    Examples
    --------
    >>>

    Reference
    --------
    http://sopnmr.blogspot.com/2016/01/processing-window-functions.html
    """
    if decayfactor <= 0:
        raise ValueError("decayfactor <= 0")
    n_arr = np.arange(num, dtype=float)
    index_arr = (-1) * decayfactor * n_arr
    window = np.exp(index_arr)  # (1 - np.exp(decayfactor / num)) *
    if verbose:
        plt.figure()
        plt.scatter(n_arr, window)
        plt.plot(n_arr, window, label=f"exp({decayfactor:.1f} i / {num:d})")
        plt.xlabel("Number")
        plt.ylabel("Window")
        plt.title(
            "Exponentially decay window\nsum of window values = %.2f" % np.sum(window)
        )
        plt.legend(loc="upper right")
        plt.tight_layout()
        plt.grid()
        plt.show()
    return window


def LIAFilterHomega(
    datax=None,
    datay=None,
    frequency=None,
    taun=None,
    order=None,
):
    """
    Return the complex array H(ω) for digital filter correction
        H(ω) = 1 / (1 + 2 * np.pi * 1j * frequency * taun) ** order

    Parameters
    ----------

    frequency : scalar
        Dmodualtor frequency of lock-in amplifier.
    taun : scalar
        taun equals TC (Time constant of the exponential running average filter).
    order : scalar
        Number of cascaded digital filters.
    verbose : bool
        choose True to display processing information


    Returns
    -------
    Complex array H(ω) : ndarray

    Examples
    --------
    >>>

    Reference
    ---------
        Zurich Instruments, MFIA User Manual, Page 275, 6.4.1. Discrete-Time RC Filter
        https://docs.zhinst.com/pdf/ziMFIA_UserManual.pdf

    """
    return 1.0 / (1.0 + 1j * 2.0 * np.pi * frequency * taun) ** order


def LIAFilterHomegaSquared(
    datax=None,
    datay=None,
    frequency=None,
    taun=None,
    order=None,
):
    return 1.0 / ((1.0 + (2.0 * np.pi * frequency * taun) ** 2.0) ** (order / 2.0))


def LIAFilterFunction(
    x=None,
    tau=None,
    order=None,
):
    return 1.0 / ((1.0 + 2.0 * np.pi * x * tau) ** order)  # no j


def LIAFilterHomegaSquared1(
    frequency=None,
    taun=None,
    # n=None,
    dmodfreq=None,
    a=None,
):
    return a / (
        (1.0 + (2.0 * np.pi * (frequency - dmodfreq) * taun) ** 2.0) ** (8.0 / 1.0)
    )


def LIAFilterHomegaSquared2(
    frequency=None,
    taun=None,
    # n=None,
    dmodfreq=None,
    a=None,
):
    return a / (
        (1.0 + (2.0 * np.pi * (frequency - dmodfreq) * taun) ** 2.0) ** (8.0 / 2.0)
    )


def LIAFilterPSD(frequency=None, taun=None, order=None, verbose=False):
    """
    Return the absolute value of complex array H(ω) for digital filter correction
        np.abs(H(ω)) = np.abs( 1 / (1 + 2 * np.pi * 1j * frequency * taun) ** order )

    Parameters
    ----------

    frequency : scalar
        Dmodualtor frequency of lock-in amplifier.
    taun : scalar
        taun equals TC (Time constant of the exponential running average filter).
    order : scalar
        Number of cascaded digital filters.
    verbose : bool
        choose True to display processing information


    Returns
    -------
    Absolute value of complex array H(ω) : ndarray

    Examples
    --------
    >>>

    Reference
    ----------
        Zurich Instruments, MFIJ User Manual, Page 275, 6.4.1. Discrete-Time RC Filter
        https://docs.zhinst.com/pdf/ziMFIA_UserManual.pdf

    """
    return np.abs(1.0 / (1.0 + 1j * 2.0 * np.pi * frequency * taun) ** order) ** 2.0


def stdPSD(
    data=None,
    samprate=None,  # in Hz
    windowfunction="rectangle",  # Hanning, Hamming, Blackman
    decayfactor=-10,
    verbose=False,
):
    """
    Return the frequency bin centers and power spectral density

    Parameters
    ----------
    data : 1-D array_like
        Time-series.

    samprate : float
        Sampling rate for the time-series.

    windowfunction : str, optional
        window function for FFT.
        Available choices:
            'rectangle'
            'expdecay'
            'Hanning' or 'hanning' or 'Han' or 'han'
            'Hamming' or 'hamming'
            'Blackman'
        Defaults to 'rectangle'.

    decayfactor : float
        parameter for window function 'expdecay'.
        The 'expdecay' window function generates an array of [exp(-df*0), exp(-df*1), ... , exp(-df*(num-1))],
        where df stands for decayfactor.
        Defaults to -10.

    verbose : bool
        Choose True to display processing information. Defaults to False.

    Returns
    -------
    np.sort(frequencies)[len(frequencies)//2:] : 1-D array
        Frequency bin centers sorted by its values. Only the right half of the bin centers will be passed.

    (PSD[np.argsort(frequencies)])[len(frequencies)//2:] : 1-D array
        Amplitudes of the signal in each frequency bin sorted by the value of frequency values.
        Only the right half of the array will be passed.

    Examples
    --------
    >>>

    Reference
    ---------

    [1] FFT and PSD computed based on https://holometer.fnal.gov/GH_FFT.pdf

    """

    # Generate window dictionary
    window_dict = {
        "rectangle".upper(): [np.ones],
        "expdecay".upper(): [
            partial(expdecaywindow, decayfactor=decayfactor, verbose=verbose)
        ],
        "Hanning".upper(): [np.hanning],
        "Han".upper(): [np.hanning],
        "Hamming".upper(): [np.hamming],
        "Hamm".upper(): [np.hamming],
        "Blackman".upper(): [np.blackman],
    }
    # Generate window array of the length of time-series
    window_arr = window_dict[windowfunction.upper()][0](len(data))

    # Compute S1 and S2. See Ref. [1]
    # S1 = np.sum(window_arr)
    S2 = np.sum(window_arr**2)
    if verbose:
        print(f"S2 = {S2:g}")

    # Compute frequency axis from time-series length and sampling rate
    frequencies = np.fft.fftfreq(
        len(data), d=1.0 / samprate
    )  # Set d to dwell time in s

    # initialize filter compensation array
    filtercomp = np.ones(frequencies.shape)

    # FFT and PSD
    FFT = np.fft.fft(data * window_arr, norm=None)
    PSD = 2.0 * np.abs(FFT / filtercomp) ** 2 / (S2 * samprate)

    return (
        np.sort(frequencies)[len(frequencies) // 2 :],
        (PSD[np.argsort(frequencies)])[len(frequencies) // 2 :],
    )


def stdLIAPSD(
    data_x: np.ndarray = None,
    data_y: np.ndarray = None,
    samprate: float = None,
    demodfreq: float = None,
    attenuation: str = 0,
    windowfunction: str = "rectangle",
    decayfactor: float = -10.0,
    showwindow=False,
    DTRCfilter: str = "off",
    DTRCfilter_TC: float = 1e-6,
    DTRCfilter_order: float = 8,
    verbose: bool = False,
):
    """
    Return the frequency bin centers and power spectral density
        This function processes data from MFIA lock-in amplifier.

    Parameters
    ----------
    data_x, data_y : 1-D array_like
        Time-series data of two output channels.

    samprate : float
        Sampling rate for the time-series in [Hz].

    dfreq : float
        Demodulator frequency of the lock-in amplifier in [Hz].
        Defaults to None.

    attenuation : float
        Attenuation of the data in terms of power ratio (in the unit of dB).
        Power ratio (10^(attenuation/10)).
        Positive value means signal was attenuated.
        e.g. an attenuation of 6 means 10^(6/10) = 3.981 ≈ 4
        Defaults to None.

    windowfunction : str, optional
        window function for FFT.
        Available choices:
            'rectangle'
            'expdecay'
            'Hanning' or 'hanning' or 'Han' or 'han'
            'Hamming' or 'hamming'
            'Blackman'
        Defaults to 'rectangle'.

    decayfactor : float, optional
        parameter for window function 'expdecay'.
        The 'expdecay' window function generates an array of [exp(-df*0), exp(-df*1), ... , exp(-df*(num-1))],
        where df stands for decayfactor.
        Defaults to -10.

    showwindow : bool, optional
        option to plot window function array.
        Defaults to False.

    DTRCfilter : string, optional
        Whether the discrete time RC filter is on.
        Defaults to 'on'.

    DTRCfilter_TC : float, optional
        Time constant of the filter.
        Defaults to 1e-6.

    DTRCfilter_order : int / float, optional
        Order of the filter.
        Defaults to 8.

    verbose : bool, optional
        Choose True to display processing information.
        Defaults to False.

    Returns
    -------
    np.sort(frequencies) : 1-D array
        Absolute frequency bin centers sorted by its values.

    PSD[np.argsort(frequencies)] : 1-D array
        Amplitudes of the signal in each frequency bin sorted by the value of frequency values.

    Examples
    --------
    >>>


    Reference
    --------
    [1] FFT and PSD computed based on https://holometer.fnal.gov/GH_FFT.pdf

    [2] Zurich Instruments, MFIJ User Manual 500 kHz / 5 MHz Impedance Analyzer
        P185 6.4. Discrete-Time Filters
        https://docs.zhinst.com/pdf/ziMFIA_UserManual.pdf

    """

    # check array lengths
    assert len(data_x) == len(data_y)

    # Generate window dictionary
    window_dict = {
        "rectangle".upper(): [np.ones],
        "expdecay".upper(): [
            partial(expdecaywindow, decayfactor=decayfactor, verbose=verbose)
        ],
        "Hanning".upper(): [np.hanning],
        "Han".upper(): [np.hanning],
        "Hamming".upper(): [np.hamming],
        "Hamm".upper(): [np.hamming],
        "Blackman".upper(): [np.blackman],
    }
    # Generate window array of the length of time-series
    window_arr = window_dict[windowfunction.upper()][0](len(data_x))

    # Compute S1 and S2. See Ref. [1]
    # S1 = np.sum(window_arr)
    S2 = np.sum(window_arr**2)
    if verbose:
        print(f"S2 = {S2:g}")

    # Compute frequency axis from time-series length and sampling rate
    frequencies: np.ndarray = np.fft.fftfreq(
        len(data_x), d=1.0 / samprate
    )  # Set d to dwell time in s

    # initialize filter compensation array
    filtercomp = np.ones(frequencies.shape)
    if DTRCfilter == "on":
        filtercomp = LIAFilterHomega(
            frequency=frequencies,  # this frequency is [-samprate/2, samprate/2]
            taun=DTRCfilter_TC,
            order=DTRCfilter_order,
        )
    # check(filtercomp)

    # add demodulator frequency to get absolute frequency
    # Notice that this action should be not be done before LIAFilterHomega()
    # because this function accepts demodulated frequencies
    frequencies += demodfreq  #

    # show window array
    if showwindow:
        xstamp = np.arange(len(data_x))
        fig = plt.figure(figsize=(12, 5))  #
        gs = gridspec.GridSpec(nrows=1, ncols=2)
        # fig.subplots_adjust(top=0.91,bottom=0.11,left=0.08,right=0.96,hspace=0.0,wspace=0.25)
        LIAraw_ax = fig.add_subplot(gs[0, 0])
        # LIAYraw_ax = fig.add_subplot(gs[1, 0])
        LIAwindowed_ax = fig.add_subplot(gs[0, 1])
        # LIAYwindowed_ax = fig.add_subplot(gs[1, 1])
        (LIAXrawline,) = LIAraw_ax.plot(
            xstamp, data_x, label="LIA X output", color="tab:green", alpha=1
        )
        (LIAYrawline,) = LIAraw_ax.plot(
            xstamp, data_y, label="LIA Y output", color="tab:brown", alpha=1
        )
        #'tab:brown'
        LIAraw_ax.set_ylabel("Amplitude / a.u.")
        # LIAraw_ax.legend(loc='upper right')
        LIAraw_ax.grid(True)
        # LIAraw_ax.tick_params(axis='y', left=False, labelleft=False)
        # LIAraw_ax.tick_params(axis='x',bottom=False, labelbottom=False)
        LIAraw_ax2 = LIAraw_ax.twinx()
        (windowline,) = LIAraw_ax2.plot(
            xstamp, window_arr, label="Window", color="tab:blue"
        )
        LIAraw_ax2.set_ylabel("Window value")
        # LIAraw_ax2.legend(loc='upper right')
        # line1, = ax.plot([1, 2, 3], label='label1')
        # line2, = ax.plot([1, 2, 3], label='label2')
        LIAraw_ax2.legend(handles=[LIAXrawline, LIAYrawline, windowline])
        # bottom, top, left, right : bool : Whether to draw the respective ticks
        # labelbottom, labeltop, labelleft, labelright : bool : Whether to draw the respective tick labels.
        # imag_ax.plot(self.frequencies, self.avgFFT.imag, \
        #     label='Imaginary part of FFT', color='tab:orange')
        # imag_ax.set_ylabel('Amplitude / a.u.')
        # imag_ax.set_xlabel('Frequency / Hz')
        # imag_ax.legend(loc='upper right')
        # imag_ax.grid(True)
        # # imag_ax.tick_params(axis='y', left=False, labelleft=False)

        (LIAXwinline,) = LIAwindowed_ax.plot(
            xstamp,
            data_x * window_arr,
            label="LIA X windowed",
            color="tab:green",
            alpha=1,
        )
        (LIAYwinline,) = LIAwindowed_ax.plot(
            xstamp,
            data_y * window_arr,
            label="LIA Y windowed",
            color="tab:brown",
            alpha=1,
        )
        LIAwindowed_ax.set_ylabel("Amplitude / a.u.")
        # LIAwindowed_ax.legend(loc='upper right')
        LIAwindowed_ax.grid(True)
        # LIAwindowed_ax.tick_params(axis='y', left=False, labelleft=False)
        # LIAwindowed_ax.tick_params(axis='x', bottom=False, labelbottom=False)
        LIAwindowed_ax2 = LIAwindowed_ax.twinx()
        (windowline,) = LIAwindowed_ax2.plot(
            xstamp, window_arr, label="Window", color="tab:blue"
        )
        LIAwindowed_ax2.set_ylabel("Window value")
        LIAwindowed_ax2.legend(handles=[LIAXwinline, LIAYwinline, windowline])

        # LIAwindowed_ax2.legend(loc='upper right')
        # phase_ax.plot(self.frequencies, np.angle(self.avgFFT, deg=True), \
        #     label='Phase of  FFT', color='tab:cyan')
        # phase_ax.set_ylabel('Phase / $\degree$')
        # phase_ax.set_xlabel('Frequency / Hz')
        # phase_ax.legend(loc='upper right')
        # phase_ax.grid(True)

        # if specxlim != None:
        #     LIAraw_ax.set_xlim(specxlim[0], specxlim[1])
        #     imag_ax.set_xlim(specxlim[0], specxlim[1])
        #     amp_ax.set_xlim(specxlim[0], specxlim[1])
        #     phase_ax.set_xlim(specxlim[0], specxlim[1])

        titletext = "Window array"  # 'All shots of '+
        fig.suptitle(titletext)  # , fontsize=8
        plt.tight_layout()
        plt.show()

    # FFT and PSD
    FFT = np.fft.fft((data_x + 1j * data_y) * window_arr, norm=None)
    PSD = (
        10.0 ** (attenuation / 10.0)
        * 1.0
        * np.abs(FFT / filtercomp) ** 2.0
        / (S2 * samprate)
    )

    # to check Parsvel theorem, use:
    # TSPower == np.mean(PSD) * samprate

    if verbose:
        print("attenuation ", attenuation)
        print("FFT.shape ", FFT.shape)
        print("filtercomp.shape ", filtercomp.shape)
        print("S2 ", S2)
        print("samprate ", samprate)

    return np.sort(frequencies), PSD[np.argsort(frequencies)]
    # return frequencies, PSD
    # return np.sort(frequencies), PSD


def stdLIAFFT(
    data_x: np.ndarray = None,
    data_y: np.ndarray = None,
    samprate: float = None,
    demodfreq: float = None,
    attenuation: str = None,
    windowfunction: str = "rectangle",
    decayfactor: float = -10.0,
    showwindow=False,
    DTRCfilter: str = "off",
    DTRCfilter_TC: float = 1e-6,
    DTRCfilter_order: float = 8,
    verbose: bool = False,
):
    """
    Return the frequency bin centers and FFT results.
        This function processes data from MFIA lock-in amplifier.

    Parameters
    ----------
    data_x, data_y : 1-D array_like
        Time-series data of two output channels.

    samprate : float
        Sampling rate for the time-series in [Hz].

    dfreq : float
        Demodulator frequency of the lock-in amplifier in [Hz].
        Defaults to None.

    attenuation : float
        Attenuation of the data in terms of power ratio (in the unit of dB).
        Power ratio (10^(attenuation/10)).
        Positive value means signal was attenuated.
        e.g. an attenuation of 6 means 10^(6/10) = 3.981 ≈ 4
        Defaults to None.

    windowfunction : str, optional
        window function for FFT.
        Available choices:
            'rectangle'
            'expdecay'
            'Hanning' or 'hanning' or 'Han' or 'han'
            'Hamming' or 'hamming'
            'Blackman'
        Defaults to 'rectangle'.

    showwindow : bool, optional
        option to plot window function array.
        Defaults to False.

    DTRCfilter : string, optional
        Whether the discrete time RC filter is on.
        Defaults to 'on'.

    DTRCfilter_TC : float, optional
        Time constant of the filter.
        Defaults to 1e-6.

    DTRCfilter_order : int / float, optional
        Order of the filter.
        Defaults to 8.

    verbose : bool, optional
        Choose True to display processing information.
        Defaults to False.

    Returns
    -------
    np.sort(frequencies) : 1-D array
        Absolute frequency bin centers sorted by its values.

    FFT[np.argsort(frequencies)] : 1-D array
        Amplitudes of the signal in each frequency bin sorted by the value of frequency values.

    Examples
    --------
    >>>


    Reference
    --------
    [1] FFT and PSD computed based on https://holometer.fnal.gov/GH_FFT.pdf

    [2] Zurich Instruments, MFIJ User Manual 500 kHz / 5 MHz Impedance Analyzer
        P185 6.4. Discrete-Time Filters
        https://docs.zhinst.com/pdf/ziMFIA_UserManual.pdf

    """

    # check array lengths
    assert len(data_x) == len(data_y)

    # Generate window dictionary
    window_dict = {
        "rectangle".upper(): [np.ones],
        "expdecay".upper(): [
            partial(expdecaywindow, decayfactor=decayfactor, verbose=verbose)
        ],
        "Hanning".upper(): [np.hanning],
        "Han".upper(): [np.hanning],
        "Hamming".upper(): [np.hamming],
        "Hamm".upper(): [np.hamming],
        "Blackman".upper(): [np.blackman],
    }
    # Generate window array of the length of time-series
    window_arr = window_dict[windowfunction.upper()][0](len(data_x))

    # Compute S1 and S2. See Ref. [1]
    # S1 = np.sum(window_arr)
    S2 = np.sum(window_arr**2)
    if verbose:
        print(f"S2 = {S2:g}")

    # Compute frequency axis from time-series length and sampling rate
    frequencies = np.fft.fftfreq(
        len(data_x), d=1.0 / samprate
    )  # Set d to dwell time in s

    # initialize filter compensation array
    filtercomp = np.ones(frequencies.shape)
    if DTRCfilter == "on":
        filtercomp = LIAFilterHomega(
            frequency=frequencies,  # this frequency is [-samprate/2, samprate/2]
            taun=DTRCfilter_TC,
            order=DTRCfilter_order,
        )
    # check(filtercomp)

    # show window array
    if showwindow:
        xstamp = np.arange(len(data_x))
        fig = plt.figure(figsize=(12, 5))  #
        gs = gridspec.GridSpec(nrows=1, ncols=2)
        # fig.subplots_adjust(top=0.91,bottom=0.11,left=0.08,right=0.96,hspace=0.0,wspace=0.25)
        LIAraw_ax = fig.add_subplot(gs[0, 0])
        # LIAYraw_ax = fig.add_subplot(gs[1, 0])
        LIAwindowed_ax = fig.add_subplot(gs[0, 1])
        # LIAYwindowed_ax = fig.add_subplot(gs[1, 1])
        (LIAXrawline,) = LIAraw_ax.plot(
            xstamp, data_x, label="LIA X output", color="tab:green", alpha=1
        )
        (LIAYrawline,) = LIAraw_ax.plot(
            xstamp, data_y, label="LIA Y output", color="tab:brown", alpha=1
        )
        #'tab:brown'
        LIAraw_ax.set_ylabel("Amplitude / a.u.")
        # LIAraw_ax.legend(loc='upper right')
        LIAraw_ax.grid(True)
        # LIAraw_ax.tick_params(axis='y', left=False, labelleft=False)
        # LIAraw_ax.tick_params(axis='x',bottom=False, labelbottom=False)
        LIAraw_ax2 = LIAraw_ax.twinx()
        (windowline,) = LIAraw_ax2.plot(
            xstamp, window_arr, label="Window", color="tab:blue"
        )
        LIAraw_ax2.set_ylabel("Window value")
        # LIAraw_ax2.legend(loc='upper right')
        # line1, = ax.plot([1, 2, 3], label='label1')
        # line2, = ax.plot([1, 2, 3], label='label2')
        LIAraw_ax2.legend(handles=[LIAXrawline, LIAYrawline, windowline])
        # bottom, top, left, right : bool : Whether to draw the respective ticks
        # labelbottom, labeltop, labelleft, labelright : bool : Whether to draw the respective tick labels.
        # imag_ax.plot(self.frequencies, self.avgFFT.imag, \
        #     label='Imaginary part of FFT', color='tab:orange')
        # imag_ax.set_ylabel('Amplitude / a.u.')
        # imag_ax.set_xlabel('Frequency / Hz')
        # imag_ax.legend(loc='upper right')
        # imag_ax.grid(True)
        # # imag_ax.tick_params(axis='y', left=False, labelleft=False)

        (LIAXwinline,) = LIAwindowed_ax.plot(
            xstamp,
            data_x * window_arr,
            label="LIA X windowed",
            color="tab:green",
            alpha=1,
        )
        (LIAYwinline,) = LIAwindowed_ax.plot(
            xstamp,
            data_y * window_arr,
            label="LIA Y windowed",
            color="tab:brown",
            alpha=1,
        )
        LIAwindowed_ax.set_ylabel("Amplitude / a.u.")
        # LIAwindowed_ax.legend(loc='upper right')
        LIAwindowed_ax.grid(True)
        # LIAwindowed_ax.tick_params(axis='y', left=False, labelleft=False)
        # LIAwindowed_ax.tick_params(axis='x', bottom=False, labelbottom=False)
        LIAwindowed_ax2 = LIAwindowed_ax.twinx()
        (windowline,) = LIAwindowed_ax2.plot(
            xstamp, window_arr, label="Window", color="tab:blue"
        )
        LIAwindowed_ax2.set_ylabel("Window value")
        LIAwindowed_ax2.legend(handles=[LIAXwinline, LIAYwinline, windowline])

        # LIAwindowed_ax2.legend(loc='upper right')
        # phase_ax.plot(self.frequencies, np.angle(self.avgFFT, deg=True), \
        #     label='Phase of  FFT', color='tab:cyan')
        # phase_ax.set_ylabel('Phase / $\degree$')
        # phase_ax.set_xlabel('Frequency / Hz')
        # phase_ax.legend(loc='upper right')
        # phase_ax.grid(True)

        # if specxlim != None:
        #     LIAraw_ax.set_xlim(specxlim[0], specxlim[1])
        #     imag_ax.set_xlim(specxlim[0], specxlim[1])
        #     amp_ax.set_xlim(specxlim[0], specxlim[1])
        #     phase_ax.set_xlim(specxlim[0], specxlim[1])

        titletext = "Window array"  # 'All shots of '+
        fig.suptitle(titletext)  # , fontsize=8
        plt.tight_layout()
        plt.show()

    # FFT and PSD
    FFT = np.fft.fft((data_x + 1j * data_y) * window_arr, norm=None)
    # PSD = 10.0 ** (attenuation / 10.) * 1.0 * np.abs(FFT / filtercomp) ** 2. / (S2 * samprate)
    FFT = (
        10.0 ** (attenuation / 20.0) * 1.0 * (FFT / filtercomp) / (S2 * samprate) ** 0.5
    )
    frequencies += demodfreq  #
    return np.sort(frequencies), FFT[np.argsort(frequencies)]


def DTRC_filter(
    signal,
    samprate: float,
    TC: float,
    order: int,
):
    """
    Discrete-Time RC Filter

    Reference
    --------
    [1] Zurich Instruments, MFIJ User Manual 500 kHz / 5 MHz Impedance Analyzer
        P185 6.4. Discrete-Time Filters
        https://docs.zhinst.com/pdf/ziMFIA_UserManual.pdf

    """
    Ts = 1.0 / samprate

    # signal_f = signal
    def DTRC_filter_1st(signal):
        sigal_f = np.exp(-Ts / TC) * signal[0:-1] + (1 - np.exp(-Ts / TC)) * signal[1:]
        return sigal_f

    for i in range(order):
        signal = DTRC_filter_1st(signal)

    return signal


def plotaxisfmt(x, y, format_string):
    return format_string.format(x)


def plotaxisfmt_ppm2MHz(x, y, format_string, referfreq):
    return format_string.format(1e-6 * (1e-6 * referfreq * x + referfreq))


def plotaxisfmt_ppm2Hz(x, y, format_string, referfreq):
    return format_string.format((1e-6 * referfreq * x + referfreq))


def plotaxisfmt_Hz2ppm(x, y, format_string, referfreq):
    return format_string.format(1e6 * (x / referfreq - 1))


def plotaxisfmt_MHz2ppm(x, y, format_string, referfreq):
    return format_string.format(1e6 * (1e6 * x / referfreq - 1))


def plotaxisfmt_linewidth2ppm(x, y, format_string, referfreq):
    return format_string.format(1e6 * (x / referfreq))


def axisfmt_C2K(x, y, format_string):
    return format_string.format(x + 273.15)


def axisfmt_K2C(x, y, format_string):
    return format_string.format(x - 273.15)


def MethanolCS2temp(
    CSval=None,
    CSunit="ppm",  # 'ppm' 'Hz'
    referfreq=1e6,  # in Hz
    tempunit="K",
):

    a = -23.832
    b = -29.46
    c = 403.0
    temp = a * CSval**2 + b * CSval + c
    if tempunit == "K":
        return temp
    elif tempunit == "C":
        return temp - 273.15
    else:
        raise ValueError("tempunit wrong")


def Methanoltemp2CS(
    temp=None,
    tempunit="C",
    CSunit="ppm",  # 'ppm' 'Hz'
    referfreq=1e6,  # in Hz
):

    a = -23.832
    b = -29.46
    c = 403.0
    # temp_arr = a * CS_arr**2 + b * CS_arr + c
    if tempunit == "C":
        CSval = (-b - np.sqrt(b**2 - 4 * a * (c - temp - 273.15))) / (2 * a)
    elif tempunit == "K":
        CSval = (-b - np.sqrt(b**2 - 4 * a * (c - temp))) / (2 * a)
    else:
        raise ValueError("tempunit wrong")

    if CSunit == "ppm":
        pass
    elif CSunit == "Hz":
        CSval *= referfreq
    else:
        raise ValueError("CSunit wrong")
    return CSval


def MethanolChemicalShift(
    temprange=[0, -90],
    temparrnum=1000,
    plotxunit="C",
    plotxunit2="K",
    plotyunit="ppm",  # 'Hz' 'kHz' 'MHz' 'GHz' 'ppm' 'ppb'
    plotyunit2="Hz",
    referencefreq=1.349500e6,  # default 1349500 Hz
    plotx2format="{:.3f}",
    # formatter = mticker.FuncFormatter(lambda y, pos: '{:.3f}'.format(specxaxis2))
    # ampunit='Phi',  #  'muV' 'muPhi' 'Phi' 'V'
    plotyscale="linear",  # 'log', 'linear'
    specxlim=[],
    specylim=[],
    top=0.9,
    bottom=0.1,
    left=0.1,
    right=0.93,
    hspace=0.4,
    wspace=0.295,
    vlinex=None,
    showplt_opt=True,
    saveplt_opt=False,
    saveSpecData_opt=False,
    save_path=None,
    save_msg=None,
    return_opt=False,
    verbose=False,
):
    """
    # T = 403.0 - 0.491 deltanu -66.2 (10^-2 deltanu)^2
    175 - 330 K
    Calibration of Methanol Nuclear Magnetic Resonance Thermometerat Low Temperature
    ANALYTICAL CHEMISTRY, VOL. 42, NO. 6, MAY 1970
    Anthony L. Van Geet

    A. L. Van Geet, Anal. Chem., 40, 2227 (1968).
    Calibration of the Methanol and Glycol Nuclear Magnetic Resonance Thermometers with a Static Thermistor Probe
    Anthony L. Van Geet
    Department of Chemistry, State University of New York, Buffalo, N. Y. 14214

    https://www.versci.com/cwnmr/index.html
    """
    temp_arr = np.linspace(
        start=temprange[0],
        stop=temprange[1],
        num=temparrnum,
        endpoint=True,
        dtype=float,
    )
    CS_arr = Methanoltemp2CS(temp=temp_arr)

    # a = -66.2*1e-4
    # b = -0.491
    # c = 403.0
    plt.rc("font", size=14)
    fig = plt.figure(figsize=(8, 6))  #
    gs = gridspec.GridSpec(nrows=1, ncols=1)  #
    fig.subplots_adjust(
        left=left, top=top, right=right, bottom=bottom, wspace=wspace, hspace=hspace
    )
    CStemp_ax = fig.add_subplot(gs[0, 0])
    CStemp_ax.plot(temp_arr, CS_arr)
    CStemp_ax.scatter(temp_arr, CS_arr)
    CStemp_ax.set_xlabel("Temperature / $\degree C$")
    CStemp_ax.set_ylabel("Chemical shift /ppm")
    # peakamp_ax_ylim = peakamp_ax.get_ylim()
    CStemp_ax.set_ylim(bottom=0, top=1.2 * (CStemp_ax.get_ylim()[1]))

    CStemp_ax2 = CStemp_ax.twiny()
    CStemp_ax2.set_xlabel("Temperature / K")
    CStemp_ax2.set_xlim(CStemp_ax.get_xlim())
    print(CStemp_ax.get_xlim())
    # apply a function formatter
    axisfmt_partial = partial(axisfmt_C2K, format_string="{:.0f}")
    formatter = mticker.FuncFormatter(axisfmt_partial)
    CStemp_ax2.xaxis.set_major_formatter(formatter)

    if showplt_opt:
        plt.show()
    plt.close()
    return 0


def constOne(t=None):
    return 1


def statUni2Pi(num=None, showplt=False, verbose=False):
    """
    return an array of N numbers which obeys Uniform distribution
    Parameters
    ----------

    num : int
    the length of return array. None by default.

    showplt : bool
    Whether to plot histogram of the distribution

    verbose : bool

    Examples
    --------
    >>>

    Reference
    ----------

    """
    r = uniform.rvs(loc=0, scale=2 * np.pi, size=num)
    if verbose:
        check(r)
    if showplt:
        fig = plt.figure(figsize=(4, 3), dpi=150)  #
        gs = gridspec.GridSpec(nrows=1, ncols=1)  #
        # fig.subplots_adjust(left=left_spc, top=top_spc, right=right_spc,
        #                     bottom=bottom_spc, wspace=xgrid_spc, hspace=ygrid_spc)
        # make plot for time domain signal
        ax = fig.add_subplot(gs[0, 0])
        ax.hist(
            r, density=False, histtype="stepfilled", alpha=0.9, range=(0, 2 * np.pi)
        )
        # ax.legend(loc='best', frameon=False)
        ax.set_ylabel("Count")
        ax.set_title("Uniform distribution [0, 2*Pi]")
        plt.show()
    return r


def fRayleigh(x, sigma, verbose=False):
    stepx = (x + abs(x)) / 2.0
    return stepx / sigma**2 * np.exp(-(stepx**2) / 2 / sigma**2)


def statRayleigh(sigma=None, num=None, showplt=False, verbose=False):
    """
    return an array of N numbers which obeys Rayleigh distribution

    Parameters
    ----------
    sigma : float

    parameter for the distribution. None by default.

    num : int

    the length of return array. None by default.

    showplt : bool

    Whether to plot histogram of the distribution

    verbose : bool

    Examples
    --------

    Exampl 1
    import numpy as np
    import matplotlib.pyplot as plt
    from functioncache import *
    r0=statRayleigh(
            sigma=3,
            num=1000,
            showplt=False,
            verbose=False
        )
    r1=statRayleigh(
            sigma=3,
            num=1000,
            showplt=False,
            verbose=False
        )
    fig = plt.figure(dpi=150)
    gs = gridspec.GridSpec(nrows=1, ncols=1)
    # fig.subplots_adjust(top=0.88,bottom=0.11,left=0.07,right=0.99,hspace=0.2,wspace=0.2)
    ax = fig.add_subplot(gs[0,0])
    ax.hist(r0,density=False, histtype='step', label='r0', alpha=0.9)
    ax.hist(r1,density=False, histtype='step', label='r1', alpha=0.9)
    ax.hist(r0+r1,density=False, histtype='step', label='r0+r1', alpha=0.9)
    ax.hist(0.25*r0+0.75*r1,density=False, histtype='step', label='0.25*r0+0.75*r1', alpha=0.9)
    ax.legend(loc='best', frameon=True, fontsize=8)
    ax.set_ylabel(f'Count')
    ax.set_xlabel(f'Amplitude')
    ax.grid()
    plt.show()

    Example 2

    import numpy as np
    import matplotlib.pyplot as plt
    from functioncache import *
    sigma=3
    lenofarray=2000
    r0=statRayleigh(
            sigma=sigma,
            num=lenofarray,
            showplt=False,
            verbose=False
        )
    r1=statRayleigh(
            sigma=sigma,
            num=lenofarray,
            showplt=False,
            verbose=False
        )
    rsum = np.sqrt(r0**2+r1**2)/ np.sqrt(2)
    g0=statGaussian(
            mu=0,
            sigma=sigma,
            num=lenofarray,
            showplt=False,
            verbose=False
        )
    g1=statGaussian(
            mu=0,
            sigma=sigma,
            num=lenofarray,
            showplt=False,
            verbose=False
        )
    g2=statGaussian(
            mu=0,
            sigma=sigma,
            num=lenofarray,
            showplt=False,
            verbose=False
        )
    r1 = np.sqrt(g0**2+g1**2)
    r2 = np.sqrt(g0**2 + g1**2 + g2**2)
    fig = plt.figure(dpi=150)  #figsize=(10, 4),
    gs = gridspec.GridSpec(nrows=1, ncols=1)  #
    ax = fig.add_subplot(gs[0,0])
    ax.hist(r0,density=True, histtype='step', label='r0', alpha=0.9)
    ax.hist(rsum,density=True, histtype='step', label='np.sqrt(r0**2+r1**2)', alpha=0.9)
    ax.hist(r1,density=True, histtype='step', label='np.sqrt(g0**2+g1**2)', alpha=0.9)
    ax.hist(r2,density=True, histtype='step', label='r2', alpha=0.9)
    ax.legend(loc='best', frameon=True, fontsize=8)
    ax.set_ylabel(f'Count')
    ax.set_xlabel(f'')
    ax.grid()
    plt.show()

    Notes
    -----
    Rayleigh distribution f(x;sigma) = x/sigma^2 * exp(-x^2/(2 sigma^2))

    Mean value: (pi/2)^0.5 sigma

    standard deviation: (2-pi/2)^0.5 sigma



    Reference
    ----------
    https://en.wikipedia.org/wiki/Rayleigh_distribution

    https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rayleigh.html
    """
    r = rayleigh.rvs(loc=0, scale=sigma, size=num)
    if verbose:
        check(r)
    if showplt:
        fig = plt.figure(figsize=(4, 3), dpi=150)  #
        gs = gridspec.GridSpec(nrows=1, ncols=1)  #
        # fig.subplots_adjust(left=left_spc, top=top_spc, right=right_spc,
        #                     bottom=bottom_spc, wspace=xgrid_spc, hspace=ygrid_spc)
        # make plot for time domain signal
        ax = fig.add_subplot(gs[0, 0])
        ax.hist(
            r, density=False, histtype="stepfilled", alpha=0.9, range=(0, np.amax(r))
        )
        # ax.legend(loc='best', frameon=False)
        ax.set_ylabel("Count")
        ax.set_title(f"Rayleigh distribution sigma={sigma}")
        plt.show()
    return r


def randomwalk2D(sigma=None, numofstep=None, num=None, showplt=False, verbose=False):
    """
    Random walk on 2D plane

    """
    r = rayleigh.rvs(loc=0, scale=sigma, size=num)
    rw2D = []
    for i in range(num):
        phase = uniform.rvs(loc=0, scale=2 * np.pi, size=numofstep)
        sin_arr = np.sin(phase)
        cos_arr = np.cos(phase)
        rw2D.append(np.sum(cos_arr) + 1j * np.sum(sin_arr))
    rw2D = sigma * np.array(rw2D, dtype=np.complex64) / np.sqrt(numofstep / 2.0)
    if verbose:
        check(rw2D)

    if showplt:
        fig = plt.figure(figsize=(10, 4), dpi=150)  #
        gs = gridspec.GridSpec(nrows=1, ncols=2)  #
        fig.subplots_adjust(
            top=0.88, bottom=0.11, left=0.07, right=0.99, hspace=0.2, wspace=0.2
        )
        # make plot for time domain signal
        ax = fig.add_subplot(gs[0, 0])
        ax.hist(
            r,
            bins=40,
            density=False,
            histtype="step",
            label="Rayleigh distribution",
            alpha=0.9,
            range=(0, np.amax(r)),
        )
        ax.hist(
            np.abs(rw2D.real),
            bins=40,
            density=False,
            histtype="step",
            label="Amplitude of 1D random walk - cos",
            alpha=0.9,
        )
        ax.hist(
            np.abs(rw2D.imag),
            bins=40,
            density=False,
            histtype="step",
            label="Amplitude of 1D random walk - sin",
            alpha=0.9,
        )
        ax.hist(
            np.abs(rw2D),
            bins=50,
            density=False,
            histtype="step",
            label="Amplitude of 2D random walk",
            alpha=0.9,
        )
        ax.legend(loc="best", frameon=True, fontsize=8)
        ax.set_ylabel("Count")
        ax.set_xlabel("Amplitude")
        # ax.set_title(f'Rayleigh distribution sigma={sigma}')
        ax.grid()

        ax2 = fig.add_subplot(gs[0, 1])
        # ax.hist(r, density=False, histtype='step', label='Rayleigh distribution', alpha=0.9, range=(0,np.amax(r)))
        ax2.hist(
            np.angle(rw2D) + np.pi,
            bins=40,
            density=False,
            histtype="bar",
            label="Phase of 2D random walk",
            alpha=0.9,
            range=(0, 2 * np.pi),
        )
        ax2.legend(loc="best", frameon=True)
        # ax2.set_ylabel(f'Count')
        ax2.set_xlabel("Phase")
        ax2.grid()

        plt.show()
    return rw2D


def statGaussian(mu=None, sigma=None, num=None, showplt=False, verbose=False):
    """ """
    r = norm.rvs(loc=mu, scale=sigma, size=num, random_state=None)
    if verbose:
        check(r)
    if showplt:
        fig = plt.figure(figsize=(4, 3), dpi=150)  #
        gs = gridspec.GridSpec(nrows=1, ncols=1)  #
        # fig.subplots_adjust(left=left_spc, top=top_spc, right=right_spc,
        #                     bottom=bottom_spc, wspace=xgrid_spc, hspace=ygrid_spc)
        # make plot for time domain signal
        ax = fig.add_subplot(gs[0, 0])
        ax.hist(r, density=False, histtype="stepfilled", alpha=0.9)
        # ax.legend(loc='best', frameon=False)
        ax.set_ylabel("Count")
        ax.set_title(f"Gaussian distribution sigma={sigma}")
        plt.show()
    return r


def sph2orth1d(vec=None):
    orth = np.zeros(3)
    orth[0] = vec[0] * np.sin(vec[1]) * np.cos(vec[2])
    orth[1] = vec[0] * np.sin(vec[1]) * np.sin(vec[2])
    orth[2] = vec[0] * np.cos(vec[1])
    return orth


def sph2orth(vec=None):
    if len(vec.shape) == 1:
        return sph2orth1d(vec)
    elif len(vec.shape) == 2:
        xyz = np.zeros(np.shape(vec))
        xyz[:, 0] = vec[:, 0] * np.sin(vec[:, 1]) * np.cos(vec[:, 2])
        xyz[:, 1] = vec[:, 0] * np.sin(vec[:, 1]) * np.sin(vec[:, 2])
        xyz[:, 2] = vec[:, 0] * np.cos(vec[:, 1])
        return xyz
    else:
        raise ValueError("len(vec.shape)<=0 or len(vec.shape)>=3")


def Rad2Deg(vec=None):
    r = vec
    r[:, 1:] *= 180.0 / np.pi
    return r


def Rad2Deg1d(vec=None):
    r = vec
    r[1:] *= 180.0 / np.pi
    return r


"""
def orth2sph(vec):
    x=vec[0]
    y=vec[1]
    z=vec[2]
    r=(x**2+y**2+z**2)**0.5
    if r==0:
        return [0,0,0]
    theta = np.arccos(z/r)
    (x ** 2 + y ** 2) ** 0.5
    if x>0 and y>0:
        phi = np.arcsin(y/)
        return [r, theta, phi]

    if x==0:
        phi = np.arcsin(y/(x**2+y**2)**0.5)
        return [r, theta, phi]
    else:
        phi = np.arcsin(y / (x ** 2 + y ** 2) ** 0.5)
        return [r, theta, phi]

    if y == 0:
        if x > 0:
            return [r, theta, 0]
        else:
            return [r, theta, np.pi]
"""


def orth2cyd(vec):
    x = vec[0]
    y = vec[1]
    z = vec[2]
    r = (x**2 + y**2) ** 0.5
    theta = np.arctan2(y, x)
    return np.array([r, theta, z])


def rotmatrix_axis(rotaxis=None, rotangle=None, unit="rad", verbose=False):
    """
    The matrix of a proper rotation R by angle θ around the rotating axis

    Reference
    ---------
    https   ://en.wikipedia.org/wiki/Rotation_matrix
    """
    u = np.array(rotaxis, dtype=np.float64).reshape((3))
    u = u / (np.vdot(u, u)) ** 0.5
    ux = np.array([0, -u[2], u[1]], [u[2], 0, -u[0]], [-u[1], u[0], 0])
    uxu = np.outer(u, u)
    matrixR = (
        np.cos(rotangle) * np.identity(3)
        + np.sin(rotangle) * ux
        + (1 - np.cos(rotangle)) * uxu
    )
    return matrixR


def Npole2station(
    theta_e=None,  # scalar
    phi_e=None,  # scalar
    theta_s=None,
    phi_s=None,
    verbose=False,
):
    """
    return in cartesian coordinates
    """
    x = np.sin(theta_e) * np.cos(theta_s) * np.cos(phi_e - phi_s) - np.cos(
        theta_e
    ) * np.sin(theta_s)
    y = np.sin(theta_e) * np.sin(phi_e - phi_s)
    z = np.sin(theta_e) * np.sin(theta_s) * np.cos(phi_e - phi_s) + np.cos(
        theta_e
    ) * np.cos(theta_s)
    if verbose:
        check(np.array([x, y, z]))
        check(np.vdot(np.array([x, y, z]), np.array([x, y, z])) ** 0.5)
    return np.array([x, y, z])


# @jit(void(object_, float_[:,:], int_[:], int_))
# def train_function(instance, X, y, H):
#     # do stuff
# class GentleBoostC(object):
#     def train(self, X, y, H):
#         train_function(self, X, y, H)
# @jit(nb.int16(nb.int64, nb.int64, nb.float64[:], nb.float64[:], nb.float64, \
#     nb.float64[:], nb.float64[:], nb.float64, nb.float64, nb.float64, nb.float64, nb.float64))

# @nb.vectorize(['int16(int64, int64, float64[:], float64[:], complex128, \
#      float64[:,:], float64[:], float64, float64, float64, float64, float64)'])


class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)

    def do_3d_projection(self, renderer=None):  #
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        return np.min(zs)


def Init_3020sphere(ax, verbose=False):
    plt.gca().invert_yaxis()
    ax.grid(False)
    ax.w_xaxis.set_pane_color((1, 1, 1, 0.0))
    ax.w_yaxis.set_pane_color((1, 1, 1, 0.0))
    ax.w_zaxis.set_pane_color((1, 1, 1, 0.0))
    # draw the cooridnate frame
    a = Arrow3D(
        [0, 2],
        [0, 0],
        [0, 0],
        mutation_scale=10,
        lw=1,
        arrowstyle="->",
        color="k",
        shrinkA=0,
        shrinkB=0,
    )
    ax.add_artist(a)
    a = Arrow3D(
        [0, 0],
        [0, 1.4],
        [0, 0],
        mutation_scale=10,
        lw=1,
        arrowstyle="->",
        color="k",
        shrinkA=0,
        shrinkB=0,
    )
    ax.add_artist(a)
    a = Arrow3D(
        [0, 0],
        [0, 0],
        [0, 1.3],
        mutation_scale=10,
        lw=1,
        arrowstyle="->",
        color="k",
        shrinkA=0,
        shrinkB=0,
    )
    ax.add_artist(a)

    ax.text(0.8, 1.55, 0, "y", color="black")
    ax.text(2.4, 0.35, 0, "x", color="black")
    ax.text(0, 0.05, 1.25, "z", color="black")
    # draw the sphere
    r = 1
    u, v = np.mgrid[0 : 2 * np.pi : 40j, 0 : np.pi : 20j]
    x = np.cos(u) * np.sin(v)
    y = np.sin(u) * np.sin(v)
    z = np.cos(v)
    ax.plot_surface(x, y, z, cmap=plt.cm.YlGnBu_r, alpha=0.2)
    # draw B0
    a = Arrow3D(
        [0, 0],
        [-0.95, -0.95],
        [0.75, 1.25],
        mutation_scale=10,
        lw=1.6,
        arrowstyle="->",
        color="k",
        shrinkA=0,
        shrinkB=0,
    )
    ax.add_artist(a)
    ax.text(0, -0.85, 1.15, "$\mathbf{B}_0$", color="black")
    # ax.text(1, 0.85, 1.25, '$\mathbf{M}$', color='g')

    # draw magnetization vectors
    # timestamp = np.linspace(start=0, stop=1, num=1000)
    # magz = np.cos(2*np.pi*nu/10*timestamp)
    # magx = np.sqrt(1 - magz**2) * np.cos(2*np.pi*nu*1*timestamp)
    # magy = np.sqrt(1 - magz**2) * np.sin(2*np.pi*nu*1*timestamp)
    # ax.quiver(
    #         0, 0, 0, # <-- starting point of vector
    #         1, 1, 1, # <-- directions of vector
    #         color = 'g', alpha = 1, lw = 1.6, length=1, normalize=False,
    #         arrow_length_ratio=.25, label='$\\vec{M}$'
    #     )
    try:
        ax.set_aspect("equal")
    except NotImplementedError:
        pass
    ax.set_xlim3d([-0.8, 0.99])
    ax.set_ylim3d([-0.8, 0.99])
    ax.set_zlim3d([-0.8, 0.99])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.axis("off")
    # ax.legend(loc='upper right')
    ax.set_box_aspect((1, 1, 1))


def Init_0090sphere(ax, verbose=False):
    plt.gca().invert_yaxis()
    ax.grid(False)
    ax.w_xaxis.set_pane_color((1, 1, 1, 0.0))
    ax.w_yaxis.set_pane_color((1, 1, 1, 0.0))
    ax.w_zaxis.set_pane_color((1, 1, 1, 0.0))
    # draw the cooridnates
    # draw the cooridnate frame
    a = Arrow3D(
        [-1, 1.2],
        [0, 0],
        [0, 0],
        mutation_scale=10,
        lw=1,
        arrowstyle="->",
        color="k",
        shrinkA=0,
        shrinkB=0,
    )
    ax.add_artist(a)
    a = Arrow3D(
        [0, 0],
        [-1, 1.2],
        [0, 0],
        mutation_scale=10,
        lw=1,
        arrowstyle="->",
        color="k",
        shrinkA=0,
        shrinkB=0,
    )
    ax.add_artist(a)
    # a = Arrow3D([0,0],[0,0],[0,1.3], mutation_scale=10, lw=1, arrowstyle="->", color="k", shrinkA=0, shrinkB=0)
    # ax.add_artist(a)
    ax.text(1.2, 0.15, 0, "x", color="black")
    ax.text(0.15, 1.2, 0, "y", color="black")

    # ax.text(0, 0.05, 1.25, 'z', color='black')
    # draw the sphere
    r = 1
    u, v = np.mgrid[0 : 2 * np.pi : 40j, 0 : np.pi : 20j]
    x = np.cos(u) * np.sin(v)
    y = np.sin(u) * np.sin(v)
    z = np.cos(v)
    ax.plot_surface(x, y, z, cmap=plt.cm.YlGnBu_r, alpha=0.2)
    try:
        ax.set_aspect("equal")
    except NotImplementedError:
        pass
    ax.set_xlim3d([-0.8, 0.99])
    ax.set_ylim3d([-0.8, 0.99])
    ax.set_zlim3d([-0.8, 0.99])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.axis("off")
    # ax.legend(loc='upper right')
    ax.set_box_aspect((1, 1, 1))


def Add_vector(
    ax,
    start=None,
    end=None,
    mutation_scale=10,
    lw=1.6,
    color="k",
    alpha=1,
    zorder=5,
    linestyle="-",
    verbose=False,
):
    a = Arrow3D(
        [start[0], end[0]],
        [start[1], end[1]],
        [start[2], end[2]],
        mutation_scale=mutation_scale,
        lw=lw,
        arrowstyle="->",
        color=color,
        alpha=alpha,
        shrinkA=0,
        shrinkB=0,
        zorder=zorder,
        linestyle=linestyle,
    )
    ax.add_artist(a)

    # ax.quiver(
    #         start[0], start[1], start[2], # <-- starting point of vector
    #         end[0], end[1], end[2], # <-- directions of vector
    #         color = 'g', alpha = 1, lw = linewidth, length=1, normalize=False,
    #         arrow_length_ratio=.25, label=''
    #     )


def DrawMethanol(
    ax,
    fontsize,
    textwidth,
    textheight,
    Cposx,
    Cposy,
    vcorr,
):
    # testtext = ax.text(x=0, y=0, s='$\\mathrm{C}$', ha='right', va='top')
    # x, y = testtext.get_va(), testtext.get_bbox_patch()
    # check((x, y))
    bondsize = 1 * textwidth

    gap = 0.08 * textwidth
    ax.text(x=Cposx, y=Cposy, s="$\\mathrm{C}$", ha="center", va="center_baseline")
    # H-C
    ax.text(
        x=Cposx - 2 * gap - bondsize - 0.5 * textwidth,
        y=Cposy,
        s="$\\mathrm{H}$",
        ha="right",
        va="center_baseline",
        color="tab:brown",
    )
    # H
    # |
    # C
    ax.text(
        x=Cposx,
        y=Cposy + vcorr * (2 * gap + bondsize + 0.5 * textheight),
        s="$\\mathrm{H}$",
        ha="center",
        va="bottom",
        color="tab:brown",
    )
    # C
    # |
    # H
    ax.text(
        x=Cposx,
        y=Cposy - vcorr * (2 * gap + bondsize + 0.5 * textheight),
        s="$\\mathrm{H}$",
        ha="center",
        va="top",
        color="tab:brown",
    )
    # C-O
    ax.text(
        x=Cposx + 3 * gap + bondsize + 0.5 * textwidth,
        y=Cposy,
        s="$\\mathrm{O}$",
        ha="left",
        va="center_baseline",
    )
    # O-H
    ax.text(
        x=Cposx
        + 3 * gap
        + bondsize
        + 1.5 * textwidth
        + bondsize * np.cos(72 * np.pi / 180),
        y=Cposy + vcorr * bondsize * np.sin(72 * np.pi / 180),
        s="$\\mathrm{H}$",
        ha="left",
        va="bottom",
        color="tab:purple",
    )

    # pt_list=[[]]
    # H-C
    ax.plot(
        [Cposx - gap - 0.5 * textwidth, Cposx - gap - 0.5 * textwidth - bondsize],
        [Cposy, Cposy],
        color="k",
        lw=1,
    )

    # C-O
    ax.plot(
        [Cposx + gap + 0.5 * textwidth, Cposx + gap + 0.5 * textwidth + bondsize],
        [Cposy, Cposy],
        color="k",
        lw=1,
    )
    # H
    # |
    # C
    ax.plot(
        [Cposx, Cposx],
        [
            Cposy + vcorr * (gap + 0.5 * textheight),
            Cposy + vcorr * (gap + 0.5 * textheight + bondsize),
        ],
        color="k",
        lw=1,
    )
    # C
    # |
    # H
    ax.plot(
        [Cposx, Cposx],
        [
            Cposy - vcorr * (gap + 0.5 * textheight),
            Cposy - vcorr * (gap + 0.5 * textheight + bondsize),
        ],
        color="k",
        lw=1,
    )
    # O-H
    ax.plot(
        [
            Cposx + 3 * gap + 1.4 * textwidth + bondsize,
            Cposx
            + 3 * gap
            + 1.5 * textwidth
            + bondsize * (1 + np.cos(72 * np.pi / 180)),
        ],
        [Cposy, Cposy + vcorr * bondsize * np.sin(72 * np.pi / 180)],
        color="k",
        lw=1,
    )


def AnalyzeHGTrzBrBz(
    file=None,
    BDR_H=None,
    BDR_R=None,
    SMP_H=None,
    SMP_R=None,
    SMP_xc=None,
    SMP_yc=None,
    SMP_zc=None,
    tol=1e-2,
    plt_opt=False,
    verbose=False,
):
    ir = 0
    iz = 1
    iBr = 2
    iBz = 3

    # T2G = 1e4
    # Nwinding = 9
    T2G = 1e4
    Nwinding = 1
    mf = np.loadtxt(file, skiprows=9)  # [shape=(20081, 6)]
    if verbose:
        check(mf)

    if BDR_H is None:
        BDR_H = abs(np.amax(mf[:, iz]) - np.amin(mf[:, iz]))
    if verbose:
        check(BDR_H)

    # BDR_R = 0
    if SMP_H is None or SMP_R is None:
        raise ValueError("SMP_H is None or SMP_R is None")

    if SMP_xc is None:
        SMP_xc = 0
    if SMP_yc is None:
        SMP_yc = 0
    if SMP_zc is None:
        SMP_zc = BDR_H / 2

    SMP_mf = []  # the magnetic field in the sample
    # SMP_wei = []
    for i, r in enumerate(mf[:, ir]):
        r = mf[i, ir]
        z = mf[i, iz]
        if (r) ** 2 < (1 + tol) * SMP_R**2 and abs(z - SMP_zc) < (1 + tol) * SMP_H:
            SMP_mf.append(mf[i, :])
            # SMP_wei.append()

    SMP_mf_arr = np.array(SMP_mf)  #  [shape=(150, 6)]
    del SMP_mf
    # check(SMP_mf_arr)
    if plt_opt:
        plt.rc("font", size=12)
        # plt.rcParams["font.family"] = "serif"
        # plt.rcParams["font.serif"] = ["Times New Roman"]
        plt.rcParams["font.family"] = "Times New Roman"
        plt.rcParams["mathtext.fontset"] = "cm"  # 'dejavuserif'
        fig = plt.figure(figsize=(8, 6), dpi=150)  #
        gs = gridspec.GridSpec(nrows=2, ncols=2)  #

        nbin = 200
        ax00 = fig.add_subplot(gs[0, 0])
        # h, edges = np.histogram(T2G * Nwinding * SMP_mf_arr[:, iBr], \
        #                         weights=SMP_mf_arr[:, ir], bins=nbin)
        h, edges = np.histogram(T2G * Nwinding * SMP_mf_arr[:, iBr], bins=nbin)
        ax00.stairs(h, edges, label="mf.Br")
        ax00.set_xlabel("Br [G]")
        del h, edges

        ax01 = fig.add_subplot(gs[0, 1])
        # h, edges = np.histogram(T2G * Nwinding * SMP_mf_arr[:, iBz], \
        #                         weights=SMP_mf_arr[:, iz], bins=nbin)
        h, edges = np.histogram(T2G * Nwinding * SMP_mf_arr[:, iBz], bins=nbin)
        ax01.stairs(h, edges, label="mf.Bz")
        ax01.set_xlabel("Bz [G]")
        del h, edges

        # ax10 = fig.add_subplot(gs[1, 0])
        # h, edges = np.histogram(SMP_mf_arr[:, iBz], weights=SMP_mf_arr[:, iz])
        # ax10.stairs(h, edges, label='mf.B')
        # del h, edges

        # ax00.legend()
        # ax01.legend()
        plt.show()

    return


def AnalyzeHGTxyzBxByBz(
    file=None,
    BDR_H=None,
    BDR_R=None,
    SMP_H=None,
    SMP_R=None,
    SMP_xc=None,
    SMP_yc=None,
    SMP_zc=None,
    tol=1e-2,
    plt_opt=False,
    verbose=False,
):
    mf = np.loadtxt(file, skiprows=9)  # [shape=(20081, 6)]
    # check(mf)
    if BDR_H is None:
        BDR_H = abs(np.amax(mf[:, 2]) - np.amin(mf[:, 2]))
    check(BDR_H)

    # BDR_R = 0
    if SMP_H is None or SMP_R is None:
        raise ValueError("SMP_H is None or SMP_R is None")

    if SMP_xc is None:
        SMP_xc = 0
    if SMP_yc is None:
        SMP_yc = 0
    if SMP_zc is None:
        SMP_zc = BDR_H / 2

    SMP_mf = []

    for i, x in enumerate(mf[:, 0]):
        y = mf[i, 1]
        z = mf[i, 2]
        if (x - SMP_xc) ** 2 + (y - SMP_yc) ** 2 < (1 + tol) * SMP_R**2 and abs(
            z - SMP_zc
        ) < (1 + tol) * SMP_H:
            SMP_mf.append(mf[i, :])

    SMP_mf_arr = np.array(SMP_mf)  #  [shape=(150, 6)]
    del SMP_mf
    # check(SMP_mf_arr)
    if plt_opt:
        plt.rc("font", size=12)
        # plt.rcParams["font.family"] = "serif"
        # plt.rcParams["font.serif"] = ["Times New Roman"]
        plt.rcParams["font.family"] = "Times New Roman"
        plt.rcParams["mathtext.fontset"] = "cm"  # 'dejavuserif'
        fig = plt.figure(figsize=(8, 6), dpi=150)  #
        gs = gridspec.GridSpec(nrows=2, ncols=2)  #

        ax00 = fig.add_subplot(gs[0, 0])
        h, edges = np.histogram(SMP_mf_arr[:, 3])
        ax00.stairs(h, edges, label="mf.Bx")
        del h, edges

        ax10 = fig.add_subplot(gs[1, 0])
        h, edges = np.histogram(SMP_mf_arr[:, 5])
        ax10.stairs(h, edges, label="mf.Bz")
        del h, edges

        plt.show()

    return


def axion_lineshape(v_0, v_lab, nu_a, nu, case="non-grad", alpha=0.0):
    """
    Calculate analytical lineshapes.

    Parameters
    ----------

    Return
    ------
    A float array of the axion lineshape

    Reference
    ---------
    A. Gramolin: https://github.com/gramolin/lineshape

    """
    c = 299792458.0  # Speed of light (in m/s)
    v_0, v_lab = np.abs(v_0), np.abs(v_lab)

    # shift = 0#max(1 - np.amin(nu), 1 + np.abs(nu_a))
    # nu_a += shift
    # nu += shift

    full_lineshape = np.zeros(len(nu))

    # Find the index of the first non-zero element
    positive_indices = np.where(nu > nu_a)[0]
    if positive_indices.size > 0:
        nu_a_index = positive_indices[0]
    else:
        return full_lineshape

    freq = nu[nu_a_index:-1]

    assert case in [
        "non-grad",
        "grad_par",
        "grad_perp",
    ], "Case should be 'non-grad', 'grad_par', or 'grad_perp'!"

    beta = 2 * c * v_lab * np.sqrt(2 * (freq - nu_a) / nu_a) / v_0**2  # Eq. (13)
    # *np.sqrt(2 * (nu - nu_a) / nu_a)
    # for arr in [beta]:
    #     print('beta check')
    #     has_nan = np.isnan(arr).any()  # Check for NaN
    #     has_inf = np.isinf(arr).any()  # Check for Inf

    #     print(f"Contains NaN: {has_nan}")  # Output: True
    #     print(f"Contains Inf: {has_inf}")  # Output: True
    if case == "non-grad":  # Non-gradient case, Eq. (12)
        ax_sq_lineshape = (
            2
            * c**2
            * np.exp(-((0.5 * beta * v_0 / v_lab) ** 2) - (v_lab / v_0) ** 2)
            * np.sinh(beta)
            / (np.sqrt(np.pi) * v_0 * v_lab * nu_a)
        )

    elif case == "grad_par":  # Parallel gradient case, Eq. (19)
        factor = (
            np.cos(alpha) ** 2
            - (1 / np.tanh(beta) - 1.0 / beta) * (2 - 3 * np.sin(alpha) ** 2) / beta
        )
        ax_sq_lineshape = (
            (4 * c**2 / (v_0**2 + 2 * (v_lab * np.cos(alpha)) ** 2))
            * (freq / nu_a - 1)
            * factor
            * axion_lineshape(v_0, v_lab, nu_a, nu)
        )
    elif case == "grad_perp":  # Perpendicular gradient case, Eq. (20)
        factor = (
            np.sin(alpha) ** 2
            + (1.0 / np.tanh(beta) - 1.0 / beta)
            * (2.0 - 3.0 * np.sin(alpha) ** 2)
            / beta
        )
        ax_sq_lineshape = (
            (2 * c**2 / (v_0**2 + (v_lab * np.sin(alpha)) ** 2))
            * (freq / nu_a - 1)
            * factor
            * axion_lineshape(v_0, v_lab, nu_a, nu)[nu_a_index:-1]
        )
    else:  # adding this to try and get rid of an error message
        return np.zeros(nu.shape)

    full_lineshape[nu_a_index:-1] += ax_sq_lineshape
    # nu_a -= shift
    # nu -= shift
    return full_lineshape


# def get_ALP_wind(
#     year=None,
#     month=None,
#     day=None,
#     time=None,
#     lat=None,
#     lon=None,
#     elev=None,
#     verbose=False,
# ):
#     """
#     returns the velocity 'v_lab' between lab frame and DM halo (SHM), in the galactic rest frame, for the specified coordinates and time
#     returns the angle [rad] between the CASPEr sensitive axis (z-direction = earth surface normal) and 'v_lab'

#     time: needs to be in the format "15:47:18"
#         if none is specified, use current time

#     lat: latitude of experiment location
#         if none is specified, use Mainz: 49.9916 deg north

#     lon: longitude of experiment location
#         if none is specified, use Mainz: 8.2353 deg east

#     elev: height of experiment location
#         if none is specified, use Uni Campus Mainz: 130 m
#     """
#     if verbose:
#         print("now calculating wind angle")

#     # assert None is not in []
#     CASPEr_lat = 49.9916  # degrees north
#     CASPER_lon = 8.2353  # degrees east
#     CASPER_elevation = 130  # meters

#     if lat is None:
#         lat = CASPEr_lat
#         if verbose:
#             print(
#                 f"no latitute input provided, using CASPEr-Mainz location: {CASPEr_lat}"
#             )
#     else:
#         if verbose:
#             print(f"latitute: {lat}")
#     if lon is None:
#         lon = CASPER_lon
#         if verbose:
#             print(
#                 f"no longitude input provided, using CASPEr-Mainz location: {CASPER_lon}"
#             )
#     else:
#         if verbose:
#             print(f"longitude: {lon}")
#     if elev is None:
#         elev = CASPER_elevation
#         if verbose:
#             print(
#                 f"no elevation input provided, using CASPEr-Mainz location: {CASPER_elevation}"
#             )
#     else:
#         if verbose:
#             print(f"elevation: {elev}")

#     if (year or month or day or time) is None:
#         timedate_DMmeasure = Time.now()
#         print(
#             f"no date and time input provided, using current date and time: {timedate_DMmeasure}"
#         )
#     else:
#         timedate_DMmeasure = rf"{year}-{month}-{day}T{time}"
#     if verbose:
#         print(f"time input: {timedate_DMmeasure}")

#     timeastro = Time(timedate_DMmeasure, format="isot", scale="utc")
#     DMtimefrac = wind.FracDay(Y=2022, M=12, D=23)
#     if verbose:
#         print("time of DM measurement (fractional days): ", DMtimefrac)

#     LABvel = wind.ACalcV(DMtimefrac)
#     if verbose:
#         print("velocity (lab frame) @DM time: ", LABvel)

#     DMtime, unit_North, unit_East, unit_Up, Vhalo = wind.get_CASPEr_vect(
#         time=timeastro,
#         lat=CASPEr_lat,
#         lon=CASPER_lon,
#         elev=CASPER_elevation,
#     )

#     # print(type(Vhalo))
#     Vlab = Vhalo.get_d_xyz()  # convert into a vector
#     Bz = (
#         unit_Up.get_xyz()
#     )  # our leading field is pointing up perpendicular to earth's surface

#     alpha_ALP = angle_between(Vlab, Bz).value
#     v_ALP = np.linalg.norm(Vlab.value) * 1e3
#     v_ALP_perp = v_ALP * np.sin(alpha_ALP)

#     if verbose:
#         # print("time of DM measurement: ", DMtime)
#         print("Bz vector @DM time (galaxy frame):", Bz)
#         print("v_halo @DM time (galaxy frame):", Vhalo)
#         print("v_lab @DM time:", Vlab)
#         print("angle between sensitive axis & lab velocity @DM time: ", alpha_ALP)

#     ###############################################################################################
#     # do not delete
#     return v_ALP, v_ALP_perp, alpha_ALP


def unit_vector(vector):
    """Returns the unit vector of the vector."""
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    """Returns the angle in radians between vectors 'v1' and 'v2'::

    >>> angle_between((1, 0, 0), (0, 1, 0))
    1.5707963267948966
    >>> angle_between((1, 0, 0), (1, 0, 0))
    0.0
    >>> angle_between((1, 0, 0), (-1, 0, 0))
    3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def extract_number(s):
    return int("".join(filter(str.isdigit, s)))


def get_random_color():
    return (random.random(), random.random(), random.random())


# for histogram
def is_non_empty(entry):
    """
    Recursively check if an entry (element or nested array) is non-empty.
    """
    if isinstance(entry, np.ndarray):
        return np.any([is_non_empty(sub_entry) for sub_entry in entry])
    elif entry is not None:
        return True
    else:
        return False


def filter_non_empty(entry):
    """
    Recursively filter non-empty entries from the nested array.
    """
    if isinstance(entry, np.ndarray):
        return np.array(
            [
                filter_non_empty(sub_entry)
                for sub_entry in entry
                if is_non_empty(sub_entry)
            ]
        )
    elif entry is not None:
        return entry
    else:
        return None


def merge_peak_bins(vals, threshold):
    # Sort values and keep track of original indices
    indexed_values = sorted((value, index) for index, value in enumerate(vals))

    merged_values = []
    merged_indices = []

    i = 0
    while i < len(indexed_values):
        current_value, current_index = indexed_values[i]
        next_index = i + 1
        to_merge = [current_value]
        original_indices = [current_index]

        # Find all values within the threshold
        while (
            next_index < len(indexed_values)
            and indexed_values[next_index][0] - current_value < threshold
        ):
            next_value, next_index_value = indexed_values[next_index]
            to_merge.append(next_value)
            original_indices.append(next_index_value)
            next_index += 1

        # Keep the highest value from the merged values
        max_value = max(to_merge)
        merged_values.append(max_value)
        merged_indices.append(original_indices[to_merge.index(max_value)])

        # Move the index to the next group of values
        i = next_index

    return merged_values, merged_indices


def merge_close_entries(floats, threshold):
    sorted_floats = sorted(floats)
    filtered_floats = []
    current_max = sorted_floats[0]

    for value in sorted_floats:
        if value - current_max > threshold:
            # If the current value is too far from the current max, update the max and add it to the filtered list
            filtered_floats.append(value)
            current_max = value
        elif value == current_max:
            # If the value is equal to the current max, just skip it
            continue
        else:
            # If the value is close to the current max, skip it
            continue
    filtered_floats.append(current_max)
    return filtered_floats


def merge_close_entries2(floats, threshold):

    merged_floats = []
    sorted_floats = sorted(floats)
    try:
        current_max = np.max(sorted_floats)
    except ValueError:
        print("empty candidate list - cannot merge!")
        return merged_floats
    merged_floats.append(current_max)
    for value in sorted_floats:
        if np.abs(current_max - value) > threshold:
            merged_floats.append(value)
            current_max = value
    return merged_floats


def read_double_precision_floats(file_path):
    doubles = []

    with open(file_path, "rb") as bin_file:
        while True:
            # Read 8 bytes (size of a double-precision float)
            bytes_read = bin_file.read(8)
            if not bytes_read:
                break  # End of file

            # Unpack the bytes into a double-precision float
            # double_value = struct.unpack('d', bytes_read)[0]
            # doubles.append(double_value)

    return doubles


def find_consistent_values(fCNDs, margin):
    consistent_set = []

    for value in fCNDs[0]:
        set_of_inds = [0]
        set_of_vals = [value]
        consistent = True
        for i in range(1, len(fCNDs)):
            found = False
            for other_value in fCNDs[i]:
                if abs(value - other_value) <= margin:
                    found = True
                    set_of_vals.append(other_value)
                    set_of_inds.append(i)
                    break
            if not found:
                consistent = False
                break
        if consistent:
            consistent_set.append((set_of_vals, set_of_inds))

    return consistent_set


def is_within_margin(value, sublist, margin):
    return any(abs(value - x) <= margin for x in sublist)


def ReturnNMRdata(basepath, scanname, index):

    if "20221223" in scanname:
        logfile = rf"{basepath}/{scanname}/OnePulse analysis/fitResults.txt"  # Dec 23
        errfile = rf"{basepath}/{scanname}/OnePulse analysis/fitUncertainties.txt"
    elif "20221214" in scanname:
        logfile = rf"{basepath}/{scanname}/1Pulse data/fitResults.txt"  # Dec 14
        errfile = rf"{basepath}/{scanname}/1Pulse data/fitUncertainties.txt"
    Log = np.genfromtxt(
        logfile,
        unpack=True,
        delimiter=" ",
        skip_header=0,
        filling_values=0,
        invalid_raise=False,
    )
    errLog = np.genfromtxt(
        errfile,
        unpack=True,
        delimiter=" ",
        skip_header=0,
        filling_values=0,
        invalid_raise=False,
    )

    # get larmor frequency from fit to NMR signal peak of 1Pulse measurement
    FLarmor = 0
    if Log[0].size < 2:
        FLarmor = Log[0]
        FLarmor_err = errLog[0]
    else:
        FLarmor = Log[0][index]
        FLarmor_err = errLog[0][index]

    # get linewidth of NMR signal
    FWHM_n = 0
    if Log[1].size < 2:
        FWHM_n = Log[1]
        FWHM_n_err = errLog[1]
    else:
        FWHM_n = Log[1][index]
        FWHM_n_err = errLog[1][index]

    # get measured flux inside SQUID during the time of the pi/2 pulse
    area = 0
    if Log[2].size < 2:
        area = Log[2]
        area_err = errLog[2]
    else:
        area = Log[2][index]
        area_err = errLog[2][index]
    # check(area)

    return FLarmor, FWHM_n, area, FLarmor_err, FWHM_n_err, area_err


def GetDateTimeSimple(year, month, day, time_hms, return_as_int=False):
    datetime_obj = datetime.strptime(
        f"{year}-{month:02}-{day:02} {time_hms}", "%Y-%m-%d %H:%M:%S"
    )
    hour = datetime_obj.hour
    minute = datetime_obj.minute
    second = datetime_obj.second
    if return_as_int:  # for easy writing into params text file
        return math.floor(
            float(f"{year}{month}{day}{hour:02d}{minute:02d}{second:02d}")
        )
    else:
        return datetime_obj


def MovAvgByStep(xstamp=None, rawsignal=None, weights=None, step_len=1, verbose=False):
    """
    A moving average with tunable step length, especially designed for axion signal search.

    Parameters
    ----------
    rawsignal : array
        raw signal
    weights : array
        The weights for doing the averaging
    step_len : int
        The step length for doing the moving average.
        Default to 1.
    verbose : bool
        It is here for no reason.

    Return
    ------
    np.array(prcdsiganl) : array
        The processed signal.
    """
    assert xstamp is not None
    assert rawsignal is not None
    assert weights is not None
    assert step_len is None or type(step_len) is int
    if step_len is None:
        step_len = 1
    if step_len < 1:
        raise ValueError("step_len < 1. Increase step_len.")

    step_size = step_len * abs(xstamp[1] - xstamp[0])
    # normalization of the template signal / weights
    # if not np.isclose([np.sum(weights)], [1.0], rtol=1e-05, atol=1e-06):
    #     print(f'Warning from {MovAvgByStep.__name__}: ' + \
    #           f'np.sum(weights) = {np.sum(weights)} != 1.0. '
    #           'The normalization of the weights is done anyway.')
    #     weights /= np.sum(weights)

    # processed signal
    prcd_xstamp = []
    prcd_siganl = []

    # calculate the number of steps
    numofstep = len(rawsignal) // step_len
    #
    for i in range(numofstep):
        if i * step_len + len(weights) > len(rawsignal):
            break
        prcd_siganl.append(
            np.vdot(rawsignal[i * step_len : i * step_len + len(weights)], weights)
        )
        prcd_xstamp.append([i * step_size + xstamp[0]])
    return np.array(prcd_xstamp), np.array(prcd_siganl)


def record_runtime_YorN(RECORD_RUNTIME):
    def record_runtime(func):
        def wrapper(*args, **kwargs):
            if RECORD_RUNTIME:
                start_time = time.time()
                result = func(*args, **kwargs)
                end_time = time.time()
                clear_lines()
                print(
                    f"Function {func.__name__} took {end_time - start_time:.2g} [s] to run."
                )
                sys.stdout.flush()
            else:
                result = func(*args, **kwargs)
            return result

        return wrapper

    return record_runtime


def PlotGaussianLine(
    ax: matplotlib.axes.Axes,
    x,
    std,
    sumofcounts,
):
    binsize = abs(x[1] - x[0]) / std
    ax.plot(x, sumofcounts * binsize * norm.pdf(x / std))


def get_value_after_keyword(file, keyword):
    with open(file, "r") as file:
        for line in file:
            if line.startswith(keyword):  # Observe Freq., 	461.79573699999997 MHz
                value = line.split()[2]
                try:
                    return float(value)
                except:
                    try:
                        if value[-1] == "u":
                            return float(value[:-1]) * 1e-6
                        elif value[-1] == "m":
                            return float(value[:-1]) * 1e-3
                    except:
                        return value
    return 0


def get_nextline_after_keyword(file, keyword):
    values = []
    with open(file, "r", encoding="utf-8", errors="ignore") as file:
        KWfound = False
        for line in file:
            if KWfound:
                values = [float(x) for x in line.strip().split()]
                break
            elif keyword in line:
                KWfound = True
    return values


def ChisqMeanDistribution(N_mean: int = 240, Nsamples: int = 5000):

    samples = norm.rvs(size=N_mean * Nsamples)

    samples = samples.reshape((N_mean, Nsamples))

    avg = np.mean(samples**2, axis=0)

    print(avg.shape)

    hist, bin_edges = np.histogram(avg, bins=10)

    hist_x = []
    hist_y = []
    for i, count in enumerate(hist):
        if count > 0:
            hist_x.append((bin_edges[i] + bin_edges[i + 1]) / 2.0)
            hist_y.append(count)

    plt.figure()
    plt.scatter(
        hist_x,
        hist_y,
        color="green",
        edgecolors="k",
        linewidths=1,
        marker="o",
        s=6,
        zorder=6,
        label="histogram",
    )
    plt.plot(hist_x, hist_y)
    plt.yscale("log")
    plt.show()


def checkDrift(
    current_data,
    reference_data,
    stattest_threshold: float = 0.05,
    makeplot: bool = False,
    verbose: bool = False,
):
    """ """
    # ensure data length is enough for statistical analysis
    assert len(current_data) > 100
    assert len(reference_data) > 100

    # pack sample data
    reference_data = pd.DataFrame(
        {"timestamp": np.arange(len(reference_data)), "value": reference_data}
    )

    current_data = pd.DataFrame(
        {"timestamp": np.arange(len(current_data)), "value": current_data}
    )

    # Create and run the report
    data_drift_report = Report(
        metrics=[DataDriftPreset(stattest_threshold=stattest_threshold)]
    )
    data_drift_report.run(reference_data=reference_data, current_data=current_data)

    # Extract the results
    results = data_drift_report.as_dict()
    threshold = results["metrics"][1]["result"]["drift_by_columns"]["timestamp"][
        "stattest_threshold"
    ]
    dataset_drift = results["metrics"][0]["result"]["dataset_drift"]
    # print(results['metrics'][0]['result'].keys())
    # dict_keys(['drift_share', 'number_of_columns', 'number_of_drifted_columns', 'share_of_drifted_columns', 'dataset_drift'])
    # print(results['metrics'][1]['result'].keys())
    # dict_keys(['number_of_columns', 'number_of_drifted_columns', 'share_of_drifted_columns', 'dataset_drift', 'drift_by_columns', 'current_fi', 'reference_fi'])

    if verbose:
        print(f"threshold = {threshold:.2e}")
        print(f"dataset_drift = {dataset_drift}")

    if makeplot:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(
            reference_data["timestamp"], reference_data["value"], label="Reference Data"
        )
        ax.plot(current_data["timestamp"], current_data["value"], label="Current Data")
        ax.set_title("Time-Series Data Drift")
        ax.set_xlabel("Timestamp")
        ax.set_ylabel("Value")
        ax.legend()
        plt.show()
    return dataset_drift  # boolen


def CompareCheckNorm(alpha=0.05):
    # Perform a Monte Carlo hypothesis test.
    print("Perform a Monte Carlo hypothesis test.")
    MCtest_mistake = 0
    example_data = np.random.normal(loc=0, scale=1, size=1000)
    for i in tqdm(range(1000)):
        data = np.random.normal(loc=0, scale=1, size=10000)
        datadrfit = checkDrift(
            current_data=data, reference_data=example_data, stattest_threshold=alpha
        )
        # Interpret the result
        if datadrfit:
            MCtest_mistake += 1
    sys.stdout.write("\n")

    # Perform Shapiro-Wilk test
    print("Shapiro-Wilk test")
    shapiro_mistake = 0
    for i in tqdm(range(1000)):
        data = np.random.normal(loc=0, scale=1, size=10000)
        stat, p = stats.shapiro(data)
        # Interpret the result
        if p < alpha:
            shapiro_mistake += 1
    sys.stdout.write("\n")

    # Perform jarque_bera test
    print("jarque_bera test")
    jarque_bera_mistake = 0
    for i in tqdm(range(1000)):
        data = np.random.normal(loc=0, scale=1, size=10000)
        stat, p = stats.jarque_bera(data)
        # Interpret the result
        if p < alpha:
            jarque_bera_mistake += 1
    sys.stdout.write("\n")

    # Perform D'Agostino and Pearson's test
    print("jarque_bera test")
    normaltest_mistake = 0
    for i in tqdm(range(1000)):
        data = np.random.normal(loc=0, scale=1, size=10000)
        stat, p = stats.normaltest(data)
        # Interpret the result
        if p < alpha:
            normaltest_mistake += 1
    sys.stdout.write("\n")

    check(MCtest_mistake)
    check(shapiro_mistake)
    check(jarque_bera_mistake)
    check(normaltest_mistake)


def print_progress_bar(
    iteration,
    total,
    prefix="Progress",
    suffix="Complete",
    decimals=3,
    length=50,
    fill="█",
):
    percent = ("{0:." + str(decimals) + "f}").format(
        100.0 * ((iteration) / float(total))
    )
    filled_length = int(length * (iteration) // (total))
    bar = fill * filled_length + "-" * (length - filled_length)
    sys.stdout.write(f"\r{prefix} |{bar}| {percent}% {suffix}")
    sys.stdout.flush()
    # write \n when the iteration finishes
    if (iteration) == total:
        sys.stdout.write("\n")


# def progress_bar(
#         current, total, bar_length=20,
#         addtext=None):
#     fraction = current / (total-1)

#     arrow = int(fraction * bar_length - 1) * '-' + '>'
#     padding = int(bar_length - len(arrow)) * ' '

#     # ending = '\n' if current == total else '\r'
#     prg_info = f'Progress: [{arrow}{padding}] {int(fraction*100)}%'

#     if addtext is not None and type(addtext) is str:
#         prg_info += ' ' + addtext
#     print(prg_info, end='\r')


def update_info(info):
    sys.stdout.write(f"{info}")
    sys.stdout.flush()


def clear_lines():
    sys.stdout.write("\r\033[K")  # Move cursor up and clear the line
    sys.stdout.flush()  #


def exampleofprogress():
    # Example usage
    total = 100
    print("Starting the process...")
    sys.stdout.flush()
    # time.sleep(2)
    for i in range(total + 1):
        if i % 10 == 0:
            clear_lines()
            print(f"i = {i}, asdafdqw=")
            print(f"i = {i}, asdafdqw=")
            sys.stdout.flush()
        time.sleep(0.1)  # Simulate some work being done
        print_progress_bar(i, total, prefix="Progress", suffix="Complete", length=50)
        # time.sleep(0.1)  # Simulate some work being done

    sys.stdout.write("\n")  # Move to the next line after the progress bar is complete


def chi_squared_pdf(bincnt, counts, v):
    return v[0] * np.max(counts) * chi2.pdf(bincnt / v[1] + v[2], v[3])


def RNCF_chi2(bincnt, counts, v):
    return np.linalg.norm(counts - chi_squared_pdf(bincnt, v))


def gamma_pdf(bincnt, counts, v):
    return gamma.pdf(bincnt, v[0], v[1], v[2])  # shape, loc, scale


def RNCF_gamma(bincnt, counts, v):
    return np.linalg.norm(counts - gamma_pdf(bincnt, v))


def gamma_pdf2(bincnt, counts, v):
    return v[0] * np.max(counts) * gamma.pdf(bincnt / v[1] + v[2], v[3], v[4], v[5])


def RNCF_gamma2(bincnt, counts, v):
    return np.linalg.norm(counts - gamma_pdf2(bincnt, v))


def count_text_rows(file_path):
    text_row_count = 0
    with open(file_path, "r") as file:
        for line in file:
            try:
                # Try to convert the line to a float
                float(line.strip())
            except ValueError:
                # If conversion fails, it's a text row
                text_row_count += 1
    return text_row_count


def getFWHM(x, y):
    """
    Calculate the Full Width at Half Maximum (FWHM) of a curve.

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
    y_max = np.max(y)
    half_max = y_max / 2.0

    # Find indices where y crosses the half-maximum
    indices = np.where(y >= half_max)[0]
    if len(indices) < 2:
        raise ValueError(
            "Cannot calculate FWHM: The curve does not have two points crossing the half-maximum."
        )

    # Extract the first and last indices crossing the half-maximum
    left_index = indices[0]
    right_index = indices[-1]

    # Interpolate to find more precise crossing points
    x_left = np.interp(
        half_max, [y[left_index - 1], y[left_index]], [x[left_index - 1], x[left_index]]
    )
    x_right = np.interp(
        half_max,
        [y[right_index], y[right_index + 1]],
        [x[right_index], x[right_index + 1]],
    )

    # Calculate FWHM
    fwhm = x_right - x_left
    return fwhm


def calculate_fwhm(x, y, peak=True):
    """
    Calculate the Full Width at Half Maximum (FWHM) of a curve.
    Works for both peaks and dips.

    Parameters:
        x (array-like): The x-values of the curve.
        y (array-like): The y-values of the curve (can be positive or negative).
        peak (bool): If True, calculate FWHM for a peak (maximum). If False, calculate for a dip (minimum).

    Returns:
        float: The FWHM of the curve.
    """
    # Ensure inputs are numpy arrays
    x = np.array(x)
    y = np.array(y)

    # Determine the maximum or minimum value and the half-maximum
    if peak:
        y_extreme = np.max(y)
        half_max = y_extreme / 2.0
        indices = np.where(y >= half_max)[0]
    else:
        y_extreme = np.min(y)
        half_max = y_extreme / 2.0
        indices = np.where(y <= half_max)[0]

    # Check if the curve crosses the half-maximum value
    if len(indices) < 2:
        raise ValueError(
            "Cannot calculate FWHM: The curve does not have two points crossing the half-maximum."
        )

    # Extract the first and last indices crossing the half-maximum
    left_index = indices[0]
    right_index = indices[-1]

    # Interpolate to find more precise crossing points
    if left_index > 0:
        x_left = np.interp(
            half_max,
            [y[left_index - 1], y[left_index]],
            [x[left_index - 1], x[left_index]],
        )
    else:
        x_left = x[left_index]

    if right_index < len(y) - 1:
        x_right = np.interp(
            half_max,
            [y[right_index], y[right_index + 1]],
            [x[right_index], x[right_index + 1]],
        )
    else:
        x_right = x[right_index]

    # Calculate FWHM
    fwhm = x_right - x_left
    return fwhm


def get_FWHM_indice(x, y):
    """
    Calculate the Full Width at Half Maximum (FWHM) of a dip in the unit of indice.

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
