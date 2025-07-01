from math import gamma
import os
import sys
from turtle import color

print(os.path.abspath(os.curdir))
sys.path.insert(0, os.path.abspath(os.curdir))
# os.chdir("..")  # if you want to go to parent folder
# os.chdir("..")
# print(os.path.abspath(os.curdir))
# sys.path.insert(0, os.path.abspath(os.curdir))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
from matplotlib.ticker import ScalarFormatter
from DataAnalysis import Exclusion, DualChanSig

# from DataAnalysis import *
from functioncache import check, Lorentzian
from scipy.stats import rayleigh, uniform, norm
from scipy import signal


Omega_e = 1
nu_e = 1e6 + 0
nu_L = 1e6
nu_rot = 100
samprate = 6000


period = 0.5
dutycycle = 0.5
numperiod = 5


periodlen = int(period * samprate)
dutylen = int(dutycycle * period * samprate)
timestamp = np.linspace(start=0, stop=numperiod * period, num=numperiod * periodlen)
timestamp_1period = np.linspace(start=0, stop=period, num=periodlen)


precessionsignature = signal.square(
    2 * np.pi * (1.0 / period) * timestamp, duty=dutycycle
)
timestamp_onduty = np.linspace(start=0, stop=dutycycle * period, num=dutylen)
timestamp_offduty = np.linspace(
    start=0, stop=(1 - dutycycle) * period, num=periodlen - dutylen
)

# plt.plot(timestamp, precessionsignature)
# plt.show()


Gamma_n = 10
T2star = 1 / (np.pi * Gamma_n)
T2 = 1


def h(x, Gamma_n):
    return Lorentzian(x, 0, Gamma_n, 1, 0)


Omgea_a = 1
NMRenvelope = (
    Omgea_a
    * T2star
    * (1 - np.exp(-timestamp_onduty / T2star))
    * np.exp(-timestamp_onduty / T2star)
)

st_x = np.array([])
st_y = np.array([])
zerosignal_offduty = np.zeros(len(timestamp_offduty))
phase_list = uniform.rvs(loc=0, scale=2 * np.pi, size=100)
for i in range(numperiod):
    phase0 = phase_list[i]
    st_x = np.append(
        st_x, NMRenvelope * np.cos(2 * np.pi * nu_rot * timestamp_onduty + phase0)
    )
    st_y = np.append(
        st_y, NMRenvelope * np.sin(2 * np.pi * nu_rot * timestamp_onduty + phase0)
    )
    st_x = np.append(st_x, zerosignal_offduty)
    st_y = np.append(st_y, zerosignal_offduty)
    # st_y_list.append(NMRenvelope * np.sin(2*np.pi*nu_rot*timestamp_onduty+phase0))
    # st_x_list.append(zerosignal_offduty)
    # st_y_list.append(zerosignal_offduty)


# st_x += uniform.rvs(loc=0, scale=.002, size=len(timestamp))
# st_y += uniform.rvs(loc=0, scale=.002, size=len(timestamp))


# peakcutoff = 5
# peaksamprange = [-peakcutoff*Gamma_n, peakcutoff*Gamma_n]
# peaksampnum = int(2*peakcutoff*Gamma_n*(np.pi * T2))*10
# check(peaksampnum)
# freqstamp = np.linspace(start=peaksamprange[0], stop=peaksamprange[1], num=peaksampnum)

# replength = int(samprate)
# stepfunction = np.zeros(len(timestamp))
# for i in range(1, 8):
# 	stepfunction[replength * 2*i:replength * (2*i+1)] = 1


# dataptnum = len(timestamp)
# st_x = np.zeros(len(timestamp))
# st_y = np.zeros(len(timestamp))
# phase_list = uniform.rvs(loc=0, scale=2*np.pi, size=100)
# check(phase_list)
# for i in range(1, 8):
# 	timestamp_i = timestamp[0:replength * (1)]
# 	phase0 = phase_list[i]
# 	for nu_m in freqstamp:
# 		if abs(nu_m) < 9.5:
# 			un = h(x=nu_m, Gamma_n=Gamma_n)
# 			st_x[replength * 2*i:replength * (2*i+1)] += \
# 				un * T2 * (1-np.exp(-timestamp_i/T2)) * np.exp(-timestamp_i/T2) * np.sin(2*np.pi*nu_m*timestamp_i+phase0)
# 			st_y[replength * 2*i:replength * (2*i+1)] += \
# 				un * T2 * (1-np.exp(-timestamp_i/T2)) * np.exp(-timestamp_i/T2) * np.cos(2*np.pi*nu_m*timestamp_i+phase0)


# 	# replength = samprate//3
# 	# check(st_x[replength * i:replength * (i+1)].shape)
# 	# check(st_x[0:replength].shape)
# 	# st_x[replength * i:replength * (i+1)] += st_x[0:replength]
# 	# st_y[replength * i:replength * (i+1)] += st_y[0:replength]

plt.rc("font", size=12)
plt.rcParams["font.family"] = "Times New Roman"
# plt.rcParams["font.family"] = "serif"
# plt.rcParams["font.serif"] = ["Times New Roman"]
plt.rcParams["mathtext.fontset"] = "cm"  # 'dejavuserif'
fig = plt.figure(figsize=(5, 2.5), dpi=300)
gs = gridspec.GridSpec(nrows=2, ncols=1)

ax00 = fig.add_subplot(gs[0, 0])
ax00.plot(timestamp, precessionsignature, label="duty cycle", alpha=1, lw=1.2)

ax01 = fig.add_subplot(gs[1, 0])

ax01.plot(timestamp, st_x, label="x", alpha=1, lw=1.2)
ax01.plot(timestamp, st_y, label="x", alpha=1, lw=1.2)

# ax00.set_xscale('log')
# ax00.set_yscale('log')
ax01.set_xlabel("")
ax01.set_ylabel("")
# ax00.legend(bbox_to_anchor=(1.0, 1), loc='upper left', ncol=1)
# ax00.set_xticks((1, 10, 100, 600))
# ax00.set_xticklabels(('$10^{0}$', '$10^{1}$', '$10^{2}$', '$6\\times10^{2}$'))
# letters = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)', '(i)']
# for i, ax in enumerate([ax00, ax00]):
#     xleft, xright = ax.get_xlim()
#     ybottom, ytop = ax.get_ylim()
#     ax.text(x=xleft, y=ytop, s = letters[i], ha='right', va = 'bottom', color='blue', fontsize=14)
plt.tight_layout()
plt.show()

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
liastream.dmodfreq = nu_L
saveintv = 1
liastream.samprate = samprate
# check(magnetization.timestamp.shape)
# check(magnetization.trjry[0:-1:saveintv, 0].shape)

liastream.dataX = st_x
liastream.dataY = st_y

liastream.GetSpinNoisePSDsub(
    chunksize=timestamp[-1] / 2,  # magnetization.T2
    analysisrange=[0, -1],  # [0, int(9*samplelinewidth*liastream.samprate)]
    interestingfreq_list=[],
    # ploycorrparas=ployparas,
    # ploycorrparas=[],
    # showstd=False,
    # stddev_range=[1.349150e6,1.349750e6],
    verbose=False,
)
# liastream.FitPSD(
# 		fitfunction = 'Lorentzian',  # 'Lorentzian' 'dualLorentzian' 'tribLorentzian' 'Gaussian 'dualGaussian' 'auto' 'Polyeven'
# 		inputfitparas = ['auto','auto','auto','auto'],
# 		smooth=False,
# 		smoothlevel=1,
# 		fitrange=['auto','auto'],
# 		alpha=0.05,

# 		getresidual=False,
# 		getchisq=False,
# 		verbose=False
# 	)
specxaxis, spectrum, specxunit, specyunit = liastream.GetSpectrum(
    showtimedomain=False,
    showacqdata=True,
    showfreqdomain=True,
    showfit=True,
    showresidual=False,
    showlegend=True,  # !!!!!show or not to show legend
    spectype="PSD",  # in 'PSD', 'ASD', 'FLuxPSD', 'FluxASD'
    ampunit="V",
    Mf=1,
    Rf=1,
    specxunit="Hz",  # 'Hz' 'kHz' 'MHz' 'GHz' 'ppm' 'ppb'
    # specxlim = [nu_L + nu_rot - 10, nu_L + nu_rot + 10],
    # specylim=[0, 4e-23],
    # specxunit2 = 'ppm',
    # referfreq=liastream.dmodfreq,
    # specx2format = '{:.0f}',
    specyscale="linear",  # 'log', 'linear'
    showstd=False,
    figsize=(10, 6),
    showplt_opt=True,
    return_opt=True,
    verbose=False,
)
# print(f'linewidth = {1.0 / (np.pi * T2star):g} Hz, np.amax(spectrum) = {np.amax(spectrum):.2e}')
# listofGammaandSAmp.append([T2star, np.amax(spectrum)])
# listofT2andavgMtsq.append([magnetization.T2, magnetization.avgMxsq + magnetization.avgMysq])  # , np.sum(spectrum), np.amax(spectrum)
# print(f'T2star = {T2star:g} , avg Mt sq = {magnetization.avgMxsq + magnetization.avgMysq:.2e}')

# listofspectrum.append(spectrum)
# K:\CASPEr data\20220522_NMRKineticSimu_data_test2
# f'\\\\Desktop-3ge6tor/d/Mainz/CASPEr/20220522_NMRKineticSimu_data/20220522_test0/sample_IDEN/'
