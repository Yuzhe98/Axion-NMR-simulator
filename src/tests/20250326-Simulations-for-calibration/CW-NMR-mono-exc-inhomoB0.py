import os
import sys

print(os.path.abspath(os.curdir))
# os.chdir("..")  # go to parent folder
# os.chdir("..")  # go to parent folder
print(os.path.abspath(os.curdir))
sys.path.insert(0, os.path.abspath(os.curdir))

import numpy as np
import time
from SimuTools import Sample, MagField, Simulation

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

ExampleSample10MHzT = Sample(
    name="TestSample",  # name of the atom/molecule
    gamma=2
    * np.pi
    * (10)
    * 10**6,  # [Hz/T]. Remember input it like 2 * np.pi * 11.777*10**6
    numofnuclei=1,  #
    tempunit="K",  # temperature scale
    T2=1 / (10 * np.pi),  # [s]
    T1=1000,  # [s]
    pol=1,
    verbose=False,
)


excField = MagField(name="excitation field")  # excitation field in the rotating frame
excField.nu = 1e6 + 0  # [Hz]

Larmor_freqs = 1e6 - np.arange(-10, 2, 2)
B0z_list = Larmor_freqs / (ExampleSample10MHzT.gamma / (2 * np.pi))
specy_list = []
for B0z in B0z_list:
    simu = Simulation(
        name="TestSample 10MHzT",
        sample=ExampleSample10MHzT,  # class Sample
        # gyroratio=(2*np.pi)*11.777*10**6,  # [Hz/T]
        init_time=0.0,  # [s]
        station=None,
        init_mag_amp=1.0,
        init_M_theta=0.0,  # [rad]
        init_M_phi=0.0,  # [rad]
        demodfreq=1e6,
        B0z=B0z,  # [T]
        simuRate=(6696.42871094),  #
        duration=10,
        excField=excField,
        verbose=False,
    )

    simu.generatePulseExcitation(
        pulseDur=1.0 * simu.duration,
        tipAngle=np.pi / 1000,
        direction=np.array([1, 0, 0]),
        showplt=False,  # whether to plot B_ALP
        plotrate=None,
        verbose=False,
    )

    tic = time.perf_counter()
    simu.GenerateTrajectory(verbose=False)
    toc = time.perf_counter()
    print(f"GenerateTrajectory time consumption = {toc-tic:.3f} s")

    # simu.MonitorTrajectory(plotrate=1000, verbose=True)
    # simu.VisualizeTrajectory3D(
    #     plotrate=1e3,  # [Hz]
    #     # rotframe=True,
    #     verbose=False,
    # )

    processdata = True
    if processdata:
        simu.analyzeTrajectory()
        specxaxis, spectrum, specxunit, specyunit = simu.trjryStream.GetSpectrum(
            # showtimedomain=True,
            showfit=True,
            spectype="PSD",  # in 'PSD', 'ASD', 'FLuxPSD', 'FluxASD'
            ampunit="V",
            specxlim=[simu.demodfreq - 20, simu.demodfreq + 12],
            return_opt=True,
            showplt_opt=False,
        )
        # simu.liastream.GetNoPulseFFT()
        # simu.liastream.plotFFT()
        specy_list.append(spectrum)

simu.analyzeB1()
specxaxis, B1spectrum, specxunit, specyunit = simu.trjryStream.GetSpectrum(
    # showtimedomain=True,
    showfit=True,
    spectype="PSD",  # in 'PSD', 'ASD', 'FLuxPSD', 'FluxASD'
    ampunit="V",
    specxlim=[simu.demodfreq - 20, simu.demodfreq + 12],
    return_opt=True,
    showplt_opt=False,
)

# plot style
plt.rc("font", size=10)  # font size for all figures
# plt.rcParams["font.family"] = "serif"
# plt.rcParams["font.serif"] = ["Times New Roman"]
plt.rcParams["font.family"] = "Times New Roman"
# plt.rcParams["mathtext.fontset"] = 'dejavuserif'

# Make math text match Times New Roman
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["mathtext.rm"] = "Times New Roman"

# plt.style.use('seaborn-dark')  # to specify different styles
# print(plt.style.available)  # if you want to know available styles


# fig = plt.figure(figsize=(8.5 * cm, 12 * cm), dpi=300)  # initialize a figure following APS journal requirements
fig = plt.figure(figsize=(6.0, 4.0), dpi=150)  # initialize a figure

gs = gridspec.GridSpec(nrows=2, ncols=1)  # create grid for multiple figures

# to specify heights and widths of subfigures
# width_ratios = [1, 1]
# height_ratios = [1]
# gs = gridspec.GridSpec(nrows=1, ncols=2, \
#   width_ratios=width_ratios, height_ratios=height_ratios)  # create grid for multiple figures

# # fix the margins
# left=0.171
# bottom=0.202
# right=0.952
# top=0.983
# wspace=0.24
# hspace=0.114
# fig.subplots_adjust(left=left, top=top, right=right,
#                     bottom=bottom, wspace=wspace, hspace=hspace)

ax00 = fig.add_subplot(gs[0, 0])
ax10 = fig.add_subplot(gs[1, 0], sharex=ax00)  #

ax00.plot(specxaxis, B1spectrum, label="excitation field")

for i, spectrum in enumerate(specy_list):
    ax10.plot(
        specxaxis,
        specy_list[len(specy_list) - 1 - i],
        label=f"1 MHz + ({Larmor_freqs[len(specy_list)-1-i]-1e6:1.0f} Hz)",
        alpha=0.5,
    )


# put figure index
letters = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)", "(g)", "(h)", "(i)"]
for i, ax in enumerate([ax00, ax10]):
    xleft, xright = ax00.get_xlim()
    ybottom, ytop = ax00.get_ylim()
    ax.text(x=xleft, y=ytop, s=letters[i], ha="left", va="top", color="k")

ax00.legend()
ax10.legend()

# ax00.set_xscale('log')
ax00.set_yscale("log")
# ax10.set_xscale('log')
ax10.set_yscale("log")

ax00.set_ylabel("PSD")
ax10.set_xlabel("Frequency [Hz]")
ax10.set_ylabel("PSD")
# ax00.set_title("Pulsed-NMR Signal Amplitude")

#############################################################################
# put a mark of script information on the figure
# Get the script name and path automatically
script_path = os.path.abspath(__file__)

# Add the annotation to the figure
plt.annotate(
    f"Generated by: {script_path}",
    xy=(0.02, 0.02),
    xycoords="figure fraction",
    fontsize=8,
    color="gray",
)
# #############################################################################
# plt.tight_layout()
# plt.savefig('example figure - one-column.png', transparent=False)
plt.show()
