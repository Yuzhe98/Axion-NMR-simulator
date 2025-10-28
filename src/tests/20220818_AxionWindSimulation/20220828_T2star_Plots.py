import os

# from random import sample
import sys

print(os.path.abspath(os.curdir))
# os.chdir("..")  # go to parent folder
# os.chdir("..")  # go to parent folder
# print(os.path.abspath(os.curdir))
sys.path.insert(0, os.path.abspath(os.curdir))
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from SimuTools import (
    TestSample10MHzT,
)  # Xe129, Methanol, , Mainz, TestStation, MagVec, AxionWind

from functioncache import check

# data from 2022-08-25
taua = 1.0
T2star_arr_0825 = np.array(
    [
        1.00000000e-04,
        3.16227766e-04,
        1.00000000e-03,
        3.16227766e-03,
        1.00000000e-02,
        3.16227766e-02,
        1.00000000e-01,
        3.16227766e-01,
        1.00000000e00,
        3.16227766e00,
        1.00000000e01,
        3.16227766e01,
        1.000000e02,
        199.52623149688787,
        316.22776601683796,
        1000.0,
        3162.27766017,
        10000.0,
        5.62341325,
        17.7827941,
        56.23413252,
        177.827941,
    ]
)

avg_sqrtMxysq_arr_0825 = np.array(
    [
        6.00729525403692e-10,
        1.892680563138194e-09,
        6.320670227486919e-09,
        1.7849948295486614e-08,
        5.722704219768959e-08,
        1.831828483075051e-07,
        4.02276477825698e-07,
        6.488768660724763e-07,
        1.1414089977788986e-06,
        1.2433839539645424e-06,
        2.2376262525500237e-06,
        1.4880993433176427e-06,
        2.2333038491821554e-06,
        4.196305375588696e-06,
        1.0291657069639695e-05,
        9.103859068716038e-06,
        3.489572228807298e-06,
        4.389712567315872e-06,
        1.8936683372404932e-06,
        3.446406711243213e-06,
        2.9824055357928308e-06,
        2.493152043067911e-06,
    ]
)

avg_sqrtBALPsq_arr_0825 = np.array(
    [
        9.563204636468877e-14,
        9.528052775590814e-14,
        1.0070631992363127e-13,
        9.019590747340721e-14,
        9.35195829409311e-14,
        1.0462683527124416e-13,
        9.894666698325927e-14,
        9.779880509450935e-14,
        9.514599957803521e-14,
        9.376164395791295e-14,
        1.0510331838693979e-13,
        9.614715461955493e-14,
        9.704109486569754e-14,
        9.271410990648845e-14,
        9.349755671661435e-14,
        8.828572572597242e-14,
        1.0037410089447239e-13,
        8.901933921516225e-14,
        8.012830385160042e-14,
        1.0060984404189555e-13,
        8.951541498537557e-14,
        9.335857934389245e-14,
    ]
)
check(T2star_arr_0825.shape)
check(avg_sqrtMxysq_arr_0825.shape)
check(avg_sqrtBALPsq_arr_0825.shape)
T2star_arr = 10 ** (
    np.array([-2.0, 0, 2])
)  # -4., -3.5, -3., -2.5, -2., -1.5, -1., -0.5, 0., 0.5, 1., 1.5
detuning_arr = np.array([0.0, 5.0])

plt.rc("font", size=10)
plt.rcParams["font.family"] = "Times New Roman"
# plt.rcParams["font.family"] = "serif"
# plt.rcParams["font.serif"] = ["Times New Roman"]
plt.rcParams["mathtext.fontset"] = "cm"  # 'dejavuserif'
fig = plt.figure(figsize=(5.8 * 0.8, 6.5 * 0.8), dpi=200)  #
gs = gridspec.GridSpec(
    nrows=5, ncols=2, height_ratios=[1, 1, 1, 0.1, 1.4]
)  # , width_ratios = [1.5, 1.5], height_ratios = [1,1,1,1]
fig.subplots_adjust(
    top=0.95, bottom=0.095, left=0.135, right=0.875, hspace=0.32, wspace=0.195
)


ax0 = fig.add_subplot(gs[0, 0])
ax1 = fig.add_subplot(gs[0, 1])
ax2 = fig.add_subplot(gs[1, 0])
ax3 = fig.add_subplot(gs[1, 1])
ax4 = fig.add_subplot(gs[2, 0])
ax5 = fig.add_subplot(gs[2, 1])

# BALPcolor = [231./256, 127./256, 0./256]
# darkorange [210./256, 142./256, 57./256]
# darker orange [241./256, 132./256, 0./256]
# darker than darker orange [204./256, 112./256, 0./256]
# trjrycolor = 'dodgerblue'  # 'tab:blue'

trjrycolor = [231.0 / 256, 127.0 / 256, 0.0 / 256]
trjrylabelcolor = [220.0 / 256, 120.0 / 256, 0.0 / 256]
# darkorange [210./256, 142./256, 57./256]
# darker orange [241./256, 132./256, 0./256]
# darker than darker orange [204./256, 112./256, 0./256]
BALPcolor = [
    73.0 / 256,
    141.0 / 256,
    206.0 / 256,
]  # 'tab:blue' slateblue  'cornflowerblue'
BALPlabelcolor = [55.0 / 256, 107.0 / 256, 156.0 / 256]

title_list = [
    f"(a) $T_2^*$ = 0.01 s. Detuning = {detuning_arr[0]:.0f} Hz",
    f"(b) $T_2^*$ = 0.01 s. Detuning = {detuning_arr[1]:.0f} Hz",
    f"(c) $T_2^*$ = 1 s. Detuning = {detuning_arr[0]:.0f} Hz",
    f"(d) $T_2^*$ = 1 s. Detuning = {detuning_arr[1]:.0f} Hz",
    f"(e) $T_2^*$ = 100 s. Detuning = {detuning_arr[0]:.0f} Hz",
    f"(f) $T_2^*$ = 100 s. Detuning = {detuning_arr[1]:.0f} Hz",
]
plotintv = 1
filepath = "Supplementary/20220818_AxionWindSimulation/20220828Results/"
for i, ax in enumerate([ax0, ax2, ax4]):
    # 1MHz_T2star_{T2star:.2e}_detuning_{detuning:.2f}_timestamp.txt', magnetization.timestamp)
    # 1MHz_T2star_{T2star:.2e}_detuning_{detuning:.2f}_BALP.txt', magnetization.trjry)
    # 1MHz_T2star_{T2star:.2e}_detuning_{detuning:.2f}_trjry.txt', magnetization.BALP_array)

    timestamp = np.loadtxt(
        filepath
        + f"1MHz_T2star_{T2star_arr[i]:.2e}_detuning_{detuning_arr[0]:.2f}_timestamp.txt"
    )
    trjry = np.loadtxt(
        filepath
        + f"1MHz_T2star_{T2star_arr[i]:.2e}_detuning_{detuning_arr[0]:.2f}_trjry.txt"
    )
    BALP = np.loadtxt(
        filepath
        + f"1MHz_T2star_{T2star_arr[i]:.2e}_detuning_{detuning_arr[0]:.2f}_BALP.txt"
    )
    # ax.plot(GammaandSAmp_arr[:, 0], GammaandSAmp_arr, label='PSD Signal Amp', color='tab:cyan', alpha=1)
    ax.plot(
        timestamp[0:-1:plotintv],
        1e13 * BALP[0:-1:plotintv, 0],
        label="",
        color=BALPcolor,
        alpha=1,
        lw=0.8,
    )
    # ax.plot(timestamp[0:-1:plotintv], trjry[1:-1:plotintv, 1], label='', color='red', alpha=1)
    ax.set_ylabel(
        "$\\mathrm{B}_{a,t}$ [$10^{-13}$T]", color=BALPlabelcolor
    )  # \\gamma  / 2
    ax.set_xlim(0, 5)
    ax.set_ylim(-2.5, 2.5)
    ax.grid()

    ax.set_xticks([0, 1, 2, 3, 4, 5])
    ax.set_yticks([-2, 0, 2])
    ax_twinx = ax.twinx()
    if i == 0:
        ax_twinx.plot(
            timestamp[0:-1:plotintv],
            1e1 * 1e5 * trjry[1:-1:plotintv, 1],
            label="",
            color=trjrycolor,
            alpha=1,
        )
        ax_twinx.text(x=3.5, y=0.8, s="$\\times 10$", fontsize=10, color=trjrycolor)
    else:
        ax_twinx.plot(
            timestamp[0:-1:plotintv],
            1e5 * trjry[1:-1:plotintv, 1],
            label="",
            color=trjrycolor,
            alpha=1,
        )
    # ax_twinx.set_ylabel('$M_{xy} / M_0$', color = trjrycolor)
    ax_twinx.set_xlim(0, 5)
    ax_twinx.set_ylim(-1.25, 1.25)
    ax_twinx.set_yticks([])


ax4.set_xlabel("Time [s]")
# ax.set_xticks([0, 2, 4, 6, 8, 10])
ax4.set_xticks([0, 1, 2, 3, 4, 5])

for i, ax in enumerate([ax1, ax3, ax5]):
    timestamp = np.loadtxt(
        filepath
        + f"1MHz_T2star_{T2star_arr[i]:.2e}_detuning_{detuning_arr[1]:.2f}_timestamp.txt"
    )
    trjry = np.loadtxt(
        filepath
        + f"1MHz_T2star_{T2star_arr[i]:.2e}_detuning_{detuning_arr[1]:.2f}_trjry.txt"
    )
    BALP = np.loadtxt(
        filepath
        + f"1MHz_T2star_{T2star_arr[i]:.2e}_detuning_{detuning_arr[1]:.2f}_BALP.txt"
    )
    # ax.plot(GammaandSAmp_arr[:, 0], GammaandSAmp_arr, label='PSD Signal Amp', color='tab:cyan', alpha=1)
    ax.plot(
        timestamp[0:-1:plotintv],
        1e13 * BALP[0:-1:plotintv, 0],
        label="",
        color=BALPcolor,
        alpha=1,
        lw=0.8,
    )  # darkorange
    # ax.plot(timestamp[0:-1:plotintv], trjry[1:-1:plotintv, 1], label='', color='red', alpha=1)
    # ax.set_ylabel('$\\gamma \\mathrm{B}_{a,t}} / 2$', color='darkorange')
    ax.set_xlim(0, 5)
    ax.set_ylim(-2.5, 2.5)
    ax.grid(True)
    ax.set_xticks([0, 1, 2, 3, 4, 5])
    # ax.set_xticks([])
    ax.set_yticks([-2, 0, 2], color="white")
    # ax.yaxis.set_major_locator(plt.NullLocator())
    # ax.xaxis.set_major_formatter(plt.NullFormatter())
    ax_twinx = ax.twinx()
    if i == 0:
        ax_twinx.plot(
            timestamp[0:-1:plotintv],
            1e1 * 1e5 * trjry[1:-1:plotintv, 1],
            label="",
            color=trjrycolor,
            alpha=1,
        )
        ax_twinx.text(x=4, y=0.8, s="$\\times 10$", fontsize=10, color=trjrycolor)
    else:
        ax_twinx.plot(
            timestamp[0:-1:plotintv],
            1e5 * trjry[1:-1:plotintv, 1],
            label="",
            color=trjrycolor,
            alpha=1,
        )
    ax_twinx.set_ylabel(
        "$M_{xy} [10^{-5}\\,M_0]$", color=trjrylabelcolor
    )  # \\times 10^7
    # ax_twinx.set_xlim(1e-13, 1e0)
    # ax_twinx.grid(True)
    ax_twinx.set_ylim(-1.25, 1.25)
    ax_twinx.set_yticks([-1, 0, 1])
    # for tick in ax.xaxis.get_major_ticks():
    #     tick.tick1line.set_visible(False)
    #     tick.tick2line.set_visible(False)
    #     tick.label1.set_visible(False)
    #     tick.label2.set_visible(False)

for i, ax in enumerate([ax0, ax1, ax2, ax3, ax4, ax5]):
    ax.set_title(title_list[i], fontsize=7)

for i, ax in enumerate([ax0, ax1, ax2, ax3]):
    for tick in ax.xaxis.get_major_ticks():
        tick.tick1line.set_visible(False)
        tick.tick2line.set_visible(False)
        tick.label1.set_visible(False)
        tick.label2.set_visible(False)
for i, ax in enumerate([ax1, ax3, ax5]):
    for tick in ax.yaxis.get_major_ticks():
        tick.tick1line.set_visible(False)
        tick.tick2line.set_visible(False)
        tick.label1.set_visible(False)
        tick.label2.set_visible(False)

ax5.set_xlabel("Time [s]")
ax5.set_xticks([0, 1, 2, 3, 4, 5])
avg_sqrtMxysq_arr_0825 = avg_sqrtMxysq_arr_0825[np.argsort(T2star_arr_0825)]
avg_sqrtBALPsq_arr_0825 = avg_sqrtBALPsq_arr_0825[np.argsort(T2star_arr_0825)]
T2star_arr_0825 = np.sort(T2star_arr_0825)

make_ax6 = True
if make_ax6:
    # ax6 = fig.add_subplot(gs[3,:])
    # axempty = fig.add_subplot(gs[3,:])
    ax6 = fig.add_subplot(gs[4, :])
    ax6.plot(
        T2star_arr_0825,
        np.pi * T2star_arr_0825 / 2,
        label="$\\tau_s N_{eff}/N_{all} = \\pi T_2^*/2$",
        color="tab:cyan",
        alpha=1,
        linestyle="--",
    )
    ax6.hlines(
        y=taua,
        xmin=1e-5,
        xmax=3.8e3,
        colors="darkred",
        linestyles="dotted",
        label="$\\tau_s N_{eff}/N_{all} = \\tau_a$",
    )
    ax6.scatter(
        T2star_arr_0825[:-5],
        avg_sqrtMxysq_arr_0825[:-5]
        / (0.5 * TestSample10MHzT.gyroratio * avg_sqrtBALPsq_arr_0825[:-5]),
        label="$\\tau_s = M_{xy} / (M_0 \\gamma \\overline{\\mathrm{B}_{a,t}^2}^{1/2}/2)$\nObtained from simulation",
        marker="x",
        s=40,
        color="tab:green",
        alpha=1,
    )  # \\langle \\rangle
    ax6.set_ylabel(
        "$\\tau_s N_{eff}/N_{all} $"
    )  # |Mxy|/($\mathrm{M}_0\gamma\mathrm{B}_a/2$)
    ax6.set_xlabel("$T_2^*$ / s")
    ax6.set_title("(g) Detuning 0 Hz", fontsize=8)
    ax6.set_xscale("log")
    ax6.set_yscale("log")
    ax6.set_xlim(0.7e-4, 3.8e2)
    ax6.set_ylim(0.7e-4, 3e0)
    ax6.set_yticks([1e-4, 1e-3, 1e-2, 1e-1, 1])
    ax6.set_xticks([1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100])
    ax6.grid()
    ax6.legend(loc="best", fontsize=7)  # bbox_to_anchor=(1.0, 1.0), loc='upper left'
    # plt.tight_layout()
plt.show()
