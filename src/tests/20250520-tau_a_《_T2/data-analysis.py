import os
import sys

os.chdir("src")  # 
print(os.path.abspath(os.curdir))
sys.path.insert(0, os.path.abspath(os.curdir))
os.chdir("..")  # go to parent folder

import pandas as pd
import numpy as np
import time
from SimuTools import Sample, MagField, Simulation, gate
from DataAnalysis import DualChanSig
from functioncache import check, GiveDateandTime

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


import numpy as np

# Load the .npy file
# savedir = rf"C:\Users\zhenf\D\Mainz\CASPEr\20250520-tau_a_《_T2/"
savedir = rf"Tests\20250520-tau_a_《_T2/"
# fname = "m_transverse_all_runs_20250520_145634"  # rand_seed not None, 300s duration, Brms=1e-10
# fname = "m_transverse_all_runs_20250520_151127"  # rand_seed not None, 100s duration, Brms=1e-10
# fname = "m_transverse_all_runs_20250520_151424"  # rand_seed None, 100s duration, Brms=1e-10
fname = "m_transverse_all_runs_20250520_153449"
data = np.load(savedir + fname + ".npz")
# plotRate = 1000

# data = np.load("sim_data.npz", allow_pickle=True)

print(data["m_t"].shape)  # (1000,)
print(data["simuRate"])  # 1000
print(data["duration"])  # 1.0
print(data["T2"])  # 1.0
print(data["Brms"])  # 1.0

m_t_avg = np.mean(data["m_t"], axis=0)
check(m_t_avg.shape)

m_t_std = np.std(data["m_t"], axis=0)
check(m_t_std.shape)

fig = plt.figure(figsize=(6.0, 4.0), dpi=150)  # initialize a figure
gs = gridspec.GridSpec(nrows=1, ncols=1)  # create grid for multiple figures
# #############################################################################
# to specify heights and widths of subfigures
# width_ratios = [1, 1]
# height_ratios = [1]
# gs = gridspec.GridSpec(nrows=1, ncols=2, \
#   width_ratios=width_ratios, height_ratios=height_ratios)  # create grid for multiple figures
# #############################################################################
# # fix the margins
# left=0.171
# bottom=0.202
# right=0.952
# top=0.983
# wspace=0.24
# hspace=0.114
# fig.subplots_adjust(left=left, top=top, right=right,
#                     bottom=bottom, wspace=wspace, hspace=hspace)
# #############################################################################
# plotintv = len(data["timeStamp"] / )
ax00 = fig.add_subplot(gs[0, 0])
# ax00.errorbar(
#     x=data["timeStamp"], y=m_t, yerr=m_t_std, label=""
# )
ax00.plot(data["timeStamp"], m_t_avg, label="tipping angle")
# ax00.plot(x=data["timeStamp"], y=m_t-m_t_std, label="")
# ax00.plot(x=data["timeStamp"], y=m_t+m_t_std, label="")


# ax00.fill_between(
#     data["timeStamp"],
#     m_t - m_t_std,
#     m_t + m_t_std,  #  where = ,
#     color="r",
#     alpha=0.2,
#     zorder=6,
# )
print(data["gyroratio"] * data["Brms"])
check(data["gyroratio"] * data["Brms"])

time_sections = [[0, data["T2"] / 10],  [data["T2"] / 3,data["T2"]]]
T_2_idx = int(len(data["timeStamp"]) * data["T2"] / data["duration"])
time_sections_idx = [[0, T_2_idx // 10], [T_2_idx // 10, T_2_idx //2]]

time_section0 = data["timeStamp"][time_sections_idx[0][0]: time_sections_idx[0][1]]
approx_line0 = data["gyroratio"] * data["Brms"] * time_section0
approx_line0 /= np.mean(approx_line0)
approx_line0 *= np.mean(m_t_avg[time_sections_idx[0][0] : time_sections_idx[0][1]])
ax00.plot(time_section0, approx_line0, label="$\\propto t$")

time_section1 = data["timeStamp"][time_sections_idx[1][0]: time_sections_idx[1][1]]
approx_line1 = data["gyroratio"] * data["Brms"] * np.sqrt(time_section1 * 1. /(1.2 * np.pi ))
approx_line1 /= np.mean(approx_line1)
approx_line1 *= np.mean(m_t_avg[time_sections_idx[1][0] : time_sections_idx[1][1]])
ax00.plot(time_section1, approx_line1, label="$\\propto (t \\tau_a)^{1/2}$")
# check(approx_line0[0:10])


ymin, ymax = ax00.get_ylim()

ax00.legend()
# ax00.plot( data["m_t"], label="")
ax00.set_xlabel("time (s)")
ax00.set_ylabel("tipping angle (rad)")
# ax00.set_xscale('log')
# ax00.set_yscale('log')

# ax00.vlines([data["T2"] / 10, data["T2"]], ymin=ymin, ymax=ymax, linestyles='dashed')


# Vertical lines and their labels
vline_positions = [data["T2"] / 10, data["T2"]]
vline_labels = ["$T_2 / 10$", "$T_2$"]
# Draw vertical lines with labels
for xpos, label in zip(vline_positions, vline_labels):
    ax00.axvline(x=xpos, color="k", linestyle="--", alpha=1)
    ax00.text(
        xpos,
        1.05,
        label,
        rotation=90,
        verticalalignment="bottom",
        horizontalalignment="center",
        color="k",
    )

# ax00.legend()
# #############################################################################
# fig.suptitle("", wrap=True)
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


# print("Shape:", data.shape)  # e.g., (100, 1000) → 100 runs, 1000 time points each
# print("Dtype:", data.dtype)  # e.g., complex128, float64, etc.
# print("First few values:\n", data[0][:10])  # Preview first 10 values from the first run
# import matplotlib.pyplot as plt


# Load it back

# df = pd.read_pickle(savedir + f"m_transverse_all_runs_" + timestr + ".pkl")
# # print("time:", df.attrs["time"][0:10])
# print("result:", df.attrs["result"][0:10])
# print("simuRate:", df.attrs["simuRate"])
# print("duration:", df.attrs["duration"])
# print(df.head())


# plt.plot(np.abs(data[0]))  # magnitude of transverse magnetization
# plt.title("Run 0: Magnitude of Transverse Magnetization")
# plt.xlabel("Time")
# plt.ylabel("|M_transverse|")
# plt.grid(True)
# plt.show()
