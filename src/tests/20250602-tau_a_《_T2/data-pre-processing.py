# #####################################
# get sin_theta mean and std for all simulations
# #####################################
import os
import sys

import numpy as np
from tqdm import tqdm
from functioncache import check

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

os.chdir("src")  #
print(os.path.abspath(os.curdir))
sys.path.insert(0, os.path.abspath(os.curdir))
os.chdir("..")  # go to parent folder


# Load the .npy file

savedir = (
    r"C:\Users\zhenf\D\Yu0702\Axion-NMR-simulator\Tests\20250602-tau_a_《_T2\data_0/"
)

fname_head = "theta_all_runs_20250602_134924_"

nu_a_offsets = np.arange(-10, 10, 0.5)

num_runs = len(nu_a_offsets)
simuRate = 500  #
duration = 100
timeLen = int(simuRate * duration)
sin_theta_means = np.empty((num_runs, timeLen), dtype=np.float64)
sin_theta_stds = np.empty((num_runs, timeLen), dtype=np.float64)
for i in tqdm(range(num_runs)):
    data = np.load(savedir + fname_head + f"{i}.npz")
    sin_theta_mean = np.mean(data["sin_theta"], axis=0)
    sin_theta_std = np.std(data["sin_theta"], axis=0)
    sin_theta_means[i] = sin_theta_mean
    sin_theta_stds[i] = sin_theta_std

check(sin_theta_means.shape)

data_file_name = savedir + "theta_all_runs_20250602_134924_summary" + ".npz"
np.savez(
    data_file_name,
    timeStamp=data["timeStamp"],
    sin_theta_means=sin_theta_means,
    sin_theta_stds=sin_theta_stds,
    duration=data["duration"],
    demodfreq=data["demodfreq"],
    T2=data["T2"],
    T1=data["T1"],
    Brms=data["Brms"],
    nu_a_offsets=nu_a_offsets,
    use_stoch=True,
    gyroratio=data["gyroratio"],
)


fig = plt.figure(figsize=(6.0, 4.0), dpi=150)  # initialize a figure
gs = gridspec.GridSpec(nrows=1, ncols=1)  # create grid for multiple figures

ax00 = fig.add_subplot(gs[0, 0])

for i, nu_a in enumerate(tqdm(nu_a_offsets)):
    ax00.plot(data["timeStamp"], sin_theta_means[i], label=f"{nu_a}")


ymin, ymax = ax00.get_ylim()

ax00.legend(bbox_to_anchor=(1.0, 1.0), loc="upper left")
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


plt.tight_layout()
# plt.savefig('example figure - one-column.png', transparent=False)
plt.show()
