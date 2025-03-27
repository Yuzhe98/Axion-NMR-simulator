############################################################
# A one-column example figure made by matplotlib
# To better illustrate the settings in matplotlib,
# we follow PHYSICAL REVIEW LETTERS guidelines [1]
# in adjusting figures
#
# Refernce:
# [1] PHYSICAL REVIEW LETTERS, Information for Authors,
# https://journals.aps.org/prl/authors
############################################################
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from matplotlib.ticker import FuncFormatter
from scipy.stats import rayleigh, uniform, norm
import mpltex

cm = 1 / 2.56  # convert cm to inch
hbar = 4.135667696e-15  # eV / Hz
VtoPhi0 = 10  # Phi0 / V


def Lorentzian(x, center, FWHM, area, offset):
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
    return offset + 0.5 * FWHM * area / (
        np.pi * ((x - center) ** 2 + (0.5 * FWHM) ** 2)
    )


############################################################
# One-column example
############################################################

# data
freqstamp = np.linspace(1.348449e6 - 100, 1.348449e6 + 100, num=50)
FWHM = 20
Lorzlin = Lorentzian(
    x=freqstamp, center=np.mean(freqstamp), FWHM=FWHM, area=1e-4, offset=1e-8
)
PSD_noise = (
    norm.rvs(
        loc=0,
        scale=1e-1 * np.sqrt(np.amax(Lorzlin)),
        size=len(freqstamp),
        random_state=None,
    )
    ** 2
)
NMR_decayspectrum = Lorzlin + PSD_noise
Axion_sensitivity = 1e-12 * 1.0 / np.sqrt(NMR_decayspectrum)

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
ax10 = fig.add_subplot(gs[1, 0])  # , sharex=ax00

ax00.plot(
    freqstamp - np.mean(freqstamp),
    NMR_decayspectrum,
    label="PSD Signal Amp",
    color="tab:blue",
    alpha=1,
    linestyle="-",
)
# ax00.scatter(freqstamp, NMR_decayspectrum, marker='x', s=30, color='tab:black', alpha=1)
ax00.errorbar(
    x=freqstamp - np.mean(freqstamp),
    y=NMR_decayspectrum,
    yerr=np.std(PSD_noise),
    fmt="s",
    color="tab:green",
    linewidth=1,
    markersize=3,
)
# ax00.step(x=, y=, where='post', label='', alpha=1)
ax00.fill_between(
    freqstamp - np.mean(freqstamp),
    NMR_decayspectrum,
    np.amin(NMR_decayspectrum),  #  where = ,
    color="r",
    alpha=0.2,
    zorder=6,
)
# X_Y_Spline = make_interp_spline(x, y)
# X_ = np.linspace(x.min(), x.max(), 500)
# Y_ = X_Y_Spline(X_)
ax00.set_xlabel("Frequency - $1.348\\,449\\times 10^6$ [Hz]")
ax00.set_ylabel("PSD [$\\Phi_0^2/ \\mathrm{Hz}$]")
ax00.set_title("Pulsed-NMR Signal Amplitude")
# ax00.set_xscale('log')
# ax00.set_yscale('log')
# ax00.set_xticks([])
# ax00.set_yticks([])
# ax00.set_xticklabels(('$a$', '$valx$', '$b$'))
# ax00.set_yticklabels(('$a$', '$valx$', '$b$'))
# plt.setp(ax00.get_xticklabels(), visible=False)
# plt.setp(ax00.get_yticklabels(), visible=False)
# ax00.set_xlim(left=, right=)
# ax00.set_ylim(bottom=0, top=)
# ax00.vlines(x=taua, ymin = 1e-5, ymax = 1e3, colors='grey', linestyles='dotted', label='')
# ax00.hlines(y=1 / ((np.pi * homog0 * 1e6) + 1 / T2), xmin = 1e2, xmax = 1e6, colors='black', linestyles='dotted', label='')
# ax00.yaxis.set_major_locator(plt.NullLocator())
# ax00.xaxis.set_major_formatter(plt.NullFormatter())
# set visibility of ticks on the axes
# for tick in ax00.xaxis.get_major_ticks():
#         tick.tick1line.set_visible(False)
#         tick.tick2line.set_visible(False)
#         tick.label1.set_visible(False)
#         tick.label2.set_visible(False)
ax00.grid()  # set gird color, linewidth and etc.
# add an arrow
# ax00.arrow(x=, y=, dx=, \
#         dy=, width=0.02, head_width=0.199,head_length=0.04, color='black', \
#             edgecolor='none',length_includes_head=True, shape='full')
ax00.text(
    x=0, y=np.amax(Lorzlin) * 0.48, s="$\\Delta\\nu$", ha="center", va="top", color="k"
)
ax00.quiver(
    [0, 0],
    [np.amax(Lorzlin) * 0.5, np.amax(Lorzlin) * 0.5],
    [0.5 * FWHM, -0.5 * FWHM],
    [0, 0],
    angles="xy",
    scale_units="xy",
    scale=1,
    color="black",
    width=0.005,
)

# ax00.legend(loc='upper right', ncol=1, frameon=False)  # bbox_to_anchor=(1.0, 1.0),

# set visibility of spines / frames
for pos in ["right", "top", "bottom", "left"]:
    ax00.spines[pos].set_visible(True)


ax10.plot(freqstamp, freqstamp, color="tab:orange", label="", alpha=1)
ax10.scatter(freqstamp, freqstamp, color="tab:orange", marker="x", s=30, alpha=1)
ax10.set_xlabel("")
ax10.set_ylabel("")

# ax01.set_xscale('log')
# ax01.set_yscale('log')
# ax01.set_xlim(-2, 2)
# ax01.set_ylim(-0.05, 1.1)
# ax01.set_xticks([])
# ax01.set_yticks([])
ax10.grid()
ax10.legend(bbox_to_anchor=(1.0, 1.0), loc="upper left")
ax10.quiver(
    [0, -0],
    [0.5, 0.5],
    [0.25, -0.25],
    [0, 0],
    angles="xy",
    scale_units="xy",
    scale=1,
    color="black",
    width=0.015,
)

ax012 = ax10.twinx()
ax012.set_ylabel("", color="tab:red")
ax012.set_yticks([])

# hist, bin_edges = np.histogram(conv_PSD, bins=30)
# for i, count in enumerate(hist):
#     if count > 0:
#         ax21.scatter(bin_edges[i+1], count, color='goldenrod', edgecolors='darkgoldenrod', linewidths=0.8, marker='o', s=2, zorder=6)


fig.suptitle("super title", wrap=True)

# put figure index
letters = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)", "(g)", "(h)", "(i)"]
for i, ax in enumerate([ax00, ax10]):
    xleft, xright = ax00.get_xlim()
    ybottom, ytop = ax00.get_ylim()
    ax.text(x=xleft, y=ytop, s=letters[i], ha="right", va="bottom", color="blue")
# ha = 'left' or 'right'
# va = 'top' or 'bottom'
# #############################################################################
# put a mark of script information on the figure
# Get the script name and path automatically
script_path = os.path.abspath(__file__)

# Add the annotation to the figure
plt.annotate(
    f"Generated by: {script_path}",
    xy=(0.02, 0.02),
    xycoords="figure fraction",
    fontsize=3,
    color="gray",
)
# #############################################################################

plt.tight_layout()
# plt.savefig('example figure - one-column.png', transparent=False)
plt.show()

# colors from Piet Cornelies Mondrian
# RGB
# red 212 1 0
# orange 242 141 2
# light grey 233 226 228
# mid grey 173 189 201
# black 0 0 0
# blue 20 17 93
# yellow 252 215 7
# purple 56 63 131
# dark blue 0 13 47

# dark color list
# 'Dark2'
#
