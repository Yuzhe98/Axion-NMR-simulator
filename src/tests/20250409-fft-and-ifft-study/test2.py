import os
import sys

# print(os.path.abspath(os.curdir))
# os.chdir("..")  # go to parent folder
# os.chdir("..")  # go to parent folder
# print(os.path.abspath(os.curdir))
sys.path.insert(0, os.path.abspath(os.curdir))

from functioncache import stdLIAPSD

import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import uniform

samprate = 1130
dt = 1 / samprate
timeLen = samprate * 10
t = np.arange(timeLen) * dt


frequnecies: np.ndarray = np.fft.fftfreq(len(t), d=dt)
frequnecies = np.sort(frequnecies)
# plt.figure()
# plt.plot(frequnecies)
# plt.title('frequnecies')
# plt.legend()
# plt.show()
# ax_fft_amp = Lorentzian(frequnecies, center=5, FWHM=20)
ax_fft_amp = np.zeros(len(frequnecies))
ax_fft_amp[int((samprate / 2 + 50) / samprate * timeLen)] = 1
del frequnecies
rvs_phase = np.exp(1j * uniform.rvs(loc=0, scale=2 * np.pi, size=len(t)))
length = len(ax_fft_amp)
ax_fft_amp = np.array([ax_fft_amp[length // 2 :], ax_fft_amp[: length // 2]]).flatten()
signal = np.fft.ifft(ax_fft_amp)

plt.figure()
plt.plot(t, signal.real)
plt.title("signal.real")
plt.legend()
plt.show()

plt.figure()
plt.plot(np.abs(np.fft.fft(signal)))
plt.title("np.abs(np.fft.fft(signal))")
plt.legend()
plt.show()

# method 0
newfreq, newASD = stdLIAPSD(
    data_x=signal.real,
    data_y=signal.imag,
    samprate=1 / dt,
    demodfreq=0.0,
)
plt.figure()
plt.plot(newfreq, newASD, "--")
plt.title("stdLIAPSD")
plt.show()

# method 1
data_x = signal.real
data_y = signal.imag
freqs: np.ndarray = np.fft.fftfreq(len(data_x), d=dt)
FFT = np.fft.fft(signal, norm=None)
PSD = np.abs(FFT) ** 2.0
plt.figure()
# plt.plot(np.sort(freqs), PSD)
plt.scatter(freqs, PSD)
plt.show()
