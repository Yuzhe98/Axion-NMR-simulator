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


dt = 1 / 1130
t = np.arange(256 * 8) * dt

freq0 = -50
s = np.cos(2 * np.pi * freq0 * t) + 1j * np.sin(2 * np.pi * freq0 * t)
s += 2 * (np.cos(2 * np.pi * 66 * t) + 1j * np.sin(2 * np.pi * 66 * t))
# newfreq, newASD = np.sort(freq), np.abs(S_f)[np.argsort(freq)]

data_x = s.real
data_y = s.imag
freqs: np.ndarray = np.fft.fftfreq(len(data_x), d=dt)
FFT = np.fft.fft(s, norm=None)
PSD = np.abs(FFT) ** 2.0
plt.figure()
plt.plot(freqs, PSD)
plt.show()

newfreq, newASD = stdLIAPSD(
    data_x=s.real,
    data_y=s.imag,
    samprate=1 / dt,
    demodfreq=0.0,
)
plt.figure()
plt.plot(newfreq, newASD, "--")
# plt.scatter(freq, S_f.imag)
plt.legend()
plt.show()
