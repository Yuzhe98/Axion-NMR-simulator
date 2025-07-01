import numpy as np
import matplotlib.pyplot as plt

# Sampling parameters
fs = 1000       # Sampling frequency in Hz
T = 1.0         # Duration in seconds
N = int(fs * T) # Total number of samples
t = np.arange(N) / fs

# Frequency to synthesize
f_target = 50  # 50 Hz

# Create a zero-filled complex spectrum
spectrum = np.zeros(N, dtype=complex)

# Get FFT bin corresponding to 50 Hz
bin_index = int(f_target * N / fs)

# Set amplitude and phase (real-valued sine wave = symmetric spectrum)
amplitude = N / 2  # for sine amplitude = 1 (scaling by N/2)
spectrum[bin_index] = -1j * amplitude  # -i for sine wave
spectrum[-bin_index] = 1j * amplitude  # conjugate symmetric

plt.plot(np.abs(spectrum), label='spectrum')
plt.xlabel('freq')
plt.ylabel('Amplitude')
# plt.title('50 Hz Signal from Frequency Domain via IFFT')
plt.grid(True)
plt.legend()
plt.show()

# Inverse FFT to get time-domain signal
y = np.fft.ifft(spectrum)

# Plot the real part (imaginary part should be ~0)
plt.plot(t, y.real, label='Reconstructed 50 Hz sine wave')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('50 Hz Signal from Frequency Domain via IFFT')
plt.grid(True)
plt.legend()
plt.show()
