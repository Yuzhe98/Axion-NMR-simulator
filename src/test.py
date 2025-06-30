import numpy as np
import matplotlib.pyplot as plt


T2 = 1
timeStamp = np.linspace(0, 4, num=500)
decaySignal = np.empty((len(timeStamp), len(timeStamp)))
for i, decay_1D in enumerate(decaySignal):
    decay_1D = np.exp(-timeStamp / T2) * np.sin(
        2 * np.pi * 10 * timeStamp + i * 2 * np.pi / 100
    )
# Create a simple 2D signal: white square in black background
image = decaySignal
# image[24:40, 24:40] = 1  # white square

# Apply 2D FFT
f_image = np.fft.fft2(image)

# Shift the zero-frequency component to the center
f_image_shifted = np.fft.fftshift(f_image)

# Visualize
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.imshow(image, cmap='gray')
plt.title("Original Image")
plt.axis('off')

plt.subplot(1,2,2)
plt.imshow(np.log(1 + np.abs(f_image_shifted)), cmap='gray')
plt.title("Magnitude Spectrum (log scale)")
plt.axis('off')
plt.show()
