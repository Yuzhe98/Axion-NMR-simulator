import numpy as np

N = 8
x = np.arange(-N,N)  # just a toy array: [0, 1, 2, 3, 4, 5, 6, 7]

print("Original array (0, +, - order):")
print(x)

print("\nfftshift (→ [-, 0, +] order):")
print(np.fft.fftshift(x))

print("\nifftshift (→ [0, +, -] order again):")
print(np.fft.ifftshift(np.fft.fftshift(x)))
