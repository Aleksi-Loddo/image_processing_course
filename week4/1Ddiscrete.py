import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

fs = 100000 
duration = 0.01  
t = np.arange(0, duration, 1/fs)

#  1kHz sine wave
f1 = 1000
sig1 = np.sin(2 * np.pi * f1 * t)

# 1kHz square wave
sig2 = signal.square(2 * np.pi * f1 * t)

# sum of 1kHz sine wave and 8kHz sine wave
f3 = 8000
sig3 = np.sin(2 * np.pi * f1 * t) + np.sin(2 * np.pi * f3 * t)

def compute_fft(sig, fs):
    N = len(sig)
    freqs = np.fft.fftfreq(N, 1/fs)
    fft_vals = np.fft.fft(sig)

    # Extract only positive frequencies to make the plot readable
    pos_mask = freqs >= 0
    # Normalize the magnitude
    return freqs[pos_mask], np.abs(fft_vals)[pos_mask] / (N/2)



# Calculate FFT for each signal
freqs1, fft1 = compute_fft(sig1, fs)
freqs2, fft2 = compute_fft(sig2, fs)
freqs3, fft3 = compute_fft(sig3, fs)
# Plotting
plt.figure(figsize=(12, 10))


# Signal 1
plt.subplot(3, 2, 1)
plt.plot(t, sig1)
plt.title('1kHz Sine Wave')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.subplot(3, 2, 2)
plt.plot(freqs1, fft1)
plt.title('Spectrum of 1kHz Sine Wave')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.xlim(0, 15000) # Limit x-axis to zoom in on the relevant frequencies
# Signal 2
plt.subplot(3, 2, 3)
plt.plot(t, sig2)
plt.title('1kHz Square Wave')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.subplot(3, 2, 4)
plt.plot(freqs2, fft2)
plt.title('Spectrum of 1kHz Square Wave')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.xlim(0, 15000)
# Signal 3
plt.subplot(3, 2, 5)
plt.plot(t, sig3)
plt.title('Sum of 1kHz and 8kHz Sine Waves')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.subplot(3, 2, 6)
plt.plot(freqs3, fft3)
plt.title('Spectrum of 1kHz + 8kHz Sine Waves')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.xlim(0, 15000)
plt.tight_layout()
plt.show()
