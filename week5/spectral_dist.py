import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import dct, fft

def analyze_spectral_content():
    # 1. Generate 8-sample linear ramp
    x = np.arange(8)
    
    # 2. Perform DCT (Type-II with orthonormal normalization) and DFT
    X_dct = dct(x, type=2, norm='ortho')
    X_dft = fft(x)
    
    # 3. Plotting results
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Spectral Content of a Linear Ramp (N=8)')
    
    # Original Signal
    axs[0, 0].stem(x)
    axs[0, 0].set_title('Original Signal (Linear Ramp)')
    axs[0, 0].set_xlabel('Sample Index (n)')
    axs[0, 0].grid(True)
    
    # DCT Coefficients
    axs[0, 1].stem(X_dct)
    axs[0, 1].set_title('DCT Coefficients')
    axs[0, 1].set_xlabel('Frequency Index (k)')
    axs[0, 1].grid(True)
    
    # DFT Real Part
    axs[1, 0].stem(np.real(X_dft), linefmt='b-', markerfmt='bo')
    axs[1, 0].set_title('DFT (Real Part)')
    axs[1, 0].set_xlabel('Frequency Index (k)')
    axs[1, 0].grid(True)
    
    # DFT Imaginary Part
    axs[1, 1].stem(np.imag(X_dft), linefmt='r-', markerfmt='ro')
    axs[1, 1].set_title('DFT (Imaginary Part)')
    axs[1, 1].set_xlabel('Frequency Index (k)')
    axs[1, 1].grid(True)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    analyze_spectral_content()
