import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import dct, idct, fft, ifft

def zero_spectral_comp():
    # Generate 8-sample linear ramp
    N = 8
    x = np.arange(N, dtype=float)
    
    # 2. Perform DCT and DFT
    X_dct = dct(x, type=2, norm='ortho')
    X_dft = fft(x)
    
    # Clear half of the non-redundant coefficients with smallest magnitude
    
    #  DCT 
   
    X_dct_zeroed = X_dct.copy()
   
    idx_dct_smallest = np.argsort(np.abs(X_dct_zeroed))[:4]
    X_dct_zeroed[idx_dct_smallest] = 0
    
    #  DFT 
    X_dft_zeroed = X_dft.copy()
    non_redundant_mags = np.abs(X_dft_zeroed[:N//2 + 1])
    idx_dft_smallest = np.argsort(non_redundant_mags)[:2]
    
    for k in idx_dft_smallest:
        X_dft_zeroed[k] = 0
       
        if k > 0 and k < N/2:
            X_dft_zeroed[N - k] = 0
            
    # Perform inverse transforms
    x_rec_dct = idct(X_dct_zeroed, type=2, norm='ortho')
    x_rec_dft = np.real(ifft(X_dft_zeroed)) 
    
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(x, 'k-o', label='Original Signal', linewidth=2, markersize=8)
    plt.plot(x_rec_dct, 'b--x', label='Reconstructed DCT (4 zeroed)', linewidth=2, markersize=8)
    plt.plot(x_rec_dft, 'r-.s', label='Reconstructed DFT (2 non-red. pairs zeroed)', linewidth=2, markersize=8)
    
    plt.title('Reconstruction after Zeroing Smallest Spectral Components')
    plt.xlabel('Sample Index (n)')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    zero_spectral_comp()
