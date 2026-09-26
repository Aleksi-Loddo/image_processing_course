import numpy as np
import matplotlib.pyplot as plt

def plot_basis_functions():
    N = 8
    n = np.arange(N)
    
    # DCT-II Functions 
    alpha_0 = np.sqrt(1/N)
    alpha_k = np.sqrt(2/N)
    
    # DCT k=0
    dct_k0 = alpha_0 * np.cos(np.pi / N * (n + 0.5) * 0)
    # DCT k=1
    dct_k1 = alpha_k * np.cos(np.pi / N * (n + 0.5) * 1)
    
    # DFT Functions 
    dft_k0 = np.exp(1j * 2 * np.pi * 0 * n / N)
    dft_k1 = np.exp(1j * 2 * np.pi * 1 * n / N)
    
    # 1. DCT Figure
    fig1, axs1 = plt.subplots(1, 2, figsize=(10, 4))
    fig1.suptitle('DCT-II Basis Functions')
    axs1[0].stem(n, dct_k0)
    axs1[0].set_title('k=0')
    axs1[0].set_ylim([-1, 1])
    axs1[0].grid(True)
    
    axs1[1].stem(n, dct_k1)
    axs1[1].set_title('k=1')
    axs1[1].set_ylim([-1, 1])
    axs1[1].grid(True)

    # 2. DFT Real Figure
    fig2, axs2 = plt.subplots(1, 2, figsize=(10, 4))
    fig2.suptitle('DFT Basis Functions (Real Part)')
    axs2[0].stem(n, np.real(dft_k0), linefmt='b-', markerfmt='bo')
    axs2[0].set_title('k=0')
    axs2[0].set_ylim([-1.5, 1.5])
    axs2[0].grid(True)
    
    axs2[1].stem(n, np.real(dft_k1), linefmt='b-', markerfmt='bo')
    axs2[1].set_title('k=1')
    axs2[1].set_ylim([-1.5, 1.5])
    axs2[1].grid(True)

    # 3. DFT Imag Figure
    fig3, axs3 = plt.subplots(1, 2, figsize=(10, 4))
    fig3.suptitle('DFT Basis Functions (Imaginary Part)')
    axs3[0].stem(n, np.imag(dft_k0), linefmt='r-', markerfmt='ro')
    axs3[0].set_title('k=0')
    axs3[0].set_ylim([-1.5, 1.5])
    axs3[0].grid(True)
    
    axs3[1].stem(n, np.imag(dft_k1), linefmt='r-', markerfmt='ro')
    axs3[1].set_title('k=1')
    axs3[1].set_ylim([-1.5, 1.5])
    axs3[1].grid(True)

    # 4. DFT Magnitude Figure
    fig4, axs4 = plt.subplots(1, 2, figsize=(10, 4))
    fig4.suptitle('DFT Basis Functions (Magnitude)')
    axs4[0].stem(n, np.abs(dft_k0), linefmt='k-', markerfmt='ko')
    axs4[0].set_title('k=0')
    axs4[0].set_ylim([0, 1.5])
    axs4[0].grid(True)
    
    axs4[1].stem(n, np.abs(dft_k1), linefmt='k-', markerfmt='ko')
    axs4[1].set_title('k=1')
    axs4[1].set_ylim([0, 1.5])
    axs4[1].grid(True)

    plt.show()

if __name__ == "__main__":
    plot_basis_functions()
