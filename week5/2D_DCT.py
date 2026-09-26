import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import dctn

def plot_2d_dct():
    image_files = ['images/smooth.png', 'images/vertical_stripes.png', 'images/checkerboard.png']
    
    for file_name in image_files:
        try:
            img = plt.imread(file_name)
            
            # Convert to grayscale if necessary
            if img.ndim == 3:
                img = np.dot(img[..., :3], [0.2989, 0.5870, 0.1140])
                
            # Perform 2D DCT
            img_dct = dctn(img, type=2, norm='ortho')
            dct_log_mag = np.log(np.abs(img_dct) + 1)
            
            # Create a separate figure window for each image
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
            
            # Plot Original Image
            ax1.imshow(img, cmap='gray')
            ax1.set_title(f'Original: {file_name}')
            ax1.axis('off')
            
            # Plot DCT Spectrum
            im = ax2.imshow(dct_log_mag, cmap='viridis')
            ax2.set_title('2D DCT Spectrum (Log Magnitude)')
            ax2.set_xlabel('Horizontal Frequency (u)')
            ax2.set_ylabel('Vertical Frequency (v)')
            fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
            
            plt.tight_layout()
            
        except FileNotFoundError:
            print(f"File not found: {file_name}")

    # Display all the generated windows at once
    plt.show()

if __name__ == "__main__":
    plot_2d_dct()
