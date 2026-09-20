import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def process_and_plot(image_path, title_prefix):
    # Load image and convert to grayscale ('L' mode)
    img = Image.open(image_path).convert('L')
    img_array = np.array(img)
    
    # Compute 2D Fourier Transform
    fourier_transformed_data = np.fft.fft2(img_array)
    
    # Shift the zero-frequency (DC) component to the center
    F_shifted = np.fft.fftshift(fourier_transformed_data)
    
    # Calculate the visual spectrum (magnitude) with logarithmic scaling
    visual_spectrum = np.log(1 + np.abs(F_shifted))
    
    # Plot the original image and its spectrum
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.imshow(img_array, cmap='gray')
    plt.title(f'{title_prefix} - Original Image')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(visual_spectrum, cmap='gray')
    plt.title(f'{title_prefix} - Magnitude Spectrum')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

images = [
    ('images/vl.png', 'Vertical Lines'),
    ('images/hl.png', 'Horizontal Lines'),
    ('images/diag.png', 'Diagonal Lines')
]

for path, title in images:
    process_and_plot(path, title)


