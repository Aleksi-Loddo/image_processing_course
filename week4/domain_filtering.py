import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
def main():
   
    image_path = os.path.join('images', 'fox.jpg')
    
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error: Could not load '{image_path}'.")
        return
   
    f = np.fft.fft2(img)
    
    fshift = np.fft.fftshift(f)
    
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)
   
    rows, cols = img.shape
    crow, ccol = rows // 2, cols // 2  
    mask = np.zeros((rows, cols), np.uint8)
    r = 50 
    
    cv2.circle(mask, (ccol, crow), r, 1, thickness=-1)
    
   
    fshift_filtered = fshift * mask
    
   
    magnitude_spectrum_filtered = 20 * np.log(np.abs(fshift_filtered) + 1)
   
    f_ishift = np.fft.ifftshift(fshift_filtered)
   
    img_back = np.fft.ifft2(f_ishift)
    
    img_back = np.abs(img_back)
    
    plt.figure(figsize=(12, 10))
    plt.subplot(221), plt.imshow(img, cmap='gray')
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(222), plt.imshow(magnitude_spectrum, cmap='gray')
    plt.title('Original Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
    plt.subplot(223), plt.imshow(img_back, cmap='gray')
    plt.title('Filtered Image (Low Pass)'), plt.xticks([]), plt.yticks([])
    plt.subplot(224), plt.imshow(magnitude_spectrum_filtered, cmap='gray')
    plt.title('Filtered Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
    plt.tight_layout()
    plt.show()
if __name__ == "__main__":
    main()
