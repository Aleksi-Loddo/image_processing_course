import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

def black_white_histogram(pixels):
    # intialize histogram array with 256 zeros
    histogram = [0]*256

    height, width = pixels.shape[:2]
    # loop trought image array
    for i in range(height):  
        for j in range(width):
            # add value to histogram array
            gray_value = int(pixels[i, j, 0])
            histogram[gray_value] += 1
       
    return histogram

def get_cdf(histogram):
    cdf = np.cumsum(histogram)
    # Normalize by the total number of pixels
    return cdf / cdf[-1]

# 3. Function to compute the mapping from Image 1 to Image 2
def histogram_matching_mapping(hist_A, hist_B):
    cdf_A = get_cdf(hist_A)
    cdf_B = get_cdf(hist_B)
    
    mapping = np.zeros(256, dtype=int)
    for a in range(256):
        b = 0
        while b < 255 and cdf_B[b] < cdf_A[a]:
            b += 1
        mapping[a] = b
        
    return mapping




picture_path_1 = "pictures/flower.jpg"
picture_path_2 = "pictures/dark_flower.jpg"

picture_1  = Image.open(picture_path_1)
picture_2  = Image.open(picture_path_2)

pixels_1 = np.array(picture_1)
pixels_2 = np.array(picture_2)

hist_1 = black_white_histogram(pixels_1)
hist_2 = black_white_histogram(pixels_2)


mapping = histogram_matching_mapping(hist_1, hist_2)

matched_pixels = np.zeros_like(pixels_1)
height, width = pixels_1.shape[:2]

for i in range(height):
    for j in range(width):
        old_val = pixels_1[i, j, 0]
        new_val = mapping[old_val]
        matched_pixels[i, j] = [new_val, new_val, new_val]


matched_image = Image.fromarray(matched_pixels.astype(np.uint8))
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.title("Original (Flower)")
plt.imshow(picture_1)
plt.subplot(1, 3, 2)
plt.title("Target (Dark Flower)")
plt.imshow(picture_2)
plt.subplot(1, 3, 3)
plt.title("Matched Image")
plt.imshow(matched_image)
plt.show()
