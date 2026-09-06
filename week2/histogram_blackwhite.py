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

picture_path_1 = "pictures/flower.jpg"
picture_path_2 = "pictures/dark_flower.jpg"

picture_1  = Image.open(picture_path_1)
picture_2  = Image.open(picture_path_2)

pixels_1 = np.array(picture_1)
pixels_2 = np.array(picture_2)

hist_1 = black_white_histogram(pixels_1)
hist_2 = black_white_histogram(pixels_2)
# Create a combined figure
plt.figure(figsize=(10, 5))
plt.title("Combined Grayscale Histogram")
plt.xlabel("Pixel Intensity (0 to 255)")
plt.ylabel("Frequency (Number of Pixels)")
plt.plot(hist_1, color='blue', label='Flower')
plt.plot(hist_2, color='red', label='Dark Flower')
plt.xlim([0, 255])
plt.legend()

# Create separate figure for Flower
plt.figure(figsize=(10, 5))
plt.title("Flower Grayscale Histogram")
plt.xlabel("Pixel Intensity (0 to 255)")
plt.ylabel("Frequency (Number of Pixels)")
plt.plot(hist_1, color='blue')
plt.xlim([0, 255])

plt.figure(figsize=(10, 5))
plt.title("Dark Flower Grayscale Histogram")
plt.xlabel("Pixel Intensity (0 to 255)")
plt.ylabel("Frequency (Number of Pixels)")
plt.plot(hist_2, color='red')
plt.xlim([0, 255])

plt.show()
