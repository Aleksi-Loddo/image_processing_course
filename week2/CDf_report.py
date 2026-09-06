
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


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

picture_1  = Image.open(picture_path_1)

pixels_1 = np.array(picture_1)


def plot_cdf():
    histogram = black_white_histogram(pixels_1)
    cdf = np.cumsum(histogram) / len(histogram)
    plt.plot(cdf, color='blue')
    plt.xlabel('Pixel Intensity (0 to 255)')
    plt.ylabel('CDF')
    plt.title('CDF via Sorting')
    plt.grid()
    plt.show()

plot_cdf()