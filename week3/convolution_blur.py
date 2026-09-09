import matplotlib.pyplot as plt
from PIL import Image
import numpy as np


def expand_edges_2d(image, pad_size=2):
    height, width, channel = image.shape

    new_height = height + (2 * pad_size)
    new_width = width +  (2 * pad_size)

    padded_image = np.zeros((new_height, new_width, channel), dtype=image.dtype)

    padded_image[pad_size : pad_size + height, pad_size : pad_size + width] = image

    padded_image[0 : pad_size, pad_size : pad_size + width] = image[0, :]
    
    padded_image[pad_size + height : new_height, pad_size : pad_size + width] = image[-1, :]

    padded_image[pad_size : pad_size + height, 0 : pad_size] = image[:, 0:1]
   
    padded_image[pad_size : pad_size + height, pad_size + width : new_width] = image[:, -1:]

    padded_image[0 : pad_size, 0 : pad_size] = image[0, 0]
    
    padded_image[0 : pad_size, pad_size + width : new_width] = image[0, -1]

    padded_image[pad_size + height : new_height, 0 : pad_size] = image[-1, 0]
   
    padded_image[pad_size + height : new_height, pad_size + width : new_width] = image[-1, -1]

    return padded_image

def convolution_and_blur_filter(image, kernel):
    
    pad_size = kernel.shape[0] // 2
    
    padded_image = expand_edges_2d(image, pad_size)
    
    height, width, channel = image.shape
    
    blurred_image = np.zeros_like(image)
    
    for channel in range(channel):
        
        for i in range(height):
            
            for j in range(width):
               
                neighborhood = padded_image[i : i + 5, j : j + 5, channel]
                
                multiplied_neighborhood = neighborhood * kernel
                
                total = np.sum(multiplied_neighborhood)
                
                blurred_image[i, j, channel] = total     
    
    return blurred_image

image_path = "images/shirt.jpg"
image = Image.open(image_path)
image_array = np.array(image)

kernel = np.array([
    [1,  4,  7,  4, 1],
    [4, 16, 26, 16, 4],
    [7, 26, 41, 26, 7],
    [4, 16, 26, 16, 4],
    [1,  4,  7,  4, 1] 
])
kernel = kernel / 273.0

filtered_image = convolution_and_blur_filter(image_array, kernel)
filtered_image = Image.fromarray(filtered_image)
filtered_image.save("images/image_filtered.jpg")
print("Saved filtered image to images/image_filtered.jpg")