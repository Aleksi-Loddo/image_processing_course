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

image_path = "images/17.png"
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

blurred_array = convolution_and_blur_filter(image_array, kernel)

original_float = image_array.astype(float)
blurred_float = blurred_array.astype(float)

unsharp_mask = original_float - blurred_float

multiplier = 1.0
sharpened_float = original_float + (multiplier * unsharp_mask)

sharpened_array = np.clip(sharpened_float, 0, 255).astype(np.uint8)

mask_display = np.clip(np.abs(unsharp_mask) * 4, 0, 255).astype(np.uint8)

mask_image = Image.fromarray(mask_display)
mask_image.save("images/17_difference_mask.jpg")
print("Saved difference mask to images/17_difference_mask.jpg")

sharpened_image = Image.fromarray(sharpened_array)
sharpened_image.save("images/17_sharpened.jpg")
print("Saved sharpened image to images/17_sharpened.jpg")