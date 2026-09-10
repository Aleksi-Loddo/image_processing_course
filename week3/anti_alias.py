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
    for c in range(channel):
        for i in range(height):
            for j in range(width):
                neighborhood = padded_image[i : i + 5, j : j + 5, c]
                multiplied_neighborhood = neighborhood * kernel
                total = np.sum(multiplied_neighborhood)
                blurred_image[i, j, c] = total     
    return blurred_image

def scale_nearest_neighbor(image_array, scale_factor):
    old_height, old_width, channels = image_array.shape
    new_width = int(old_width * scale_factor)
    new_height = int(old_height * scale_factor)
    width_scale = new_width / old_width
    height_scale = new_height / old_height
    scaled_pixels = np.zeros((new_height, new_width, channels), dtype=np.uint8)
    for y in range(new_height):
        for x in range(new_width):
            old_x = min(int(x / width_scale), old_width - 1)
            old_y = min(int(y / height_scale), old_height - 1)
            scaled_pixels[y, x] = image_array[old_y, old_x]
    return scaled_pixels

if __name__ == "__main__":
    image_path = "images/shirt.jpg"
    print(f"Loading {image_path}...")
    image = Image.open(image_path)
    image_array = np.array(image)

    kernel = np.array([
        [0.00390625, 0.015625, 0.0234375, 0.015625, 0.00390625],
        [0.015625, 0.0625, 0.09375, 0.0625, 0.015625],
        [0.0234375, 0.09375, 0.140625, 0.09375, 0.0234375],
        [0.015625, 0.0625, 0.09375, 0.0625, 0.015625],
        [0.00390625, 0.015625, 0.0234375, 0.015625, 0.00390625]
    ])

    print("Applying Gaussian blur (5x5)...")
    blurred_image_array = convolution_and_blur_filter(image_array, kernel)
    
    print("Scaling down to 0.17x factor...")
    scaled_array = scale_nearest_neighbor(blurred_image_array, 0.17)

    scaled_image = Image.fromarray(scaled_array)
    out_path = "images/shirt_anti_aliased_0.17.jpg"
    scaled_image.save(out_path)
    print(f"Saved anti-aliased and scaled image to {out_path}")
