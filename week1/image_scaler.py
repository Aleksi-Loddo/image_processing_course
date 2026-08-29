from PIL import Image
import numpy as np

def scale_nearest_neighbor(image_array, scale_factor):
    old_height, old_width, channels = image_array.shape
    new_width = int(old_width * scale_factor)
    new_height = int(old_height * scale_factor)

    width_scale = new_width / old_width
    height_scale = new_height / old_height

    # Create an empty array for the new image
    scaled_pixels = np.zeros((new_height, new_width, channels), dtype=np.uint8)

    for y in range(new_height):
        for x in range(new_width):
            # Find corresponding original pixel
            old_x = min(int(x / width_scale), old_width - 1)
            old_y = min(int(y / height_scale), old_height - 1)
            
            scaled_pixels[y, x] = image_array[old_y, old_x]

    return scaled_pixels

# Load the image
picture_path = "pictures/shirt.jpg"
picture = Image.open(picture_path)
pixels = np.array(picture)

# Scale to 0.17x
scaled_pixels = scale_nearest_neighbor(pixels, 0.17)

# Save the scaled image
scaled_image = Image.fromarray(scaled_pixels)
scaled_image.save("pictures/shirt_scaled_0.17.jpg")
print("Saved scaled image to pictures/shirt_scaled_0.17.jpg")
