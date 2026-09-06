from PIL import Image
import numpy as np
import math

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

def bilinear_interpolation(original_pixels,new_height, new_width):
    #getting original pixel dimensions
    old_height, old_width, channels = original_pixels.shape
    #array go desired shape 
    resized_image = np.zeros((new_height, new_width, channels))
    #calculate width and height
    width_scale = (old_width)/(new_width) if new_width > 0 else 0
    height_scale = (old_height)/(new_height) if new_height > 0 else 0
    for i  in range(new_height):
        #maping cordinates back to original image
        for j in range(new_width):
            x = i * height_scale
            y = j * width_scale
            
            # calculate the coordinate values for 4 surrounding pixels
            x_floor = int(x)
            y_floor = int(y)
            x_ceil = min(old_height - 1, math.ceil(x))
            y_ceil = min(old_width - 1, math.ceil(y))
            
            # get the neighbouring pixel values
            p1 = original_pixels[x_floor, y_floor]
            p2 = original_pixels[x_floor, y_ceil]
            p3 = original_pixels[x_ceil, y_floor]
            p4 = original_pixels[x_ceil, y_ceil]
            
            # Estimate the pixel value q using pixel values of neighbours
            # q1 = p1 * (x_ceil - x) + p2 * (x - x_floor)
            # q2 = p3 * (x_ceil - x) + p4 * (x - x_floor)
            # q = q1 * (y_ceil - y) + q2 * (y - y_floor)
            if (x_ceil == x_floor) and (y_ceil == y_floor):
                q = original_pixels[int(x), int(y), :]
            elif (x_ceil == x_floor):
                q1 = original_pixels[int(x), int(y_floor), :]
                q2 = original_pixels[int(x), int(y_ceil), :]
                q = q1 * (y_ceil - y) + q2 * (y - y_floor)
            elif (y_ceil == y_floor):
                q1 = original_pixels[int(x_floor), int(y), :]
                q2 = original_pixels[int(x_ceil), int(y), :]
                q = (q1 * (x_ceil - x)) + (q2 * (x - x_floor))
            else:
                v1 = original_pixels[x_floor, y_floor, :]
                v2 = original_pixels[x_ceil, y_floor, :]
                v3 = original_pixels[x_floor, y_ceil, :]
                v4 = original_pixels[x_ceil, y_ceil, :]

                q1 = v1 * (x_ceil - x) + v2 * (x - x_floor)
                q2 = v3 * (x_ceil - x) + v4 * (x - x_floor)
                q = q1 * (y_ceil - y) + q2 * (y - y_floor) 
            resized_image[i, j] = q
    return resized_image
    
# Load the image
picture_path = "pictures/shirt_small.jpg"
picture = Image.open(picture_path)
pixels = np.array(picture)

# Scale to 4x
scaled_pixels = scale_nearest_neighbor(pixels, 4)
scaled_pixels_bilinear = bilinear_interpolation(pixels, pixels.shape[0]*4, pixels.shape[1]*4)

# Save the scaled image
scaled_image = Image.fromarray(scaled_pixels)
scaled_image_bilinear = Image.fromarray(scaled_pixels_bilinear.astype(np.uint8))
scaled_image.save("pictures/shirt_small_scaled_4.jpg")
scaled_image_bilinear.save("pictures/shirt_small_scaled_4_bilinear.jpg")
print("Saved scaled image to pictures/shirt_small_scaled_4.jpg")
