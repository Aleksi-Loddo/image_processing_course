from PIL import Image
import numpy as np

def convert_to_grayscale_average(image_array):
    """
    Converts an RGB image array to grayscale by averaging the R, G, and B channels.
    """
    # 1. Convert to float to prevent integer overflow when adding R+G+B
    img_float = image_array.astype(float)
    
    # 2. Calculate the average of the 3 channels (R, G, B) for every pixel
    # axis=2 means we are averaging across the "color channel" depth of the array
    average_values = img_float.mean(axis=2)
    
    # 3. Create a new image where R, G, and B are all set to this average value
    # np.stack duplicates the average values 3 times so it has an R, G, and B channel
    gray_pixels = np.stack([average_values, average_values, average_values], axis=-1)
    
    # 4. Convert back to 8-bit integers (0-255) which is what images require
    return gray_pixels.astype(np.uint8)

if __name__ == "__main__":
    # Load the image
    picture_path = "pictures/linnanmaa.jpg"
    picture = Image.open(picture_path)
    
    # Convert to RGB just in case the image has a transparency channel (RGBA)
    picture = picture.convert('RGB')
    pixels = np.array(picture)
    
    # Convert to grayscale
    gray_pixels = convert_to_grayscale_average(pixels)
    
    # Save the resulting image
    gray_image = Image.fromarray(gray_pixels)
    gray_image.save("pictures/linnanmaa_grayscale.jpg")
    print("Saved grayscale image to pictures/linnanmaa_grayscale.jpg")
