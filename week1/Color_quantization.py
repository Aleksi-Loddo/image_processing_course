from PIL import Image
import numpy as np

def reduce_bit_depth(image_array, bits):
    """
    Reduces the number of bits per color channel by zeroing out the lowest bits.
    """
    # Calculate how many bits we need to throw away
    shift = 8 - bits
    
    # Right-shift to drop the least significant bits, 
    # then left-shift to scale the values back into the 0-255 range.
    reduced_array = (image_array >> shift) << shift
    
    return reduced_array

if __name__ == "__main__":
    # Load the image
    picture_path = "pictures/linnanmaa.jpg"
    picture = Image.open(picture_path).convert('RGB')
    pixels = np.array(picture)
    
    # Generate and save for 4, 3, 2, and 1 bits
    for bits in [4, 3, 2, 1]:
        reduced_pixels = reduce_bit_depth(pixels, bits)
        reduced_image = Image.fromarray(reduced_pixels)
        
        output_name = f"pictures/linnamaa_{bits}bits.png"
        reduced_image.save(output_name, format="PNG")
        print(f"Saved {output_name}")
