import matplotlib.pyplot as plt
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

if __name__ == "__main__":

 np.random.seed(42) 
 original_img = np.random.randint(0, 256, size=(5, 5, 3), dtype=np.uint8)

 print("--- ORIGINAL ---")
 print(original_img)

 expanded_img = expand_edges_2d(original_img, pad_size=2)
    
 print("\n--- EXPANDED ---")
 print(expanded_img)

#Visually show the Original Image
plt.subplot(1, 2, 1)
plt.title("Original 5x5")
plt.imshow(original_img)
# Visually show the Expanded Image
plt.subplot(1, 2, 2)
plt.title("Expanded 9x9")
plt.imshow(expanded_img)
# Display the window
plt.show()
