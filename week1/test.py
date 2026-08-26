from PIL import Image
import numpy as np
import matplotlib


#print(np.__version__)
#print(Image.__version__)
#print(matplotlib.__version__)

picture1 = "pictures/linnanmaa.jpg"

picture =Image.open(picture1)
pixels=np.array(picture)


#prints any pixels color value

#ask user for a pixel to select
#pixel_x = int(input("Enter x coordinate: "))
#pixel_y = int(input("Enter y coordinate: "))
#print color value of any pixel in a image
#print(pixels[pixel_x,pixel_y])



#print pixel value of upper left corner of image
#print(pixels[0,0])

width, height = picture.size

print(f"width:{width},height:{height}")

num_rows, num_cols,color_channels = pixels.shape

print(f"the image is {num_rows} pixel height and {num_cols} pixel width ")