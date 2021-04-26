import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import cv2
import os
from matplotlib import image
from scipy import signal
from filters import magnitude, create_LoG_filter
import sys

def load_images(d, plot=False, get_mag=False, get_LoG=False):
    
    # Initialize dictionnaries
    imgs, mags, LoG = {}, {}, {}
    
    # Loop through the given directory
    for filename in sorted(os.listdir(d)):
        
        # If the file is an image
        if filename.endswith(".jpg") or filename.endswith(".png"):
            
            # Read the image using opencv and extract the image number
            img = cv2.imread(os.path.join(d, filename))
            num = filename.split(".")[0]
            imgs[num] = img
            
            # Compute magnitude if specified
            if get_mag:
                mags[num] = magnitude(img)
                
            # Compute Laplacian of Gaussians if sepcified
            if get_LoG:
                LoG[num] = signal.convolve2d(img[:, :, 0], create_LoG_filter(1),
                                             boundary='symm', mode='same')
            
            # Plot the image if specified
            if plot:
                plt.imshow(img[:, :, ::-1])
                plt.title(num)
                plt.show()

    return imgs, mags, LoG

if __name__ == "__main__":

    # "../data/train/image_left"
    path = sys.argv[1]
    left_imgs, mags, LoGs = load_images(path, get_mag=True, get_LoG=True)
    
    # Looking at an image
    plt.imshow(left_imgs["um_000019"][:, :, ::-1])
    plt.show()
