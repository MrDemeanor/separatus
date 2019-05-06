"""
    File: create_histograms.py

    Usage:  Given a set of images, create a pair of positive and negative histograms 
            that represent positive RGB values representing an object, and negative examples. 
"""

__author__ = "Brent Redmon"
__copyright__ = "Copyright 2019, Texas State University"
__credits__ = ["Brent Redmon", "Nicholas Warren"]
__license__ = "GPL"
__version__ = "1.0.1"
__maintainer__ = ["Brent Redmon", "Nicholas Warren"]
__email__ = "btr26@txstate.edu"
__status__ = "Production"

import numpy as np 
from PIL import Image
import os
from math import floor
import json

# Open config file
config = json.load(open("configs/histogram_generation_config.json"))

# Initialize positive histogram
positive_histogram = np.zeros((32, 32, 32))

# Get the directory of the sample images
sample_images_dir = config["positive_folder"]

# For each sample image in the sample images directory
for image in os.listdir("{}/{}".format(os.getcwd(), sample_images_dir)):
    
    # Read in the image
    this_dir = os.getcwd()
    im = Image.open("{}/{}/{}".format(os.getcwd(), sample_images_dir, image))
    
    # Load in the RGB values 
    pix = im.load()
    
    # Get the number of rows and columns
    rows = im.size[0]
    cols = im.size[1]

    # For each pixel in the image, use the RGB values to access the RGB'th index in the water histogram and increment it by 1
    for row in range(rows):
        for col in range(cols):
            pix_values = pix[row, col]

            # Normalize all RGB values to fit in the 32x32x32 histogram
            red = floor(pix_values[0] / 8)
            green = floor(pix_values[1] / 8)
            blue = floor(pix_values[2] / 8)

            positive_histogram[red, green, blue] = positive_histogram[red, green, blue] + 1

# Initialize positive histogram
negative_histogram = np.zeros((32, 32, 32))

# Get the directory of the sample images
sample_images_dir = config["negative_folder"]

# For each sample image in the sample images directory
for image in os.listdir("{}/{}".format(os.getcwd(), sample_images_dir)):
    
    # Read in the image
    this_dir = os.getcwd()
    im = Image.open("{}/{}/{}".format(os.getcwd(), sample_images_dir, image))
    
    # Load in the RGB values 
    pix = im.load()
    
    # Get the number of rows and columns
    rows = im.size[0]
    cols = im.size[1]

    # For each pixel in the image, use the RGB values to access the RGB'th index in the water histogram and increment it by 1
    for row in range(rows):
        for col in range(cols):
            pix_values = pix[row, col]

            # Normalize all RGB values to fit in the 32x32x32 histogram
            red = floor(pix_values[0] / 8)
            green = floor(pix_values[1] / 8)
            blue = floor(pix_values[2] / 8)

            negative_histogram[red, green, blue] = negative_histogram[red, green, blue] + 1

# Save positive and negative histograms
try:
    np.save("histograms/{}_positive_histogram.npy".format(config["object_of_interest"]), positive_histogram)
    np.save("histograms/{}_negative_histogram.npy".format(config["object_of_interest"]), negative_histogram)

    print("Successfully made histograms!")
except:
    print("Could not create histograms")