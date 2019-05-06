"""
    File: maskify_dataset.py

    Usage:  Given a dataset that contains bounding box annotations and positive and negative
            histograms, we can derive the segmentation, or shape, of the object of interest
            and record those segmentation values in a COCO formatted JSON data structure
"""

__author__ = "Brent Redmon"
__copyright__ = "Copyright 2019, Texas State University"
__credits__ = ["Brent Redmon", "Nicholas Warren"]
__license__ = "GPL"
__version__ = "1.0.1"
__maintainer__ = ["Brent Redmon", "Nicholas Warren"]
__email__ = "btr26@txstate.edu"
__status__ = "Production"

from PIL import Image
from sklearn.cluster import MeanShift, estimate_bandwidth
from skimage.filters import gaussian
from skimage.measure import label
import matplotlib.pyplot as plt
import numpy as np
from math import floor
from scipy.ndimage.morphology import binary_dilation, binary_fill_holes
import json
from pycocotools import mask
from skimage import measure
import os
from progress.bar import IncrementalBar

def print_report(reports, config):
    print("Printing reports for each set processed:\n\n")
    print("Dataset: ".format(config["dataset"]))

    for subset_report in reports:
        print("Subset: ".format(subset_report["name"]))
        print("Annotation file save directory: ".format(subset_report["annotation_directory"]))
        print("Processed {} images out of {}\n".format(subset_report["processed_images"], subset_report["total_images"]))

def maskify(    im, 
                crop_box, 
                threshold, 
                positive_histogram, 
                negative_histogram, 
                threshold
):
    """ For each annotation, create a COCO formated segmentation

        Arguments:
            - im:                   The input image
            - crop_box:             Tuple of coordinates isolating object of interest
            - threshold:            Percentage value representing liklihood of a pixel belonging
                                    to the positive histogram class
            - positive_histogram:   Histogram representing pixels pertaining to an object
            - negative_histogram:   Histogram representing non-example pixels pertaining to 
                                    an object 

        Return: Array of COCO styled segmentation annotations

    """
    # Get the size of the image
    original_rows, original_cols = im.size

    # Crop the image around the bounding box
    im = im.crop(crop_box)

    # Load pixel RGB data
    pix = im.load()

    # Get row and cols of cropped image
    cols, rows = im.size

    # Convert cropped image to numpy array
    im = np.array(im)

    # Get the height and width of the cropped image
    rows = np.shape(im)[0]
    cols = np.shape(im)[1]

    # Get histogram bins
    histogram_bins = np.shape(positive_histogram)[0]

    # Get the factor based on the histogram bins. Used to index into to the histogram. 
    factor = 256 / histogram_bins

    # Declare a results numpy array that contains only zeros
    result = np.zeros((rows, cols))

    # Determine the probability of water given RGB and histograms representing water and non water
    for row in range(rows):
        for col in range(cols):

            # Get each RGB value
            red = float(pix[col, row][0])
            green = float(pix[col, row][1])
            blue = float(pix[col, row][2])
            
            # Get the index into histograms based on RGB value and histogram factor size (declared above)
            red_index = floor(red / factor)
            green_index = floor(green / factor)
            blue_index = floor(blue / factor)
            
            # Get positive and negative values from histograms
            positive = positive_histogram[red_index, green_index, blue_index]
            negative_value = negative_histogram[red_index, green_index, blue_index]
            
            total = positive + negative
            
            if total is not 0:
                result[row, col] = water_value / total

    # Set threshold equal to the median value of the resulting numpy array if 
    threshold = np.median(result) if threshold is 'auto' else threshold

    # The intuition here is that if our threshold is equal to the median value of the resulting
    # array, then there will be a largest connected component. Any other value, and we're risking
    # the possibility of no largest connected component existing, which is a potential error that we 
    # have to account for. 
    if threshold != np.median(result):
        result_backup = np.copy(result)

    # Parse values of result given threshold
    for row in range(rows):
        for col in range(cols):
            if result[row, col] < threshold:
                result[row, col] = 1
            else:
                result[row, col] = 0
    
    # Retry if all values in result are 0 (ie - no largest connected component)
    if np.sum(result) == 0:
        result = result_backup

        for row in range(rows):
            for col in range(cols):
                if result[row, col] < np.median(result):
                    result[row, col] = 1
                else:
                    result[row, col] = 0

    
    # Get the largest connected component
    labels = label(result)
    assert( labels.max() != 0 ) # assume at least 1 CC
    largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1
    
    # Fill holes in the boat
    largestCC = binary_fill_holes(largestCC)

    # Dialate to expand the mask
    largestCC = binary_dilation(largestCC, iterations=4)
    plt.imshow(largestCC)

    # Create numpy zeros array the same size as the original image before cropping
    image_with_mask = np.zeros((original_cols, original_rows))

    # Overlay binary mask onto zeros array
    image_with_mask[crop_box[1]:crop_box[1] + rows, crop_box[0]:crop_box[0] + cols] = largestCC

    """ Convert the binary mask to COCO JSON format. Code referenced from:
            - https://github.com/cocodataset/cocoapi/issues/131#issuecomment-371250565
    """
    image_with_mask = np.array(image_with_mask, dtype=np.uint8)
    fortran_ground_truth_binary_mask = np.asfortranarray(image_with_mask)
    encoded_ground_truth = mask.encode(fortran_ground_truth_binary_mask)
    ground_truth_area = mask.area(encoded_ground_truth)
    ground_truth_bounding_box = mask.toBbox(encoded_ground_truth)
    contours = measure.find_contours(image_with_mask, 0.5)

    segmentations = []

    for contour in contours:
        contour = np.flip(contour, axis=1)
        segmentation = contour.ravel().tolist()
        segmentations.append(segmentation)
    
    return segmentations


if __name__ == "__main__":

    # Report data structure to display at the end
    report = list()

    # Load the config file
    maskify_config = json.load(open("configs/maskify_config.json"))
    dataset = maskify_config["dataset"]

    """ A subset in this case is typically the following:
            - Test set
            - Train set
            - Validation set
    """
    for subset in maskify_config["subsets"]:

        subset_report = {
            "name": subset
            "annotation_directory": "", 
            "processed_images": 0, 
            "total_images": ""
        }

        # Assign metadata to variables
        coco_json = json.load(open("datasets/{}/train/annotations.json".format(dataset)))

        # What do I need to pass into the function?
        #   - Image file name
        #   - Bounding box (original segmentation)
        #   - Threshold (usually 0.48)
        #   - Positive and negative histograms 

        # Initialize progress bar
        bar = IncrementalBar("Processing {} images".format(subset), max = len(coco_json["annotations"]))
        
        # Log the number of images being processed
        subset_report["total_images"] = len(coco_json["annotations"])
        print("\n\n")

        for annotation in coco_json["annotations"]:

            # Find the picture related to the image
            for image in coco_json["images"]:
                if annotation["image_id"] == image["id"]:
                    this_image = image["file_name"]
                    image_filename = image["file_name"]
                    break
            
            # Open image
            this_image = Image.open("datasets/{}/{}/images/{}".format(dataset, subset, this_image))
            
            # Get crop boundary
            crop_box = (annotation["segmentation"][0][0], annotation["segmentation"][0][1], annotation["segmentation"][0][4], annotation["segmentation"][0][5])

            # Set threshold
            threshold = maskify_config["threshold"]

            # Load positive and negative histograms
            positive_histogram = np.load(maskify_config["positive_histogram"])
            negative_histogram = np.load(maskify_config["negative_histogram"])

            try:
                annotation["segmentation"] = maskify(this_image, crop_box, threshold, RGB_Water_Histogram, RGB_Non_Water_Histogram)
                subset_report["processed_images"] = subset_report["processed_images"] + 1
            except Exception as e:
                print("Could not process training image: {} -- {}".format(image_filename, e))
            
            bar.next()
        bar.finish()

        print("Saving {} annotations...".format(subset))

        # Set the save directory too the "annotation_directory" index of subset_report
        subset_report["annotation_directory"] = "datasets/{}/{}/annotations_maskified.json".format(dataset, subset)

        with open("datasets/{}/{}/annotations_maskified.json".format(dataset, subset), "w") as outfile:
            json.dump(coco_json, outfile)
        
        # Append the report to the master report list
        report.append(subset_report)
    
    print_report(report, maskify_config)