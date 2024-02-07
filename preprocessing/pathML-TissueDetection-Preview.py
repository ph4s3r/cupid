##########################################################################################################
# Author: Peter Karacsonyi                                                                               #
# Last updated: 2024 jan 4                                                                               #
# This workbook has been made to determine the perfect PathML transformation configurations ->           #
# It reads wsis and plots the image as well as all mask transformations to simulate final mask           #
# Input: wsi image files, tested with OpenSlide .tif                                                     #
# Transformations: BinaryThreshold, MorphClose, MorphOpen, ForegroundDetection                           #
# Read more on transformations: https://pathml.readthedocs.io/en/latest/examples/link_gallery.html       #
# Output: matplotlib plots dislpayed, nothing saved                                                      #
##########################################################################################################

############################################
# mask data structure:                     #
# two values in a numpy.ndarray: 0 and 127 #
# (dtype: uint8)                           #
# 127 is obviously the mask                #
############################################



#########
# usage #
#########

# set the path of wsi files
wsi_folder = "G:\\placenta\\wsi"

#######################################
# configure transformation parameters #
#######################################

# binary transformation
bt_otsu = True
bt_threshold = 225 # ignored if otsu=True
bt_inverse = False

# morphOpen
okernel = 10
oiter = 5

# morphCLose
ckernel = 40
citer = 5

# foreground detection
fgd_min_region_size = 5000
fgd_max_hole_size = 5500
fgd_outer_contours_only = False # always False except on purpose!

################################################################
# run the program, close plot window to get the next displayed #
################################################################

# imports
import os
if os.name == "nt":
    import helpers.openslideimport  # on windows, openslide needs to be installed manually, check local openslideimport.py

import random
import numpy as np
from pathlib import Path
from pathml.preprocessing import (
    TissueDetectionHE
)
import matplotlib.pyplot as plt
from pathml.utils import plot_mask
from pathml.core import HESlide, Tile, types

fontsize = 8

# debugging random empty spaces covered by mask
coords = [(0, 1000), (0, 3000)]
rect_size = (500, 500) # width, height of the rectangles

# 0 is the highest resolution, need to use the index of level_downsamples
def transform_and_plot(wsi, resolution_level):
    pml_wsi = HESlide(
        wsi,
        backend="openslide",
        slide_type=types.HE,
    )

    print("some details of ", pml_wsi.name)
    print("shape: ", pml_wsi.shape)
    print("level_count: ", pml_wsi.slide.level_count)
    print("level_downsamples (from 0 index to n): ", pml_wsi.slide.slide.level_downsamples)
    print("level_dimensions (h,w): ", pml_wsi.slide.slide.level_dimensions)
    try:
        print("color profile: ", pml_wsi.slide.slide.color_profile.profile.profile_description)
    except:
        print("no color profile information found")
    print("rgb?: ", pml_wsi.slide_type.rgb)
    print("masks: ", len(pml_wsi.masks))
    print("tiles: ", len(pml_wsi.tiles))
    print("labels: ", pml_wsi.labels)

    try:
        # dimensions are transposed!!! needed to invert.. (pml_wsi.slide.slide.level_dimensions[0][1], pml_wsi.slide.slide.level_dimensions[0][0]))
        region = pml_wsi.slide.extract_region(
            location=(0, 0),
            size=(pml_wsi.slide.slide.level_dimensions[resolution_level][1], pml_wsi.slide.slide.level_dimensions[resolution_level][0]),
            level=resolution_level
            )
        print("Working with pyramid resolution level", resolution_level, "shape:", region.shape[0:1])
    except Exception as e:
        print("Resolution level not found, using original size. Error: ", e)
        # dimensions are transposed!!! needed to invert.. (pml_wsi.slide.slide.level_dimensions[0][1], pml_wsi.slide.slide.level_dimensions[0][0]))
        region = pml_wsi.slide.extract_region(
            location=(0, 0),
            size=(pml_wsi.slide.slide.level_dimensions[0][1], pml_wsi.slide.slide.level_dimensions[0][0])
            )

    # some tiff images (echino) had half of the image array blank so removing improves plotting 
    # smaller_dim = min(region.shape[0:2])
    # smallregion = region[0:smaller_dim, 0:smaller_dim]
    # print("removed unnecessary part (right half of img is displayed blank.. not sure why), new region shape:", region.shape)

    random_x = np.random.randint(0, high=region.shape[0], size=10, dtype=int)
    random_y = np.random.randint(0, high=region.shape[1], size=10, dtype=int)

    print("region dtype:", type(region))

    print("some random samples of the image: ")
    for i in range(10):
        print(region[random_x[i], random_y[i]])

    def customTile():
        return Tile(region, coords=(0, 0), name="testregion", slide_type=types.HE)

    tile = customTile()

    ###########################################################

    # do TissueDetectionHE
    TissueDetectionHE(
        mask_name = "tissue", 
        threshold = 19,
        min_region_size=1000,
        outer_contours_only=False,
        max_hole_size = 10000,
        use_saturation=True
    ).apply(tile)

    _, ax = plt.subplots(figsize=(8, 8))
    plot_mask(im = tile.image, mask_in=tile.masks["tissue"], ax = ax)

    for x, y in coords:
            rect = plt.Rectangle((y, x), rect_size[1], rect_size[0], linewidth=1, edgecolor='black', facecolor='none')
            ax.add_patch(rect)

    plt.title("Overlay with suspicious tileboxes", fontsize = fontsize)
    plt.axis('off')
    plt.show()

wsi_paths = list(Path(wsi_folder).glob("*.tif*")) ; random.shuffle(wsi_paths)

for wsi in wsi_paths:
    transform_and_plot(wsi, resolution_level=2)

pass
