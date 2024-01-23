##########################################################################################################
# Author: Peter Karacsonyi                                                                               #
# Last updated: 2024 jan 23                                                                              #
# Input: write jpeg tiles from a wsi                                                                     #
##########################################################################################################

##############
# IO folders #
##############
wsi_folder = "/mnt/bigdata/placenta/wsitest"  # reads all wsi files in folder
out_folder = "/mnt/bigdata/placenta/tilestest"  # creates tiles in a directory with wsi name


##########
# Config #
##########
resolution = 0      # 0 is the highest wsi resolution level, for available resolutions see the index of level_downsamples (will be printed below)
coverage   = 0.35   # percentage the mask should cover the tile to avoid getting dropped
tile_size  = 500


###########
# imports #
###########
import os # os._exit(0)
import cv2
import openslide
import time
import numpy as np
from PIL import Image
from pathlib import Path
from pathml.core import Tile, types, HESlide
from pathml.preprocessing import TissueDetectionHE


wsi_paths = list(Path(wsi_folder).glob("*.tif*"))


def TissueDetectandSave(image_nparray, coverage):
    im = Tile(image_nparray, coords=(0, 0), name="quartertile", slide_type=types.HE)

    st = time.time()
    TissueDetectionHE(
        mask_name="tissue",
        threshold=19,
        min_region_size=1000,
        outer_contours_only=False,
        max_hole_size=10000,
        use_saturation=True,
    ).apply(im)
    print(f"TissueDetectionHE took {(time.time() - st) // 60:.0f}m {(time.time() - st) % 60:.0f}s")

    st = time.time()
    im.masks["tissue"][im.masks["tissue"] == 127] = 1  # convert 127s to 1s
    im.image *= np.expand_dims(im.masks["tissue"], 2)  # element wise in-place mm

    maxx = ((im.shape[0] // tile_size) * tile_size) - tile_size
    maxy = ((im.shape[1] // tile_size) * tile_size) - tile_size

    tilecounter = 0
    tile_mask_sum_max = 0
    wsiname = wsi.stem
    Path(out_folder + "/" + wsiname).mkdir(parents=True, exist_ok=True)
    coverage = coverage * tile_size * tile_size

    for tile_x_start in range(0, maxx, tile_size):
        for tile_y_start in range(0, maxy, tile_size):
            tile_x_end = tile_x_start + tile_size
            tile_y_end = tile_y_start + tile_size
            tile_mask = im.masks["tissue"][
                tile_x_start:tile_x_end, tile_y_start:tile_y_end
            ]
            tile_mask_sum = np.sum(tile_mask)
            if tile_mask_sum < coverage:
                continue
            else:
                if tile_mask_sum_max < tile_mask_sum:
                    tile_mask_sum_max = tile_mask_sum
                Image.fromarray(
                    im.image[tile_x_start:tile_x_end, tile_y_start:tile_y_end]
                ).save(
                    f"{out_folder}/{wsiname}/{wsiname}-{tile_x_start}_{tile_y_start}.jpg"
                )
                tilecounter += 1

    print(f"PIL Image save took {(time.time() - st) // 60:.0f}m {(time.time() - st) % 60:.0f}s") ; print("\r\n")

    return tilecounter
    

#############################
# function to extract tiles #
#############################
def processWSI(wsi):

    tiles_total = 0
    openslide_wsi = openslide.OpenSlide(wsi)

    # width of image e.g. 20001
    w = openslide_wsi.level_dimensions[resolution][1]
    h = openslide_wsi.level_dimensions[resolution][0]

    # need to be divisible by tile_size
    half_width = ((w // tile_size) // 2) * tile_size
    half_height = ((h // tile_size) // 2) * tile_size
    # another half 10001
    other_half_width = int(w - half_width)
    other_half_height = int(w - half_height)

    print("shape and resolution details of the input file: ")
    print("\tshape: ", openslide_wsi.dimensions)
    print("\tlevel_count: ", openslide_wsi.level_count)
    print("\tlevel_downsamples (from 0 index to n): ", openslide_wsi.level_downsamples)
    print("\tlevel_dimensions (h,w): ", openslide_wsi.level_dimensions)
    print("\r\n")

    st = time.time()
    image_pil = openslide_wsi.read_region(
        location=(0, 0),
        size=(
            half_width,
            half_height,
        ),
        level=resolution,
    )
    print(
        f"openslide read_region() {(time.time() - st) // 60:.0f}m {(time.time() - st) % 60:.0f}s"
    )

    st = time.time()
    image_nparray = cv2.cvtColor(np.asarray(image_pil), cv2.COLOR_RGBA2RGB).astype(
        np.uint8
    )
    print(f"cv2 cv2.cvtColor(np.asarray()) took {(time.time() - st) // 60:.0f}m {(time.time() - st) % 60:.0f}s")
    
    tiles_total += TissueDetectandSave(image_nparray, coverage)

    return tiles_total

    


print(f"**************************************************")
print(f"***************** TILE EXTRACTOR *****************")
print(f"**************************************************")
print("\r\n")

#######
# run #
#######

for wsi in wsi_paths:
    print(
        f"************ Processing {wsi.name} on resolution level {resolution} ************"
    )
    print("\r\n")
    start_time = time.time()
    wsi_tilecount = processWSI(wsi)
    print(
        f"************ Extracted {wsi_tilecount} tiles from {wsi.name} in {(time.time() - start_time) // 60:.0f}m {(time.time() - start_time) % 60:.0f}s ************"
    )
    print("\r\n")
