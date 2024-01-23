##########################################################################################################
# Author: Peter Karacsonyi                                                                               #
# Last updated: 2024 jan 23                                                                              #
# Input: Creates tiles from a wsi                                                                        #
##########################################################################################################

##############
# IO folders #
##############
wsi_folder = "/mnt/bigdata/placenta/wsitest" # reads all wsi files in folder
out_folder = "/mnt/bigdata/placenta/tajlz"   # creates tiles in a directory with wsi name


##########
# Config #
##########
wsi_resolution_level = 2 # 0 is the highest resolution, for available resolutions please use the index of level_downsamples (will be printed below)
tile_size = 500


###########
# imports #
###########
import cv2
import time
import openslide
import numpy as np
from PIL import Image
from pathlib import Path
from pathml.core import Tile, types
from pathml.preprocessing import TissueDetectionHE


wsi_paths = list(Path(wsi_folder).glob("*.tif*"))

#############################
# function to extract tiles #
#############################


def extract_tiles(wsi, lvl, tile_size):
    
    openslide_wsi = openslide.OpenSlide(wsi)

    print("shape and resolution details of the input file: ")
    print("\tshape: ", openslide_wsi.dimensions)
    print("\tlevel_count: ", openslide_wsi.level_count)
    print("\tlevel_downsamples (from 0 index to n): ", openslide_wsi.level_downsamples)
    print("\tlevel_dimensions (h,w): ", openslide_wsi.level_dimensions)
    print("\r\n")


    st = time.time()
    image_pil = openslide_wsi.read_region(
        location=(0, 0), 
        size=(openslide_wsi.level_dimensions[lvl][1], openslide_wsi.level_dimensions[lvl][0]),
        level=lvl
        )
    print(f"openslide read_region() {(time.time() - st) // 60:.0f}m {(time.time() - st) % 60:.2f}s")


    st = time.time()
    image_nparray = cv2.cvtColor(
        np.asarray(image_pil), cv2.COLOR_RGBA2RGB
        ).astype(np.uint8)
    print(f"cv2 cv2.cvtColor(np.asarray()) took {(time.time() - st) // 60:.0f}m {(time.time() - st) % 60:.2f}s")


    im = Tile(image_nparray, coords=(0, 0), name="testregion", slide_type=types.HE)


    st = time.time()
    TissueDetectionHE(
        mask_name = "tissue", 
        threshold = 19,
        min_region_size=1000,
        outer_contours_only=False,
        max_hole_size = 10000,
        use_saturation=True
    ).apply(im)
    print(f"TissueDetectionHE took {(time.time() - st) // 60:.0f}m {(time.time() - st) % 60:.2f}s")

    
    st = time.time()
    im.masks["tissue"][im.masks["tissue"] == 127] = 1 # convert 127s to 1s
    im.image *= np.expand_dims(im.masks["tissue"], 2) # element wise in-place mm
    
    maxx = ((im.shape[0] // tile_size) * tile_size) - tile_size
    maxy = ((im.shape[1] // tile_size) * tile_size) - tile_size

    tilecounter = 0; tile_mask_sum_max = 0
    wsiname = openslide_wsi._filename.stem
    Path(out_folder + "/" + wsiname).mkdir(parents=True, exist_ok=True)
    coverage = 0.35 * tile_size * tile_size
    for tile_x_start in range(0,maxx,tile_size):
        for tile_y_start in range(0,maxy,tile_size):
            tile_x_end = tile_x_start+tile_size
            tile_y_end = tile_y_start+tile_size
            tile_mask = im.masks["tissue"][tile_x_start:tile_x_end, tile_y_start:tile_y_end]
            tile_mask_sum = np.sum(tile_mask)
            if tile_mask_sum < coverage:
                continue
            else:
                if tile_mask_sum_max < tile_mask_sum:
                    tile_mask_sum_max = tile_mask_sum
                Image.fromarray(
                    im.image[tile_x_start:tile_x_end, tile_y_start:tile_y_end]
                    ).save(
                        f'{out_folder}/{wsiname}/{wsiname}-{tile_x_start}_{tile_y_start}.jpg'
                        )
                tilecounter += 1


    print(f"PIL Image save took {(time.time() - st) // 60:.0f}m {(time.time() - st) % 60:.2f}s")
    print("\r\n")
    openslide_wsi.close()
    return tilecounter

print(f"**************************************************") 
print(f"***************** TILE EXTRACTOR *****************") 
print(f"**************************************************")
print("\r\n")
for wsi in wsi_paths:
    print(f"************ Processing {wsi.name} on resolution level {wsi_resolution_level} ************") 
    print("\r\n")
    start_time = time.time()
    wsi_tilecount = extract_tiles(wsi, lvl=wsi_resolution_level, tile_size=tile_size)
    print(f"************ Extracted {wsi_tilecount} tiles from {wsi.name} in {(time.time() - start_time) // 60:.0f}m {(time.time() - start_time) % 60:.2f}s ************")
    print("\r\n")