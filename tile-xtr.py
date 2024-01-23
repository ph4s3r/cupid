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
wsi_resolution_level = 1 # 0 is the highest resolution, for available resolutions please use the index of level_downsamples (will be printed below)
tile_size = 500


###########
# imports #
###########
import time
import openslide
import numpy as np
from pathml.utils import pil_to_rgb
from pathlib import Path
from PIL import Image as pil_image
from pathml.core import Tile, types
from pathml.preprocessing import TissueDetectionHE


wsi_paths = list(Path(wsi_folder).glob("*.tif*"))

#############################
# function to extract tiles #
#############################


def extract_tiles(wsi, resolution_level, tile_size):
    
    openslide_wsi = openslide.OpenSlide(wsi)

    print("shape and resolution details of the input file: ")
    print("\tshape: ", openslide_wsi.dimensions)
    print("\tlevel_count: ", openslide_wsi.level_count)
    print("\tlevel_downsamples (from 0 index to n): ", openslide_wsi.level_downsamples)
    print("\tlevel_dimensions (h,w): ", openslide_wsi.level_dimensions)

    start_time = time.time()
    openslide_region = openslide_wsi.read_region(
        location=(0, 0), 
        size=(openslide_wsi.level_dimensions[resolution_level][1], openslide_wsi.level_dimensions[resolution_level][0]),
        level=resolution_level
        )
    print(f"openslide_region {(time.time() - start_time) // 60:.0f}m {(time.time() - start_time) % 60:.2f}s")
    
    start_time = time.time()
    region = pil_to_rgb(openslide_region)
    print(f"pil_to_rgb took {(time.time() - start_time) // 60:.0f}m {(time.time() - start_time) % 60:.2f}s")

    start_time = time.time()
    im = Tile(region, coords=(0, 0), name="testregion", slide_type=types.HE)
    print(f"pathML Tile() took {(time.time() - start_time) // 60:.0f}m {(time.time() - start_time) % 60:.2f}s")

    start_time = time.time()
    TissueDetectionHE(
        mask_name = "tissue", 
        threshold = 19,
        min_region_size=1000,
        outer_contours_only=False,
        max_hole_size = 10000,
        use_saturation=True
    ).apply(im)
    print(f"TissueDetectionHE took {(time.time() - start_time) // 60:.0f}m {(time.time() - start_time) % 60:.2f}s")

    
    start_time = time.time()
    im.masks["tissue"][im.masks["tissue"] == 127] = 1 # convert 127s to 1s
    im.image *= np.expand_dims(im.masks["tissue"], 2) # element wise in-place mm
    
    maxx = ((region.shape[0] // tile_size) * tile_size) - tile_size
    maxy = ((region.shape[1] // tile_size) * tile_size) - tile_size

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
                pil_image.fromarray(im.image[tile_x_start:tile_x_end, tile_y_start:tile_y_end]).save(f'{out_folder}/{wsiname}/{wsiname}-{tile_x_start}_{tile_y_start}.jpg')
                tilecounter += 1
    
    print(f"pil_image.save took {(time.time() - start_time) // 60:.0f}m {(time.time() - start_time) % 60:.2f}s")
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
    wsi_tilecount = extract_tiles(wsi, resolution_level=wsi_resolution_level, tile_size=tile_size)
    print(f"************ Extracted {wsi_tilecount} tiles from {wsi.name} in {(time.time() - start_time) // 60:.0f}m {(time.time() - start_time) % 60:.2f}s ************")
    print("\r\n")