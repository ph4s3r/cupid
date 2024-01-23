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
resolution = 0      # 0 is the highest wsi resolution level, for available lvls see the index of level_downsamples (will be printed below)
coverage   = 0.35   # percentage the mask should cover the tile to avoid getting dropped
tile_size  = 500


###########
# imports #
###########
import os # os._exit(0)
import time
import numpy as np
from PIL import Image
from pathlib import Path
from pathml.core import Tile, types, HESlide
from pathml.preprocessing import TissueDetectionHE


wsi_paths = list(Path(wsi_folder).glob("*.tif*"))

#############################
# function to extract tiles #
#############################
def extract_tiles(wsi, lvl, tile_size, threshold):

    openslide_wsi = HESlide(
        wsi,
        backend="openslide",
        slide_type=types.HE,
    )

    print("shape and resolution details of the input file: ")
    print("\tlevel 0 shape: ", openslide_wsi.shape)
    print("\tlevel_count: ", openslide_wsi.slide.slide.level_count)
    print("\tlevel_downsamples (from 0 index to n): ", openslide_wsi.slide.slide.level_downsamples)
    print("\tlevel_dimensions (h,w): ", openslide_wsi.slide.slide.level_dimensions)
    print("\r\n")

    # width of image e.g. 20001
    w = openslide_wsi.slide.slide.level_dimensions[lvl][1]
    h = openslide_wsi.slide.slide.level_dimensions[lvl][0]

    # half width e.g. 10000
    x1 = int(w // 2)
    y1 = int(h // 2)
    # another half 10001
    x2 = int(w - x1)
    y2 = int(w - y1)

    st = time.time()
    # only the left half 
    region1 = openslide_wsi.slide.extract_region(
        location=(0, 0),
        size=(x1, y1),
        level=lvl
        )
    print(f"pathml extract_region() 1 {(time.time() - st) // 60:.0f}m {(time.time() - st) % 60:.0f}s")

    # st = time.time()
    # region2 = openslide_wsi.slide.extract_region(
    #     location=(x1+1, 0),
    #     size=(x2, openslide_wsi.slide.slide.level_dimensions[lvl][0]),
    #     level=lvl
    #     )
    # print(f"pathml extract_region() 2 {(time.time() - st) // 60:.0f}m {(time.time() - st) % 60:.0f}s")

    im1 = Tile(region1, coords=(0, 0), name="testregion", slide_type=types.HE)
    # im2 = Tile(region2, coords=(0, 0), name="testregion", slide_type=types.HE)

    st = time.time()
    TissueDetectionHE(
        mask_name="tissue",
        threshold=19,
        min_region_size=1000,
        outer_contours_only=False,
        max_hole_size=10000,
        use_saturation=True,
    ).apply(im1)
    print(
        f"TissueDetectionHE took {(time.time() - st) // 60:.0f}m {(time.time() - st) % 60:.0f}s"
    )

    st = time.time()
    im1.masks["tissue"][im1.masks["tissue"] == 127] = 1  # convert 127s to 1s
    im1.image *= np.expand_dims(im1.masks["tissue"], 2)  # element wise in-place mm

    maxx = ((im1.shape[0] // tile_size) * tile_size) - tile_size
    maxy = ((im1.shape[1] // tile_size) * tile_size) - tile_size

    tilecounter = 0
    tile_mask_sum_max = 0
    wsiname = wsi.stem
    Path(out_folder + "/" + wsiname).mkdir(parents=True, exist_ok=True)
    coverage = threshold * tile_size * tile_size

    for tile_x_start in range(0, maxx, tile_size):
        for tile_y_start in range(0, maxy, tile_size):
            tile_x_end = tile_x_start + tile_size
            tile_y_end = tile_y_start + tile_size
            tile_mask = im1.masks["tissue"][
                tile_x_start:tile_x_end, tile_y_start:tile_y_end
            ]
            tile_mask_sum = np.sum(tile_mask)
            if tile_mask_sum < coverage:
                continue
            else:
                if tile_mask_sum_max < tile_mask_sum:
                    tile_mask_sum_max = tile_mask_sum
                Image.fromarray(
                    im1.image[tile_x_start:tile_x_end, tile_y_start:tile_y_end]
                ).save(
                    f"{out_folder}/{wsiname}/{wsiname}-{tile_x_start}_{tile_y_start}.jpg"
                )
                tilecounter += 1

    print(
        f"PIL Image save took {(time.time() - st) // 60:.0f}m {(time.time() - st) % 60:.0f}s"
    )
    print("\r\n")
    return tilecounter


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
    wsi_tilecount = extract_tiles(wsi, lvl=resolution, tile_size=tile_size, threshold=coverage)
    print(
        f"************ Extracted {wsi_tilecount} tiles from {wsi.name} in {(time.time() - start_time) // 60:.0f}m {(time.time() - start_time) % 60:.0f}s ************"
    )
    print("\r\n")
