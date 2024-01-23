##########################################################################################################
# Author: Peter Karacsonyi                                                                               #
# Last updated: 2024 jan 23                                                                              #
# Input: write jpeg tiles from a wsi                                                                     #
##########################################################################################################

##############
# IO folders #
##############
wsi_folder = "/mnt/bigdata/placenta/wsi_biggest"  # reads all wsi files in folder
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


def TissueDetectandSave(image_nparray, coverage, i):
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
    print(f"TissueDetectionHE [{i}] took {(time.time() - st) // 60:.0f}m {(time.time() - st) % 60:.0f}s")

    st = time.time()
    im.masks["tissue"][im.masks["tissue"] == 127] = 1  # convert 127s to 1s
    im.image *= np.expand_dims(im.masks["tissue"], 2)  # element wise in-place mm

    maxx = ((im.shape[0] // tile_size) * tile_size) - tile_size
    maxy = ((im.shape[1] // tile_size) * tile_size) - tile_size

    tilecounter = 0
    tile_x_max = 0
    tile_y_max = 0
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
                if tile_x_max < tile_x_start:
                    tile_x_max = tile_x_start
                if tile_y_max < tile_y_start:
                    tile_y_max = tile_y_start
                Image.fromarray(
                    im.image[tile_x_start:tile_x_end, tile_y_start:tile_y_end]
                ).save(
                    f"{out_folder}/{wsiname}/{wsiname}-{tile_x_start}_{tile_y_start}.jpg"
                )
                tilecounter += 1
    print(f"DEBUG [{i}]: tile_x_max: {tile_x_max}, tile_y_max: {tile_y_max}, tiles: {tilecounter}")
    print(f"PIL Image save [{i}] took {(time.time() - st) // 60:.0f}m {(time.time() - st) % 60:.0f}s") ; print("\r\n")

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

    # first half of the pixels per dimension - better to be divisible by tile_size
    x1 = ((w // tile_size) // 2) * tile_size
    y1 = ((h // tile_size) // 2) * tile_size
    # another half of the pixels per dimension
    x2 = int(w - x1)
    y2 = int(h - y1)

    print("shape and resolution details of the input file: ")
    print("\tlevel_count: ", openslide_wsi.level_count)
    print("\tfirst 3 level_dimensions (h,w): ", openslide_wsi.level_dimensions[0:2])
    print("\tfirst 3 level_downsamples (from 0 index to n): ", openslide_wsi.level_downsamples[0:2])
    print("\r\n")

    qregions = [
        # (location_start_x, location_start_y, size_x, size_y)
        (0,0, x1, y1),
        (x1,0, x2, y1),
        (0,y1, x1, y2),
        (x1,y1, x2, y2)
    ]

    for i in range(4):

        st = time.time()
        image_pil = openslide_wsi.read_region(
            location=(qregions[i][0], qregions[i][1]),
            size=(
                qregions[i][2],
                qregions[i][3],
            ),
            level=resolution,
        )
        print(
            f"openslide read_region()[{i}] {(time.time() - st) // 60:.0f}m {(time.time() - st) % 60:.0f}s"
        )

        st = time.time()
        image_nparray = cv2.cvtColor(np.asarray(image_pil), cv2.COLOR_RGBA2RGB).astype(
            np.uint8
        )
        del image_pil
        print(f"cv2 cv2.cvtColor(np.asarray())[{i}] took {(time.time() - st) // 60:.0f}m {(time.time() - st) % 60:.0f}s")
        
        tiles_total += TissueDetectandSave(image_nparray, coverage, i)
        del image_nparray

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
