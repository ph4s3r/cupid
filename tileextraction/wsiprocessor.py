##########################################################################################################
# Author: Peter Karacsonyi                                                                               #
# Last updated: 2024 feb 8                                                                               #
# Functionality to be run by TileExtractor-Runner.py: (processing a single wsi only)                     #
# extracts useful jpeg tiles (using pathml TissueDetectionHE mask) from a wsi                            #
# slicing the wsi to 4 parts to make sure it can fit into memory, warns when swap needs to be used       #
##########################################################################################################


###########
# imports #
###########
import os
import cv2
import time
import json
# import psutil
import argparse
import openslide
import numpy as np
from PIL import Image
from pathlib import Path
from pathml.core import Tile, types
from pathml.preprocessing import TissueDetectionHE


#####################################################################################
# operations on a numpy.ndarray region: pathML TissueDetectionHE and PIL Image save #
#####################################################################################
def TissueDetectandSave(image_nparray, coverage, i, config: dict, wsiname, out_folder):
    im = Tile(image_nparray, coords=(0, 0), name="quadrant", slide_type=types.HE)

    st = time.time()
    TissueDetectionHE(
        mask_name="tissue",
        threshold=config.get('TissueDetectionHE_threshold'),
        min_region_size=config.get('TissueDetectionHE_min_region_size'),
        outer_contours_only=False,
        max_hole_size=config.get('TissueDetectionHE_max_hole_size'),
        use_saturation=config.get('TissueDetectionHE_use_saturation')
    ).apply(im)
    print(f"\tTissueDetectionHE [{i}] took {(time.time() - st) // 60:.0f}m {(time.time() - st) % 60:.0f}s")

    st = time.time()
    im.masks["tissue"][im.masks["tissue"] == 127] = 1  # convert 127s to 1s
    im.image *= np.expand_dims(im.masks["tissue"], 2)  # element wise in-place mm

    maxx = ((im.shape[0] // config.get('tile_size')) * config.get('tile_size')) - config.get('tile_size')
    maxy = ((im.shape[1] // config.get('tile_size')) * config.get('tile_size')) - config.get('tile_size')

    tilecounter = 0
    tile_x_max = 0
    tile_y_max = 0
    tiles_checked = 0
    Path(out_folder + "/" + wsiname).mkdir(parents=True, exist_ok=True)
    # can be 250.000 maximum (tiles_size**2) x value of 1 where mask is covering the image
    threshold = coverage * config.get('tile_size') * config.get('tile_size')

    for tile_x_start in range(0, maxx, config.get('tile_size')):
        for tile_y_start in range(0, maxy, config.get('tile_size')):
            tile_x_end = tile_x_start + config.get('tile_size')
            tile_y_end = tile_y_start + config.get('tile_size')
            tile_mask = im.masks["tissue"][
                tile_x_start:tile_x_end, tile_y_start:tile_y_end
            ]
            if np.sum(tile_mask) >= threshold:
                if tile_x_max < tile_x_start:
                    tile_x_max = tile_x_start
                if tile_y_max < tile_y_start:
                    tile_y_max = tile_y_start
                    # for options see https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#jpeg
                Image.fromarray(
                    im.image[tile_x_start:tile_x_end, tile_y_start:tile_y_end]
                ).save(
                    f"{out_folder}/{wsiname}/{wsiname}-{tile_x_start}_{tile_y_start}.jpg",
                    quality=95,
                    keep_rgb=True
                )
                tilecounter += 1
            tiles_checked += 1
    # print(f"\tDEBUG             [{i}]: tile_x_max: {tile_x_max}, tile_y_max: {tile_y_max}, tiles: {tilecounter} from total of {tiles_checked}")
    print(f"\tPIL Image save    [{i}] took {(time.time() - st) // 60:.0f}m {(time.time() - st) % 60:.0f}s\r\n")
    return tilecounter

    

##################################################################################################
# load the slide with Openslide, convert 1/4th of the PIL Image to np.ndarray then run 4 passes  #
##################################################################################################

def processWSI(wsi, config: dict, out_folder):

    tiles_total = 0
    openslide_wsi = openslide.OpenSlide(wsi)

    q_pixels = ((openslide_wsi.level_dimensions[config.get('resolution')][1] * openslide_wsi.level_dimensions[config.get('resolution')][0]) / 4)
    w = openslide_wsi.level_dimensions[config.get('resolution')][1]
    h = openslide_wsi.level_dimensions[config.get('resolution')][0]
    q_pixels = int((w * h) / 4)

    # first half of the pixels per dimension - better to be divisible by config.get('tile_size')
    x1 = ((w // config.get('tile_size')) // 2) * config.get('tile_size')
    y1 = ((h // config.get('tile_size')) // 2) * config.get('tile_size')
    # another half of the pixels per dimension
    x2 = int(w - x1) + config.get('tile_size')
    y2 = int(h - y1) + config.get('tile_size')

    print("pyramid info: ")
    print("\tlevel_count: ", openslide_wsi.level_count)
    print("\tfirst 3 level_dimensions (h,w): ", openslide_wsi.level_dimensions[0:3])
    print(f"\trgb pixels to process on this resolution: {q_pixels:,}")
    print("")

    qregions = [
        (0,0, x1, y1),
        (x1,0, x2, y1),
        (0,y1, x1, y2),
        (x1,y1, x1, y2)
    ]

    for i in range(4):
        print(f"processing quadrant {i+1}")
        # print(f"\tDEBUG             [{i}]: bounding boxes (loc_x, loc_y, size_x, size_y): {qregions[i]}")
        st = time.time()
        image_pil = openslide_wsi.read_region(
            location=(qregions[i][0], qregions[i][1]),
            size=(
                qregions[i][2],
                qregions[i][3],
            ),
            level=config.get('resolution')
        )

        print(f"\tread_region()     [{i}] {(time.time() - st) // 60:.0f}m {(time.time() - st) % 60:.0f}s")
        st = time.time()
        # mem_available_mb = int(psutil.virtual_memory().available / (1024**2))
        image_nparray = cv2.cvtColor(np.asarray(image_pil), cv2.COLOR_RGBA2RGB).astype(
            np.uint8
        )

        # nparray_memory_usage_mb = int((image_nparray.size * image_nparray.itemsize) / (1024**2))
        # if nparray_memory_usage_mb > mem_available_mb:
        #     print(f"\tWARNING: image (size: {nparray_memory_usage_mb }MB) did not fit in the memory ({mem_available_mb}MB was available), swapped out..")
        print(f"\tnp.asarray()      [{i}] took {(time.time() - st) // 60:.0f}m {(time.time() - st) % 60:.0f}s")
        tiles_total += TissueDetectandSave(image_nparray, config.get('coverage'), i, config, Path(wsi).stem, out_folder)
        del image_nparray
    
    openslide_wsi.close()
    return tiles_total


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="WSI Tile Extractor")
    parser.add_argument("wsi", type=str, help="Path to the WSI file")
    parser.add_argument("config_json", type=str, help="JSON string of the config")
    parser.add_argument("out_folder", type=str, help="Output folder path")
    args = parser.parse_args()
    
    st = time.time()

    print(f"************ Processing {args.wsi} ************")
    print("")

    try:
        assert os.path.isfile(args.config_json), f"file not found {args.config_json}"
        with open(args.config_json, 'r') as f:
            config = json.loads(f.read())
    except Exception as e:
        print(f"Could not read config file {args.config_json}")
        print(f"Error: {e}")
        os._exit(78)

    tiles_extracted = processWSI(args.wsi, config, args.out_folder)
    print(f"************ Processed {Path(args.wsi).stem} extracting {tiles_extracted} tiles in {(time.time() - st) // 60:.0f}m {(time.time() - st) % 60:.0f}s ************\r\n")
