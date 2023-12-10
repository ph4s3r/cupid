# always start with openslide import
import os

if os.name == "nt":
    import helpers.openslideimport  # on windows, openslide needs to be installed manually, check local openslideimport.py

import numpy as np
from pathml.core import HESlide, Tile, types
import matplotlib.pyplot as plt
from pathml.preprocessing import (
    BoxBlur,
    GaussianBlur,
    MedianBlur,
    NucleusDetectionHE,
    StainNormalizationHE,
    SuperpixelInterpolation,
    ForegroundDetection,
    TissueDetectionHE,
    BinaryThreshold,
    MorphClose,
    MorphOpen,
)

fontsize = 8

wsi_folder = "G:\\echino\\wsi\\"
wsi_filename = "echino40.tif"
wsi = wsi_folder + wsi_filename

pml_wsi = HESlide(
    wsi,
    backend="openslide",
    slide_type=types.HE,
)

print("shape: ", pml_wsi.shape)
print("level_count: ", pml_wsi.slide.level_count)
print("level_dimensions: ", pml_wsi.slide.slide.level_dimensions)
print("color profile: ", pml_wsi.slide.slide.color_profile.profile.profile_description)
print("rgb?: ", pml_wsi.slide_type.rgb)
print("masks: ", len(pml_wsi.masks))
print("tiles: ", len(pml_wsi.tiles))

resolution_level = 5

region = pml_wsi.slide.extract_region(location=(0, 0), size=pml_wsi.slide.slide.level_dimensions[resolution_level], level=resolution_level)

print("extracted region.shape:", region.shape)

random_x = np.random.randint(0, high=region.shape[0], size=10, dtype=int)
random_y = np.random.randint(0, high=region.shape[1], size=10, dtype=int)

print("region dtype:", type(region))

print("some random samples: ")
for i in range(10):
    print(region[random_x[i], random_y[i]])

def smalltile():
    return Tile(region, coords=(0, 0), name="testregion", slide_type=types.HE)

thresholds = ["original", 220, "otsu"]
fig, axarr = plt.subplots(nrows=1, ncols=len(thresholds), figsize=(12, 6))
for i, thresh in enumerate(thresholds):
    tile = smalltile()
    if thresh == "original":
        axarr[i].set_title("Original Image", fontsize=fontsize)
        axarr[i].imshow(tile.image)
    elif thresh == "otsu":
        t = BinaryThreshold(mask_name="binary_threshold", inverse=True, use_otsu=True)
        t.apply(tile)
        axarr[i].set_title(f"Otsu Threshold", fontsize=fontsize)
        axarr[i].imshow(tile.masks["binary_threshold"])
    else:
        t = BinaryThreshold(
            mask_name="binary_threshold", threshold=thresh, inverse=True, use_otsu=False
        )
        t.apply(tile)
        axarr[i].set_title(f"Threshold = {thresh}", fontsize=fontsize)
        axarr[i].imshow(tile.masks["binary_threshold"])
for ax in axarr.ravel():
    ax.set_yticks([])
    ax.set_xticks([])
plt.tight_layout()
plt.show()
