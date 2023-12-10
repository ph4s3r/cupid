
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

wsi_folder = "G:\\echinov2\\wsi\\"
wsi_filename = "echino41.tif"
wsi = wsi_folder + wsi_filename

pml_wsi = HESlide(
    wsi,
    backend="openslide",
    slide_type=types.HE,
)

print("shape: ", pml_wsi.shape)
print("level_count: ", pml_wsi.slide.level_count)
print("level_dimensions (h,w): ", pml_wsi.slide.slide.level_dimensions)
print("color profile: ", pml_wsi.slide.slide.color_profile.profile.profile_description)
print("rgb?: ", pml_wsi.slide_type.rgb)
print("masks: ", len(pml_wsi.masks))
print("tiles: ", len(pml_wsi.tiles))

resolution_level = 4

region = pml_wsi.slide.extract_region(location=(0, 0), size=pml_wsi.slide.slide.level_dimensions[resolution_level], level=resolution_level)

print("extracted region.shape:", region.shape[0:1])

smaller_dim = min(region.shape[0:2])

region = region[0:smaller_dim, 0:smaller_dim]

print("removed unnecessary part (right half of img is displayed blank.. not sure why), new region shape:", region.shape)

random_x = np.random.randint(0, high=region.shape[0], size=10, dtype=int)
random_y = np.random.randint(0, high=region.shape[1], size=10, dtype=int)

print("region dtype:", type(region))

print("some random samples: ")
for i in range(10):
    print(region[random_x[i], random_y[i]])

def customTile():
    return Tile(region, coords=(0, 0), name="testregion", slide_type=types.HE)

tile = customTile()

fig, axarr = plt.subplots(nrows=2, ncols=3)

# plot original image
axarr[0][0].set_title("Original Mask", fontsize = fontsize)
axarr[0][0].imshow(tile.image)

# do binary threshold
bt = BinaryThreshold(
    mask_name="binary_threshold",
    use_otsu=False,
    threshold=220,
    inverse=False
).apply(tile)

# plot bin threshold
axarr[0][1].set_title("Binary threshold Mask", fontsize = fontsize)
axarr[0][1].imshow(tile.masks["binary_threshold"])

# do morphClose
ckernel = 7
citer = 7

mc = MorphClose(
    mask_name = "binary_threshold", 
    kernel_size=ckernel,
    n_iterations=citer
).apply(tile)

# plot morphClose
axarr[1][0].set_title(f"MorphClose, kernel={ckernel}, n={citer}", fontsize = fontsize)
axarr[1][0].imshow(tile.masks["binary_threshold"])

for ax in axarr.ravel():
    ax.set_yticks([])
    ax.set_xticks([])

# do morphOpen
okernel = 7
oiter = 7

mc = MorphOpen(
    mask_name = "binary_threshold", 
    kernel_size=okernel,
    n_iterations=oiter
).apply(tile)

# plot morphOpen
axarr[1][1].set_title(f"MorphOpen, kernel={okernel}, n={oiter}", fontsize = fontsize)
axarr[1][1].imshow(tile.masks["binary_threshold"])

# do ForegroundDetection
fgd = ForegroundDetection(mask_name="binary_threshold", min_region_size=500, max_hole_size=1500, outer_contours_only=False).apply(tile)

# plot ForegroundDetection
axarr[1][2].set_title(f"ForegroundDetection", fontsize = fontsize)
axarr[1][2].imshow(tile.masks["binary_threshold"])

# conclude plotting
for ax in axarr.ravel():
    ax.set_yticks([])
    ax.set_xticks([])

plt.show()
