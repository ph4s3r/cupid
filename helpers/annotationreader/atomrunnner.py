import atom

from pathml import types
from pathml.core import SlideData
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import time

# set working directory
base_dir = Path("G:\echinov3")
# wsi folder
wsi_subfolder = "wsi"
# place geojsons into this folder inside base with same name as the image (echino23.tiff / echino38.gejson)
geojson_subfolder = "geojson"
# writing the masks as numpy arrays here
numpy_mask_subfolder = "wsiannotation-dumps"

data_dir = base_dir / Path(wsi_subfolder)               # input
geojson_dir = base_dir / Path(geojson_subfolder)        # input 
numpy_mask_dir = base_dir / Path(numpy_mask_subfolder)  # output

# read wsi files
wsi_paths = list(data_dir.glob("*.tif"))

print(f"picked up {wsi_paths} wsi(s) to process annotations for")

for wsi in wsi_paths:
    filename = wsi.stem
    geojson_file = f"{geojson_dir}\{filename}.geojson"
    slide = SlideData(wsi.as_posix(), name = wsi.as_posix(), backend = "openslide",  slide_type = types.HE)

    start_time = time.time()
    mask_array = atom.annotationToMask(slide.shape, slide.name, geojson_file)

    time_elapsed = time.time() - start_time
    print(f"Creating mask_array for {geojson_file} took {time_elapsed} s")

    np.save(f"{str(numpy_mask_dir)}\{filename}", mask_array, allow_pickle=True)
    print(f"wrote annotation mask to disk G:\echinov3\wsiannotation-dumps\{filename}")

    # plotting wsi
_, axs = plt.subplots(figsize=(2, 2))
slide.plot(ax=axs)
x_limits = axs.get_xlim()
y_limits = axs.get_ylim()

# plotting mask on wsi
axs.imshow(mask_array, cmap='plasma', alpha=0.2, extent=(x_limits[0], x_limits[1], y_limits[0], y_limits[1]))
axs.set_title(label=f"{slide.name.split('/')[-1]} + annotation mask in yellow" , fontsize=8)