# https://kirillstrelkov.medium.com/python-profiling-with-vscode-3a17c0407833
# cProfiler saves to G/tmp

import atom_CUDA

from pathml import types
from pathml.core import SlideData
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import time

# set working directory
base_dir = Path("G:\v3")
# wsi folder
wsi_subfolder = "wsi"
# place geojsons into this folder inside base with same name as the image (23.tiff / 38.gejson)
geojson_subfolder = "geojson"
# writing the masks as numpy arrays here
numpy_mask_subfolder = "wsiannotation-dumps"

data_dir = base_dir / Path(wsi_subfolder)               # input
geojson_dir = base_dir / Path(geojson_subfolder)        # input 
numpy_mask_dir = base_dir / Path(numpy_mask_subfolder)  # output

# read wsi files
wsi_paths = list(data_dir.glob("*.tif"))
for wsi in wsi_paths:
    filename = wsi.stem
    geojson_file = f"{geojson_dir}\{filename}.geojson"
    slide = SlideData(wsi.as_posix(), name = wsi.as_posix(), backend = "openslide",  slide_type = types.HE)

    start_time = time.time()
    mask_array = atom_CUDA.annotationToMask(slide.shape, slide.name, geojson_file)

    time_elapsed = time.time() - start_time
    print(f"Creating mask_array for {geojson_file} took {time_elapsed} s")

    np.save(f"{str(numpy_mask_dir)}\{filename}", mask_array, allow_pickle=True)
    print(f"wrote annotation mask to disk G:\v3\wsiannotation-dumps\{filename}")