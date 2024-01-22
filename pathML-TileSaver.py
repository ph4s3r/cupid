##########################################################################################################
# Author: Mihaly Sulyok & Peter Karacsonyi                                                               #
# Last updated: 2024 jan 18                                                                              #
# Input: h5path files                                                                                    #
# Output: tiles                                                                                          #
##########################################################################################################


# imports
import os
# local files
if os.name == "nt":
    import helpers.openslideimport  # on windows, openslide needs to be installed manually, check local openslideimport.py
from lib import TransformedPathmlTileSet, timeit, print_file_sizes
# pip
import time
import torch
from pathlib import Path
from datetime import datetime
from torchvision.utils import save_image


##############################
# where to load the h5s from #
##############################
base_dir = "/mnt/bigdata/placenta"
h5_subdir = "h5-train"


##################
# set batch size #
##################
batch_size = 512
# makes sense to max the batch out by seeing how much memory the dataloader takes (nvitop!!)


###########################
# where to save the tiles #
###########################
tile_dir = "tiles-train-500"


##########################
# function to save tiles #
##########################
@timeit
def save_tiles(dataloader, save_dir):
    """
    Save tiles from dataloader as PNG images.
    """

    for batch in dataloader:
        images, _, tile_labels, _ = batch
        # tiles will be put into a dir named after their slide filename
        for im, key, name in zip(images, tile_labels['tile_key'], tile_labels['wsi_name']):
            filename = f"{name}_{key}.png".replace("(", "").replace(")", "").replace(",", "_").replace(" ", "")
            try:
                save_image(im, os.path.join(save_dir, str(name), filename))
            except FileNotFoundError: # need to create tile subdir
                Path(os.path.join(save_dir, str(name))).mkdir(parents=True, exist_ok=True)
                save_image(im, os.path.join(save_dir, str(name), filename))


#########################################################################
# inserting tiles from h5path with TransformedPathmlTileSet to datasets #
#########################################################################      
h5folder = Path(base_dir) / Path(h5_subdir)
h5files = list(h5folder.glob("*.h5path"))

# see the file sizes
print_file_sizes(h5folder)

## check if tiles exist for a slide, then skip
tile_dir = Path(base_dir) / Path(tile_dir)
tile_dir.mkdir(parents=True, exist_ok=True)

tiles_exist_for = [f.stem for f in tile_dir.iterdir() if f.is_dir()]

print(f"WSI(s) {tiles_exist_for} will be skipped due to existing output folders. \r\n")

for h5file in h5files:
    if h5file.stem not in tiles_exist_for:

        ############################################
        # read h5, create dataset, prep dataloader #
        ############################################
        st = time.time()
        print(f"{str(h5file.stem)}: started creating dataset @ {datetime.fromtimestamp(time.time()).strftime('%H:%M:%S')}")
        dataloader = torch.utils.data.DataLoader(
            TransformedPathmlTileSet(h5file), batch_size=batch_size, num_workers=4, pin_memory=True
        )
        ct = time.time() - st
        print(f"{str(h5file.stem)}: processed in {ct // 60:.0f}m {ct % 60:.0f}s")
        print(f"{str(h5file.stem)}: {len(dataloader.dataset)} tiles left from the original {dataloader.dataset.original_length()} after dropping empty tiles.")

        ##############
        # save tiles #
        ##############
        save_tiles(dataloader, tile_dir)

    else:
        print(f"skipping {h5file.stem} because tiles directory exists")

