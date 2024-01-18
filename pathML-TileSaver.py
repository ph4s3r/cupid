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
import lib
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
h5_subdir = "h5-infer-unknown"

###########################
# where to save the tiles #
###########################
tile_dir = "tiles-infer-zero"








#########################################################################
# inserting tiles from h5path with TransformedPathmlTileSet to datasets #
#########################################################################
datasets = []
ds_fullsize = 0
h5folder = Path(base_dir) / Path(h5_subdir)
h5files = list(h5folder.glob("*.h5path"))
for h5file in h5files:
    ds_start_time = time.time()
    print(f"creating dataset from {str(h5file)} started at {datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')}")
    datasets.append(lib.TransformedPathmlTileSet(h5file))
    ds_complete = time.time() - ds_start_time
    print('dataset processed in {:.0f}m {:.0f}s'.format(ds_complete // 60, ds_complete % 60))

for ds in datasets:
    ds_fullsize += ds.dataset_len

full_ds = torch.utils.data.ConcatDataset(datasets)
del h5files, datasets, ds_start_time


######################
# set up dataloaders #
######################
batch_size = 1024 # need to max the batch out by seeing how much memory it takes (nvitop!!)
dataloader = torch.utils.data.DataLoader(
    full_ds, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True
)

print(f"after filtering the dataset for usable tiles, we have left with {dataloader.dataset.cummulative_sizes[-1]} tiles from the original {ds_fullsize}. Please note that this length might only be one dataset's length, so double check.")


##########################
# function to save tiles #
##########################
def save_tiles(dataloader, save_dir):
    """
    Save tiles from multiple DataLoaders as PNG images.

    Args:
    dataloaders (list of DataLoader): List of DataLoaders to process.
    save_dir (str): Directory path where images will be saved.

    Returns:
    None
    """

    for batch in dataloader:
        images, _, tile_labels, _ = batch

        tile_keys = tile_labels['tile_key']
        classes = tile_labels.get('class', ['infer'] * len(images))
        wsi_name = tile_labels['wsi_name']

        # training tiles will be put into a directory named after their class [0,1]
        # tiles without class (for inference) will be put into a dir named 'infer'
        for im, key, cl, name in zip(images, tile_keys, classes, wsi_name):
            if type(cl) == str:
                classlabel = str(cl)
            else:
                classlabel = str(cl.cpu().item())
            filename = f"{name}_{key}.png".replace("(", "").replace(")", "").replace(",", "_").replace(" ", "")
            try:
                save_image(im, os.path.join(save_dir, classlabel, filename))
            except FileNotFoundError: # need to create tile subdir
                Path(os.path.join(save_dir, classlabel)).mkdir(parents=True, exist_ok=True)
                save_image(im, os.path.join(save_dir, classlabel, filename))
    
                


##############
# save tiles #
##############
tile_dir = Path(base_dir) / Path(tile_dir)
tile_dir.mkdir(parents=True, exist_ok=True)
st_start_time = time.time()
save_tiles(dataloader, tile_dir)
st_complete = time.time() - st_start_time
print('saving completed in {:.0f}m {:.0f}s'.format(st_complete // 60, st_complete % 60))

