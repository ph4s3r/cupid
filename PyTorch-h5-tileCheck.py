##########################################################################################################
# Author: Mihaly Sulyok & Peter Karacsonyi                                                               #
# Last updated: 2023 Dec 10                                                                              #
# This workbook loads wsi processed slides/tiles h5path file and trains a deep learning model with them  #
# Input: h5path files                                                                                    #
# Output: trained model & results                                                                        #
##########################################################################################################


# imports
import os
# local files
if os.name == "nt":
    import helpers.openslideimport  # on windows, openslide needs to be installed manually, check local openslideimport.py
import helpers.ds_means_stds
import lib
# pip
import numpy as np
import torch
import time
import pathml
from pathlib import Path
from torchvision.transforms import (
    v2,
)

# set h5path directory
h5folder = Path("G:\\placenta\\h5\\")
h5files = list(h5folder.glob("*.h5path"))
# set model checkpoint directory
model_checkpoint_dir = Path("G:\\placenta\\training_checkpoints\\")
model_checkpoint_dir.mkdir(parents=True, exist_ok=True)
# set training results directory
result_path = Path("G:\\placenta\\training_results\\")
result_path.mkdir(parents=True, exist_ok=True)
# set tiles directory
tile_dir = Path("G:\\placenta\\tiles\\")
tile_dir.mkdir(parents=True, exist_ok=True)

# don't change the order without knowing exactly what you are doing! all transformations have specific input requirements.
transforms = v2.Compose(
    [
        v2.ToImage(),                                           # this operation reshapes the np.ndarray tensor from (3,h,w) to (h,3,w) shape
        v2.ToDtype(torch.float32, scale=True),                  # works only on tensor
        v2.Lambda(lambda x: x.permute(1, 0, 2)),                # get our C, H, W format back, otherwise Normalize will fail
        v2.Lambda(lambda x: x / 255.0),                         # convert pixel values to [0, 1] range
    ]
)

maskforms = v2.Compose(
    [
        v2.ToImage(),                                           # this operation reshapes the np.ndarray tensor from (3,h,w) to (h,3,w) shape
        v2.Lambda(lambda x: x.permute(1, 0, 2)),                # get our C, H, W format back
        v2.Lambda(lambda x: x / 127.),                          # convert pixel values to [0., 1.] range
        v2.ToDtype(torch.uint8)                                 # float to int
    ]
)

class TransformedPathmlTileSet(pathml.ml.TileDataset):
    def __init__(self, h5file):
        super().__init__(h5file)
        self.usable_indices = self._find_usable_tiles()
        self.file_label = Path(self.file_path).stem  # Extract the filename without extension

    def _find_usable_tiles(self):
        usable_indices = []
        threshold_percent = 0.4
        threshold_val = int(500 * 500 * threshold_percent)
        initial_length = super().__len__()

        for idx in range(initial_length):
            _, tile_masks, _, _ = super().__getitem__(idx)
            coverage = np.sum(tile_masks == 127.)
            if coverage >= threshold_val:
                usable_indices.append(idx)

        return usable_indices

    def __len__(self):
        return len(self.usable_indices)

    def __getitem__(self, idx):
        actual_idx = self.usable_indices[idx]
        tile_image, tile_masks, tile_labels, slide_labels = super().__getitem__(actual_idx)
        tile_image = transforms(tile_image)
        tile_masks = maskforms(tile_masks)

        # Extract tile key from the original dataset
        tile_labels['tile_key'] = self.tile_keys[actual_idx]
        tile_labels['source_file'] = self.file_label

        return (tile_image, tile_masks, tile_labels, slide_labels)

datasets = []
ds_fullsize = 0
for h5file in h5files:
    datasets.append(TransformedPathmlTileSet(h5file))

for ds in datasets:
    ds_fullsize += ds.dataset_len

full_ds = torch.utils.data.ConcatDataset(datasets)

# determine global means and stds for the full dataset (reads ~5GB/min)
# mean, std = helpers.ds_means_stds.mean_stds(full_ds)
# if done, just write it back to v2.Normalize() and run agains

# fixed generator for reproducible split results (https://pytorch.org/docs/stable/data.html#torch.utils.data.random_split)
generator = torch.Generator().manual_seed(42)
train_cases, val_cases, test_cases = torch.utils.data.random_split( # split to 70% train, 20% val & 10% test
    full_ds, [0.7, 0.2, 0.1], generator=generator
)

batch_size = 64

# num_workers>0 still causes problems...
train_loader = torch.utils.data.DataLoader(
    train_cases, batch_size=batch_size, shuffle=True, num_workers=0
)
val_loader = torch.utils.data.DataLoader(
    val_cases, batch_size=batch_size, shuffle=True, num_workers=0
)
test_loader = torch.utils.data.DataLoader(
    test_cases, batch_size=batch_size, shuffle=True, num_workers=0
)
print(f"after filtering the dataset for usable tiles, we have left with {len(train_cases) + len(val_cases) + len(test_cases)} tiles from the original {ds_fullsize}")

# saving tiles
start_time = time.time()
dataloaders = [train_loader, val_loader, test_loader]
lib.save_tiles(dataloaders, tile_dir)
time_elapsed = time.time() - start_time
print('saving completed in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))