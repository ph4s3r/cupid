import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter # launch with http://localhost:6006/
import os
import torch
from pathlib import Path
from torchvision.transforms import v2
import pathml
import numpy as np
from torchvision.utils import save_image

def human_readable_size(size, decimal_places=2):
    """Convert a size in bytes to a human-readable format."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size < 1024.0 or unit == "TB":
            break
        size /= 1024.0
    return f"{size:.{decimal_places}f} {unit}"

def print_file_sizes(directory):
    """List files and their sizes in a directory."""
    p = Path(directory)
    print("file sizes in directory", p)
    if not p.is_dir():
        return "Provided path is not a directory"

    for file in p.glob('*'):
        if file.is_file():
            size = file.stat().st_size
            print(file.name, human_readable_size(size))


# plot a batch of tiles with masks
def vizBatch(im, tile_masks, tile_labels = None):
    # create a grid of subplots
    _, axes = plt.subplots(4, 4, figsize=(8, 8))  # Adjust figsize as needed
    axes = axes.flatten()

    for i in range(8):  # Display only the first batch_size tiles, duplicated
        img = im[i].permute(1, 2, 0).numpy()
        mask = tile_masks[i].permute(1, 2, 0).numpy()

        # Display image without mask
        axes[2*i].imshow(img)
        axes[2*i].axis("off")
        if tile_labels is not None:
            label = ", ".join([f"{v[i]}" for _, v in tile_labels.items()])
        else:
            label = ""
        axes[2*i].text(3, 10, label, color="white", fontsize=6, backgroundcolor="black")

        # Display image with mask overlay
        axes[2*i + 1].imshow(img)
        axes[2*i + 1].imshow(mask, alpha=0.5, cmap='terrain')  # adjust alpha as needed
        axes[2*i + 1].axis("off")

    plt.tight_layout()
    plt.show()


#################
# augmentations #
#################
transforms = v2.Compose( # don't change the order without knowing exactly what you are doing! all transformations have specific input requirements.
    [
        v2.ToImage(),                                           # this operation reshapes the np.ndarray tensor from (3,h,w) to (h,3,w) shape
        v2.ToDtype(torch.float32, scale=True),                  # works only on tensor
        v2.Lambda(lambda x: x.permute(1, 0, 2)),                # get our C, H, W format back, otherwise Normalize will fail
        v2.Lambda(lambda x: x / 255.0),                         # convert pixel values to [0, 1] range
        v2.RandomApply(
            transforms=[
                v2.RandomRotation(degrees=(0, 359)),
                v2.ColorJitter(brightness=.3, hue=.2, saturation=.2, contrast=.3)
            ]
        , p=0.5),
        v2.Resize(size=256, antialias=False),                   # same size as the tile im
    ]
)

maskforms = v2.Compose(
    [
        v2.ToImage(),                                           # this operation reshapes the np.ndarray tensor from (3,h,w) to (h,3,w) shape
        v2.Lambda(lambda x: x.permute(1, 0, 2)),                # get our C, H, W format back
        v2.Lambda(lambda x: x / 127.),                          # convert pixel values to [0., 1.] range
        v2.ToDtype(torch.uint8),                                # float to int
        v2.Resize(size=256, antialias=False)                    # same size as the tile im
    ]
)


##########################################################
# drop tiles covered less than min_mask_coverage by mask #
##########################################################
min_mask_coverage = 0.35
class TransformedPathmlTileSet(pathml.ml.TileDataset):
    def __init__(self, h5file):
        super().__init__(h5file)
        self.dimx = self.tile_shape[0]
        self.dimy = self.tile_shape[1]
        self.usable_indices = self._find_usable_tiles()
        self.file_label = Path(self.file_path).stem  # Extract the filename without extension+

    def _find_usable_tiles(self):
        usable_indices = []
        threshold_percent = min_mask_coverage
        threshold_val = int(self.dimx * self.dimy * threshold_percent)
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


def save_tiles(dataloaders, save_dir):
    """
    Save tiles from multiple DataLoaders as PNG images.

    Args:
    dataloaders (list of DataLoader): List of DataLoaders to process.
    save_dir (str): Directory path where images will be saved.

    Returns:
    None
    """

    for loader in dataloaders:
        for batch in loader:
            images, _, tile_labels, _ = batch  # Adjust if your dataloader structure is different

            tile_keys = tile_labels['tile_key']
            classes = tile_labels['class']
            wsi_name = tile_labels['wsi_name']

            for im, key, cl, name in zip(images, tile_keys, classes, wsi_name):
                # Construct filename using 'source_file' and 'tile_key'
                filename = f"{key}cl{cl}.png".replace("(", "").replace(")", "_").replace(",", "_").replace(" ", "")
                try:
                    save_image(im, os.path.join(save_dir, name, filename))
                except FileNotFoundError: # need to create tile subdir
                    Path(os.path.join(save_dir, name)).mkdir(parents=True, exist_ok=True)
                    save_image(im, os.path.join(save_dir, name, filename))
                
