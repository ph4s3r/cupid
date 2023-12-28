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
import time
import numpy as np
import torch
import pathml
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support
from torchvision.models import resnet
from torchvision.transforms import (
    v2,
)  # v2 is the newest: https://pytorch.org/vision/stable/transforms.html

# set h5path directory
h5folder = Path("G:\\placenta\\h5\\")
h5files = list(h5folder.glob("*.h5path"))
model_checkpoint_dir = Path("G:\\placenta\\training_checkpoints\\")
model_checkpoint_dir.mkdir(parents=True, exist_ok=True)
result_path = Path("G:\\placenta\\training_results\\")
result_path.mkdir(parents=True, exist_ok=True)
tile_dir = Path("G:\\placenta\\tiles\\")
tile_dir.mkdir(parents=True, exist_ok=True)


# tile transformations based on resnet18 requirements (https://pytorch.org/hub/pytorch_vision_resnet/) it needs:
# - (3,h,w) shape
# - normalization (with global means and stds)
#  - values to be converted to range [0,1]

# don't change the order without knowing exactly what you are doing! all transformations have specific input requirements.
transforms = v2.Compose(
    [
        v2.ToImage(),                                           # this operation reshapes the np.ndarray tensor from (3,h,w) to (h,3,w) shape
        v2.ToDtype(torch.float32, scale=True),                  # works only on tensor
        v2.Lambda(lambda x: x.permute(1, 0, 2)),                # get our C, H, W format back, otherwise Normalize will fail
        v2.Lambda(lambda x: x / 255.0),                         # convert pixel values to [0, 1] range
        # v2.Normalize(                                         # problem is that images are mostly white. need to calc mean and stds of usable pixels only
        #     mean=[.0, .0, .0],
        #     std=[1., 1., 1.]
        # ),
        # v2.Resize(size=(256, 256), antialias=False)             # resnet in its base form works best with 224/256, senet however works well with 500 
    ]
)

maskforms = v2.Compose(
    [
        v2.ToImage(),                                           # this operation reshapes the np.ndarray tensor from (3,h,w) to (h,3,w) shape
        v2.Lambda(lambda x: x.permute(1, 0, 2)),                # get our C, H, W format back
        v2.Lambda(lambda x: x / 127.),                          # convert pixel values to [0, 1] range
        v2.ToDtype(torch.uint8)
    ]
)

class TransformedPathmlTileSet(pathml.ml.TileDataset):
    def __init__(self, h5file):
        super().__init__(h5file)
        self.usable_indices = self._find_usable_tiles()
        self.file_label = Path(self.file_path).stem  # Extract the filename without extension

    def _find_usable_tiles(self):
        usable_indices = []
        threshold_percent = 0.3
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

# (optional) with fixed generator for reproducible split results (https://pytorch.org/docs/stable/data.html#torch.utils.data.random_split)
generator = torch.Generator().manual_seed(42)
# splitting to 70% train, 20% val & 10% test
train_cases, val_cases, test_cases = torch.utils.data.random_split(
    full_ds, [0.7, 0.2, 0.1], generator=generator
)

batch_size = 16

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
dataloaders = [train_loader, val_loader, test_loader]
lib.save_tiles(dataloaders, tile_dir)

# tile_image, tile_masks, tile_labels, slide_labels = next(iter(train_loader))

# lib.vizBatch(tile_image, tile_masks)
# elnezest
# label_mapping = {classname: i for i, classname in enumerate(set(slide_labels.get('class')))}

# pretrained resnet18 (https://pytorch.org/vision/main/models/generated/torchvision.models.resnet18.html#torchvision.models.ResNet18_Weights)
ResNet = torch.hub.load(
    "pytorch/vision:v0.10.0", "resnet18", weights=resnet.ResNet18_Weights.IMAGENET1K_V1
)

# a se_resnet50 from torch hub
# hub_model = torch.hub.load(
#     'moskomule/senet.pytorch',
#     'se_resnet50',
#     pretrained=True,)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using {device}")

model = ResNet.to(device)

start_time = time.time()

# hyper-params
num_epochs = 10
learning_rate = 0.002

# loss and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

# to update learning rate
def update_lr(optimizer, lr):    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# train
total_step = len(train_loader)
curr_lr = learning_rate

learning_stats = {'train': [], 'val': []}
learning_stats_phase = "train"

for epoch in range(num_epochs):

    total_loss = 0
    total_correct = 0
    total = 0
    all_labels = []
    all_predictions = []

    for i, (images, tile_masks, tile_labels, slide_labels) in enumerate(train_loader):
        images = images.to(device)
        
        # kukazzuk ha a labelek jol lesznek fenn (ld README) vagy collate_fn
        labels = torch.tensor([int(label_mapping.get(tl)) for tl in tile_labels.get('class')]).to(device)

        # forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        total_correct += (predicted == labels).sum().item()
        all_labels.extend(labels.cpu().numpy())
        all_predictions.extend(predicted.cpu().numpy())
        
        if (i+1) % 100 == 0:
            time_elapsed = time.time() - start_time
            print ("Epoch [{}/{}], Step [{}/{}] Loss: {:.4f} in {:.0f}m {:.0f}s"
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item(), time_elapsed // 60, time_elapsed % 60))
    
    epoch_loss = total_loss / total_step
    epoch_acc = total_correct / total
    precision, recall, f1_score, _ = precision_recall_fscore_support(all_labels, all_predictions, average='weighted')

    # save epoch learning stats
    learning_stats[learning_stats_phase].append({
        'loss': epoch_loss, 
        'accuracy': epoch_acc, 
        'weighted_precision': precision,
        'weighted_recall': recall,
        'weighted_f1': f1_score
    })

    # show stats at the end of epoch
    print(f"Epoch {epoch+1}/{num_epochs} - Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1_score:.4f}")

    # decay learning rate
    if (epoch+1) % 20 == 0:
        curr_lr /= 3
        update_lr(optimizer, curr_lr)

# test the model
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for i, (images, tile_masks, tile_labels, slide_labels) in enumerate(test_loader):
        images = images.to(device)
        # kukazzuk ha a labelek jol lesznek fenn (ld README) vagy collate_fn
        labels = torch.tensor([int(label_mapping.get(tl)) for tl in tile_labels.get('class')]).to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Accuracy of the model on the test images: {} %'.format(100 * correct / total))

# save model checkpoint
dtcomplete = time.strftime("%Y%m%d-%H%M%S")
torch.save(model.state_dict(), str(model_checkpoint_dir)+"\\"+"resnet18-"+dtcomplete+".ckpt")
pass

lib.plotLearningCurve(learning_stats, result_path)