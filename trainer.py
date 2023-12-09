##########################################################################################################
# Author: Mihaly Sulyok & Peter Karacsonyi                                                               #
# Last updated: 2023 Dec 9                                                                               #
# This workbook loads wsi processed slides/tiles h5path file and trains a deep learning model with them  #
# Input: h5path files                                                                                    #
# Output: trained model & results                                                                        #
##########################################################################################################

# imports
import os
if os.name == "nt":
    import helpers.openslideimport  # on windows, openslide needs to be installed manually, check local openslideimport.py
import helpers.ds_means_stds
import time
import torch
import pathml
import matplotlib.pyplot as plt
from pathlib import Path
from torchvision.models import resnet
from torchvision.transforms import (
    v2,
)  # v2 is the newest / fastest: https://pytorch.org/vision/stable/transforms.html

# set h5path directory
h5folder = Path("G:\\echinov3\\h5\\")
h5files = list(h5folder.glob("*.h5path"))

# tile transformations based on resnet18 requirements (https://pytorch.org/hub/pytorch_vision_resnet/)
# need (3,h,w) shape
# need normalization with mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
# values need to be converted to range [0,1]

# don't change the order without knowing exactly what you are doing! all transformations have specific input requirements.
transforms = v2.Compose(
    [
        v2.ToImage(),                                           # this operation reshapes the np.ndarray tensor from (3,h,w) to (h,3,w) shape
        v2.ToDtype(torch.float32, scale=True),                  # works only on tensor
        v2.Lambda(lambda x: x.permute(1, 0, 2)),                # get our C, H, W format back, otherwise Normalize will fail
        # v2.Normalize(                                         # need to get the global means and stds to do this
        #     mean=[0.485, 0.456, 0.406],
        #     std=[0.229, 0.224, 0.225]
        # ),
        v2.Lambda(lambda x: x / 255.0),                         # convert pixel values to [0, 1] range (this one always works)
        v2.RandomResizedCrop(size=(224, 224), antialias=True)   # 224 is probably the optimal size, but can be tuned
    ]
)

class TransformedPathmlTileSet(pathml.ml.TileDataset):
    def __getitem__(self, idx):
        tile_image, tile_masks, tile_labels, slide_labels = super(TransformedPathmlTileSet, self).__getitem__(idx)

        tile_image = transforms(tile_image)

        return (tile_image, tile_masks, tile_labels, slide_labels)

datasets = []
for h5file in h5files:
    datasets.append(TransformedPathmlTileSet(h5file))

full_ds = torch.utils.data.ConcatDataset(datasets)

# determine global means and stds for the full dataset (reads ~5GB/min)
# mean, std = helpers.ds_means_stds.mean_stds(full_ds)
# if done, just write it back to v2.Normalize() and run again

# (optional) with fixed generator for reproducible split results (https://pytorch.org/docs/stable/data.html#torch.utils.data.random_split)
generator = torch.Generator().manual_seed(42)
# splitting to 70% train, 20% val & 10% test
train_cases, val_cases, test_cases = torch.utils.data.random_split(
    full_ds, [0.7, 0.2, 0.1], generator=generator
)

batch_size = 48

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

# plot a batch of tiles
def vizBatch(batch_tensor, tile_labels):
    # create a grid of subplots
    _, axes = plt.subplots(4, 4, figsize=(8, 8))
    axes = axes.flatten()

    for i, (ax, img) in enumerate(zip(axes, batch_tensor)):
        # imshow requires (w,h,c) shape
        ax.imshow(img.permute(1, 2, 0).numpy()) 
        ax.axis("off")
        # extract label for the current tile
        label = ", ".join([f"{v[i]}" for _, v in tile_labels.items()])
        # draw label on each image
        ax.text(3, 10, label, color="white", fontsize=6, backgroundcolor="black")

    plt.tight_layout()
    plt.show()

# get a batch of transformed training data just to visualize
images, tile_masks, tile_labels, slide_labels = next(iter(train_loader))
# vizBatch(images, tile_labels)

# elnezest
label_mapping = {classname: i for i, classname in enumerate(set(slide_labels.get('class')))}

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
num_epochs = 2
learning_rate = 0.001

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

for epoch in range(num_epochs):
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
        
        if (i+1) % 100 == 0:
            time_elapsed = time.time() - start_time
            print ("Epoch [{}/{}], Step [{}/{}] Loss: {:.4f} in {:.0f}m {:.0f}s"
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item(), time_elapsed // 60, time_elapsed % 60))

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
PATH = '"G:\\echinov3\\clinical\\training_checkpoints\\"'
torch.save(model.state_dict(), PATH+"resnet.ckpt")
pass
