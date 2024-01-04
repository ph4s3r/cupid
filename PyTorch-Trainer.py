##########################################################################################################
# Author: Mihaly Sulyok & Peter Karacsonyi                                                               #
# Last updated: 2024 jan 4                                                                              #
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
import pathml
import torch
from pathlib import Path
from coolname import generate_slug # ...
from sklearn.metrics import precision_recall_fscore_support
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter # launch with http://localhost:6006/
from torchvision.transforms import (
    v2,
)  # v2 is the newest: https://pytorch.org/vision/stable/transforms.html

#####################
# configure folders #
#####################
base_dir = Path("/mnt/bigdata/placenta")
h5folder = base_dir / Path("h5")
h5files = list(h5folder.glob("*.h5path"))
model_checkpoint_dir = base_dir / Path("training_checkpoints")
result_path = base_dir / Path("training_results")
test_dataset_dir = base_dir / Path("test_dataset")

model_checkpoint_dir.mkdir(parents=True, exist_ok=True)
result_path.mkdir(parents=True, exist_ok=True)
test_dataset_dir.mkdir(parents=True, exist_ok=True)


##############################################################################################################
# instantiate tensorboard summarywriter (write the run's data into random subdir with some random funny name)#
##############################################################################################################
session_name = generate_slug(2)
print("Starting session ", session_name)
tensorboard_log_dir = base_dir / "tensorboard_data" / session_name
tensorboard_log_dir.mkdir(parents=True, exist_ok=True)
writer = SummaryWriter(log_dir=tensorboard_log_dir, comment=session_name)


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


#########################################################################
# inserting tiles from h5path with TransformedPathmlTileSet to datasets #
#########################################################################
datasets = []
ds_fullsize = 0
for h5file in h5files:
    print(f"creating dataset from {str(h5file)} with TransformedPathmlTileSet")
    datasets.append(TransformedPathmlTileSet(h5file))

for ds in datasets:
    ds_fullsize += ds.dataset_len

full_ds = torch.utils.data.ConcatDataset(datasets)


########################
# global std and means #
########################
determine_global_std_and_means = False
# speed = ~5GB/min
# if done, just write it back to v2.Normalize() and run again
if determine_global_std_and_means:
    mean, std = helpers.ds_means_stds.mean_stds(full_ds)


######################
# set up dataloaders #
######################
batch_size = 32 # larger batch is faster!
# fixed generator for reproducible split results
generator = torch.Generator().manual_seed(42)
train_cases, val_cases, test_cases = torch.utils.data.random_split( # split to 70% train, 20% val & 10% test
    full_ds, [0.7, 0.2, 0.1], generator=generator
)
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

######################################
# saving test dataset for evaluation #
######################################
test_dataset_file = str(test_dataset_dir)+"/"+session_name+"-test-dataset.pt"
print("saving test dataset to ", test_dataset_file)
test_data = []
test_targets = []

for data in test_cases:
    # data contains (tile_image, tile_masks, tile_labels, slide_labels)
    test_data.append(data[0]) # tile_image
    test_targets.append(data[2]) # tile_labels

torch.save({'data': test_data, 'targets': test_targets}, test_dataset_file)


###############
# save tiles? #
###############
tile_dir = base_dir / "tiles"
savetiles = False

tile_dir.mkdir(parents=True, exist_ok=True)
if savetiles:
    start_time = time.time()
    dataloaders = [train_loader, val_loader, test_loader]
    lib.save_tiles(dataloaders, tile_dir)
    time_elapsed = time.time() - start_time
    print('saving completed in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))


##################
# begin training #
##################
SE_RESNET50 = torch.hub.load(
    'moskomule/senet.pytorch',
    'se_resnet50',
    pretrained=True,
    verbose=True
    )

num_ftrs = SE_RESNET50.fc.in_features
SE_RESNET50.fc = torch.nn.Linear(num_ftrs, 2)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using {device}")
model = SE_RESNET50.to(device)

start_time = time.time()

# hyper-params
num_epochs = 80
learning_rate = 0.001

# loss and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model.parameters(), lr=0.6 / 1024 * batch_size, momentum=0.9, weight_decay=1e-4)

# to update learning rate
# scheduler = StepLR(optimizer, step_size=7, gamma=0.1, verbose=True)
scheduler = MultiStepLR(optimizer=optimizer, milestones=[10,30], gamma=0.1, verbose=True)

# train
total_step = len(train_loader)
curr_lr = learning_rate

# early stop class (val_loss)
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.001, verbose=False, consecutive=False):

        self.patience = patience
        self.verbose = verbose
        self.min_delta = min_delta
        self.consecutive = consecutive
        
        self.epoch = 0
        self.counter = 0
        self.last_val_loss = 12350
        self.early_stop = False

    def __call__(self, val_loss):
        self.epoch = self.epoch + 1
        if self.last_val_loss - val_loss < self.min_delta:
            self.counter +=1
            if self.counter >= self.patience:  
                self.early_stop = True
        else:
            if self.consecutive:    # stopping only on consecutive <patience> number of degradation epochs
                self.counter = 0 
        if self.verbose and self.epoch > 1:
            print(f"Early stop checker: current validation loss: {val_loss:.4f}, last validation loss: {self.last_val_loss:.4f}, delta: {(self.last_val_loss - val_loss):.4f}, min_delta: {self.min_delta:.4f}, hit_n_run-olt torrentek szama: {self.counter} / {self.patience}")
        self.last_val_loss = val_loss
        if self.early_stop:
            print("Early stop condition reached, stopping training")
            return True
        else:
            return False

# early stop on val loss not decreasing for <patience> epochs with more than <min_delta>
early_stop_val_loss = EarlyStopping(
    min_delta=0.001,
    patience=10,
    verbose=True,
    consecutive=False
)

for epoch in range(num_epochs):

    model.train()

    total_loss = 0
    total_correct = 0
    total = 0
    all_labels = []
    all_predictions = []

    for i, (images, _, labels_dict, _) in enumerate(train_loader):
        images = images.to(device)
        labels = labels_dict['class'].to(device)
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
            print ("Training   - Epoch [{}/{}], Step [{}/{}] Loss: {:.4f} in {:.0f}m {:.0f}s"
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item(), time_elapsed // 60, time_elapsed % 60))

    epoch_loss = total_loss / total_step
    epoch_acc = total_correct / total
    # https://scikit-learn.org/0.15/modules/generated/sklearn.metrics.precision_recall_fscore_support.html
    precision, recall, f1_score, _ = precision_recall_fscore_support(all_labels, all_predictions, labels=[0,1], average='weighted')

    # write epoch learning stats to tensorboard & file
    writer.add_scalar("loss/train", loss, epoch)
    writer.add_scalar('accuracy/train', epoch_acc, epoch)
    writer.add_scalar('weighted_precision/train', precision, epoch)
    writer.add_scalar('weighted_recall/train', recall, epoch)
    writer.add_scalar('weighted_f1/train', f1_score, epoch)
    latest_lr = torch.optim.lr_scheduler.MultiStepLR.get_last_lr(scheduler)[-1]
    writer.add_scalar('learning_rate', latest_lr, epoch)

    # show stats at the end of epoch
    print(f"Epoch {epoch+1}/{num_epochs} - Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1_score:.4f}, Last lr: {latest_lr:.4f}")

    # validation
    model.eval()

    val_loss = 0
    val_correct = 0
    val_total = 0
    val_all_labels = []
    val_all_predictions = []

    with torch.no_grad():
        for images, _, labels_dict, _ in val_loader:
            images = images.to(device)
            labels = labels_dict['class'].to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()
            val_all_labels.extend(labels.cpu().numpy())
            val_all_predictions.extend(predicted.cpu().numpy())

    val_epoch_loss = val_loss / len(val_loader)
    val_epoch_acc = val_correct / val_total
    val_precision, val_recall, val_f1_score, _ = precision_recall_fscore_support(val_all_labels, val_all_predictions, labels=[0,1], average='weighted')

    # log validation stats
    writer.add_scalar("loss/val", val_epoch_loss, epoch)
    writer.add_scalar('accuracy/val', val_epoch_acc, epoch)
    writer.add_scalar('weighted_precision/val', val_precision, epoch)
    writer.add_scalar('weighted_recall/val', val_recall, epoch)
    writer.add_scalar('weighted_f1/val', val_f1_score, epoch)

    print(f"Validation - Epoch {epoch+1}/{num_epochs} - Loss: {val_epoch_loss:.4f}, Acc: {val_epoch_acc:.4f}, Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1: {val_f1_score:.4f}")

    # decay learning rate
    scheduler.step()

    # save model checkpoint (epoch)
    if epoch > 2:
        model_file = str(model_checkpoint_dir)+"/"+session_name+str(epoch)+".ckpt"
        torch.save(model.state_dict(), model_file)

    # check early stopping conditions, stop if necessary
    if early_stop_val_loss(val_epoch_loss):
      break

    # end of epoch run (identation!)

# make sure that all pending events have been written to disk.
writer.flush()
writer.close()

# save model checkpoint (final)
dtcomplete = time.strftime("%Y%m%d-%H%M%S")
model_file = str(model_checkpoint_dir)+"/"+session_name+dtcomplete+".ckpt"
torch.save(model.state_dict(), model_file)

# test (can be run with testrunner as well later)
lib.test_model(test_loader, model_file, 'cuda', model, tensorboard_log_dir, session_name=session_name)