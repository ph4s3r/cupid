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
import torch
from pathlib import Path
from coolname import generate_slug # ...
from sklearn.metrics import precision_recall_fscore_support
from torchvision.models import resnet
from torchvision.datasets import PCAM
from torch.utils.tensorboard import SummaryWriter # launch with http://localhost:6006/
from torchvision.transforms import (
    v2,
)  # v2 is the newest: https://pytorch.org/vision/stable/transforms.html

# set h5path directory
h5folder = Path("G:\\pcam\\h5\\")
h5files = list(h5folder.glob("*.h5path"))
model_checkpoint_dir = Path("G:\\pcam\\training_checkpoints\\")
model_checkpoint_dir.mkdir(parents=True, exist_ok=True)
result_path = Path("G:\\pcam\\training_results\\")
result_path.mkdir(parents=True, exist_ok=True)

# instantiate tensorboard summarywriter (write the run's data into random subdir with some funny name)
writer = SummaryWriter(log_dir=f"G:\\pcam\\tensorboard_data\\{generate_slug(2)}\\", comment="pcam_resnet")

transforms = v2.Compose(
    [
        v2.ToImage(),                                           # this operation reshapes the np.ndarray tensor from (3,h,w) to (h,3,w) shape
        v2.ToDtype(torch.float32, scale=True),                  # works only on tensor
    ]
)

pcam_train_dataset = PCAM(root="G:\\pcam\\", transform=transforms, split="train", download=True)
pcam_val_dataset = PCAM(root="G:\\pcam\\", transform=transforms, split="val", download=True)
pcam_test_dataset = PCAM(root="G:\\pcam\\", transform=transforms, split="test", download=True)

print("pcam_train_dataset len: ", len(pcam_train_dataset))
print("pcam_val_dataset len: ", len(pcam_val_dataset))
print("pcam_test_dataset len: ", len(pcam_test_dataset))

batch_size = 64

# num_workers>0 still causes problems...
train_loader = torch.utils.data.DataLoader(
    pcam_train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
)
val_loader = torch.utils.data.DataLoader(
    pcam_val_dataset, batch_size=batch_size, shuffle=True, num_workers=0
)
test_loader = torch.utils.data.DataLoader(
    pcam_test_dataset, batch_size=batch_size, shuffle=True, num_workers=0
)

# init model on the device
ResNet = torch.hub.load(
    "pytorch/vision:v0.10.0", "resnet18", weights=resnet.ResNet18_Weights.IMAGENET1K_V1
)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using {device}")
model = ResNet.to(device)

start_time = time.time()

# hyper-params
num_epochs = 42
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

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.001, verbose=False):

        self.patience = patience
        self.verbose = verbose
        self.min_delta = min_delta
        
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
        if self.verbose and self.epoch > 1:
            print(f"Early stop checker: current validation loss: {val_loss}, last validation loss: {self.last_val_loss}, delta: {self.last_val_loss - val_loss}, min_delta: {self.min_delta}, hit_n_run-olt torrentek szama: {self.counter} / {self.patience}")
        self.last_val_loss = val_loss
        if self.early_stop:
            print("Early stop condition reached, stopping training")
            return True
        else:
            return False

# early stop on val acc not decreasing for <patience> rounds with more than <min_delta>
early_stop_val_loss = EarlyStopping(
    min_delta=0.002,
    patience=4,
    verbose=True
)

for epoch in range(num_epochs):

    model.train()

    total_loss = 0
    total_correct = 0
    total = 0
    all_labels = []
    all_predictions = []

    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
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
    # https://scikit-learn.org/0.15/modules/generated/sklearn.metrics.precision_recall_fscore_support.html
    precision, recall, f1_score, _ = precision_recall_fscore_support(all_labels, all_predictions, labels=[0,1], average='weighted')

    # write epoch learning stats to tensorboard & file
    writer.add_scalar("loss/train", loss, epoch)
    writer.add_scalar('accuracy/train', epoch_acc, epoch)
    writer.add_scalar('weighted_precision/train', precision, epoch)
    writer.add_scalar('weighted_recall/train', recall, epoch)
    writer.add_scalar('weighted_f1/train', f1_score, epoch)

    # show stats at the end of epoch
    print(f"Epoch {epoch+1}/{num_epochs} - Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1_score:.4f}")

    # validation
    model.eval()

    val_loss = 0
    val_correct = 0
    val_total = 0
    val_all_labels = []
    val_all_predictions = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

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
    if (epoch+1) % 20 == 0:
        curr_lr /= 3
        update_lr(optimizer, curr_lr)

    # check early stopping conditions, stop if necessary
    if early_stop_val_loss(val_epoch_loss):
      break

    # end of epoch run (identation!)

# make sure that all pending events have been written to disk.
writer.flush()
writer.close()

# save model checkpoint
dtcomplete = time.strftime("%Y%m%d-%H%M%S")
model_file = str(model_checkpoint_dir)+"\\"+"resnet18-"+dtcomplete+".ckpt"
torch.save(model.state_dict(), model_file)

# test (can be run with testrunner as well later)
lib.test_model(test_loader, model_file, 'cuda')