##########################################################################################################
# Author: Mihaly Sulyok & Peter Karacsonyi                                                               #
# Last updated: 2024 jan 7                                                                               #
# This workbook continues a training process from a saved checkpoint file                                #
# Input: h5path files, checkpoint file                                                                   #
# Output: trained model & results                                                                        #
##########################################################################################################


# imports
import os
# local files
import lib
if os.name == "nt":
    import helpers.openslideimport  # on windows, openslide needs to be installed manually, check local openslideimport.py
# pip
import time
import torch
import signal
from pathlib import Path
from coolname import generate_slug
from sklearn.metrics import precision_recall_fscore_support
from torch.optim.lr_scheduler import MultiStepLR

from torch.utils.tensorboard import SummaryWriter # launch with http://localhost:6006/


######################################
# which checkpoint we should pick up #
######################################
model_checkpoint = "orchid-ibex7.ckpt"

#################
# training data #
#################
base_dir = Path("/mnt/bigdata/placenta")
h5folder = base_dir / Path("h5-train")


#####################
# configure folders #
#####################
h5files = list(h5folder.glob("*.h5path"))
model_checkpoint_dir = base_dir / Path("training_checkpoints")
result_path = base_dir / Path("training_results")
model_checkpoint_dir.mkdir(parents=True, exist_ok=True)
result_path.mkdir(parents=True, exist_ok=True)


##############################################################################################################
# instantiate tensorboard summarywriter (write the run's data into random subdir with some random funny name)#
##############################################################################################################
session_name = model_checkpoint.split(".")[0][:-1]
print("Continuing session ", session_name)
tensorboard_log_dir = base_dir / "tensorboard_data" / session_name
tensorboard_log_dir.mkdir(parents=True, exist_ok=True)
writer = SummaryWriter(log_dir=tensorboard_log_dir, comment=session_name+"-contd")


############################################$###########
# KeyboardInterrupt: stop training and save checkpoint #
#############################################$$#########
global interrupted
interrupted = False
# signal handler
def signal_handler(signum, frame):
    global interrupted
    interrupted = True
    print("Interrupt received, stopping...")
# attach signal handler
signal.signal(signal.SIGINT, signal_handler)

def save_model(epoch, model, optimizer, scheduler, checkpoint_file):
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            }, checkpoint_file)
        print(f"Model successfully saved to file {checkpoint_file}")

#########################################################################
# inserting tiles from h5path with TransformedPathmlTileSet to datasets #
#########################################################################
datasets = []
ds_fullsize = 0
for h5file in h5files:
    print(f"creating dataset from {str(h5file)} with TransformedPathmlTileSet")
    datasets.append(lib.TransformedPathmlTileSet(h5file))

for ds in datasets:
    ds_fullsize += ds.dataset_len

full_ds = torch.utils.data.ConcatDataset(datasets)


######################
# set up dataloaders #
######################
batch_size = 88 # need to max the batch out by seeing how much memory it takes (nvitop!!)
# however smaller batch sizes can sometimes provide better generalization
# fixed generator for reproducible split results
generator = torch.Generator().manual_seed(42)
train_cases, val_cases = torch.utils.data.random_split( # split to 70% train, 30% val
    full_ds, [0.7, 0.3], generator=generator
)

train_loader = torch.utils.data.DataLoader(
    train_cases, batch_size=batch_size, shuffle=True, num_workers=8
)
val_loader = torch.utils.data.DataLoader(
    val_cases, batch_size=batch_size, shuffle=True, num_workers=8
)
print(f"after filtering the dataset for usable tiles, we have left with {len(train_cases) + len(val_cases)} tiles from the original {ds_fullsize}")


#####################
# continue training #
#####################
model_checkpoint_file = base_dir / "training_checkpoints" / model_checkpoint
checkpoint = torch.load(model_checkpoint_file)

SE_RESNET50 = torch.hub.load(
    'moskomule/senet.pytorch',
    'se_resnet50',
    pretrained=True,
    verbose=True
    )

num_ftrs = SE_RESNET50.fc.in_features
SE_RESNET50.fc = torch.nn.Linear(num_ftrs, 2)
# loading the saved model
SE_RESNET50.load_state_dict(checkpoint["model_state_dict"])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using {device}")
model = SE_RESNET50.to(device)

start_time = time.time()

# hyper-params
num_epochs = 20 + int(checkpoint["epoch"])
learning_rate = 0.001

# loss and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

# to update learning rate
scheduler = MultiStepLR(optimizer, milestones=[], gamma=1.0)
scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

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

for epoch in range(int(checkpoint["epoch"]), num_epochs):
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

    # save model checkpoint and data (epoch)
    if epoch > 2 or interrupted:
        checkpoint_file = str(model_checkpoint_dir)+"/"+session_name+str(epoch)+".ckpt"
        save_model(epoch, model, optimizer, scheduler, checkpoint_file)

    if interrupted:
        print(f"KeyboardInterrupt received: saving model for session {session_name} and exiting")
        break

    # check early stopping conditions, stop if necessary
    if early_stop_val_loss(val_epoch_loss):
        break
    # end of epoch run (identation!)

writer.flush()
writer.close()