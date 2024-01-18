##########################################################################################################
# Author: Mihaly Sulyok & Peter Karacsonyi                                                               #
# Last updated: 2024 jan 17                                                                              #
# Training model                                                                                         #
# Input: h5path files                                                                                    #
# Output: trained model, optimizer, scheduler, epoch state, tensorboard data                             #
##########################################################################################################


# imports
import os
# local files
import dali
if os.name == "nt":
    import helpers.openslideimport  # on windows, openslide needs to be installed manually, check local openslideimport.py
import helpers.ds_means_stds
import lib
# pip
import time
import torch
import signal
from apex import amp
from pathlib import Path
from datetime import datetime
from coolname import generate_slug
from sklearn.metrics import precision_recall_fscore_support
from torch.optim.lr_scheduler import MultiStepLR
from nvidia.dali.plugin.pytorch import DALIGenericIterator
from nvidia.dali.plugin.base_iterator import LastBatchPolicy

from nvidia_resnets.resnet import (
    se_resnext101_32x4d,
)

from torch.utils.tensorboard import SummaryWriter # launch with http://localhost:6006/


start_time = time.time()
print(f"training prep started at {datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')}")

#####################
# configure folders #
#####################
base_dir = Path("/mnt/bigdata/placenta")
h5folder = base_dir / Path("h5-medium")
h5files = list(h5folder.glob("*.h5path"))
model_checkpoint_dir = base_dir / Path("training_checkpoints")
model_checkpoint_dir.mkdir(parents=True, exist_ok=True)
tiles_dir = base_dir / Path("tiles")


##############################################################################################################
# instantiate tensorboard summarywriter (write the run's data into random subdir with some random funny name)#
##############################################################################################################
session_name = generate_slug(2)
print("Starting session ", session_name)
tensorboard_log_dir = base_dir / "tensorboard_data" / session_name
tensorboard_log_dir.mkdir(parents=True, exist_ok=True)
writer = SummaryWriter(log_dir=tensorboard_log_dir, comment=session_name)


############################################$###########
# KeyboardInterrupt: stop training and save checkpoint #
#############################################$$#########

# global KeyboardInterrupt Flag
global interrupted
interrupted = False
# signal handler
def signal_handler(signum, frame):
    global interrupted
    interrupted = True
    print("Interrupt received, stopping...")
# attach signal handler
signal.signal(signal.SIGINT, signal_handler)

def save_model(epoch, model, optimizer, scheduler, amp, checkpoint_file):
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'amp': amp.state_dict()
            }, checkpoint_file)
        print(f"Model successfully saved to file {checkpoint_file}")


########################
# read tiles with DALI #
########################

batch_size = 128 # need to max the batch out by seeing how much memory it takes (nvitop!!)
# we make a 80-20 split by using 5 shards (splitting the images to 5 batches: each shard number refers to 20% of the data)
num_shards = 5
train_shard_ids = list(range(4))  # Shards 0-3 for training
val_shard_id = 4  # Shard 4 for validation

# train and val pipeline and iterator
train_pipelines = [dali.cpupipe(
    tiles_dir, 
    shard_id=i, 
    num_shards=num_shards, 
    num_threads=16, 
    device_id=0, 
    batch_size=batch_size
) for i in train_shard_ids]

val_pipeline = dali.cpupipe(
    tiles_dir, 
    shard_id=val_shard_id, 
    num_shards=num_shards, 
    num_threads=16, 
    device_id=0, 
    batch_size=batch_size
)

# loaders

train_data = DALIGenericIterator(
    train_pipelines,
    ['data', 'label'],
    reader_name='Reader',
    last_batch_policy=LastBatchPolicy.DROP
)

val_data = DALIGenericIterator(
    [val_pipeline],
    ['data', 'label'],
    reader_name='Reader',
    last_batch_policy=LastBatchPolicy.DROP
)

# https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Classification/ConvNets/se-resnext101-32x4d
# se-resnext101-32x4d paper: https://arxiv.org/pdf/1611.05431.pdf
# torchhub: https://pytorch.org/hub/nvidia_deeplearningexamples_se-resnext/

model = se_resnext101_32x4d(
    pretrained=True
)
# default input image dim: 224

##################
# begin training #
##################

num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(in_features=2048, out_features=2, bias=True)

#####################################
# can load saved weights and biases #
#####################################
checkpoint = None
if 0:
    checkpoint = torch.load(base_dir / "training_checkpoints" / "")
    model.load_state_dict(checkpoint["model_state_dict"])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using {device}")
# make sure we use cudnn
print("torch.backends.cudnn.enabled?: ", torch.backends.cudnn.enabled)
# enable cudnn benchmarks
torch.backends.cudnn.benchmark = True
model = model.to(device)

# hyper-params
num_epochs = 120

# loss and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model.parameters(), lr=0.1 / 1024 * batch_size, momentum=0.9, weight_decay=1e-4)

# Automatic Mixed Precision (AMP) https://github.com/NVIDIA/apex
model, optimizer = amp.initialize(model, optimizer, opt_level="O2", loss_scale="dynamic")
if checkpoint is not None:
    if checkpoint.get('amp') is not None:
        amp.load_state_dict(checkpoint['amp'])

# to update learning rate
# scheduler = StepLR(optimizer, step_size=7, gamma=0.1, verbose=True)
scheduler = MultiStepLR(optimizer=optimizer, milestones=[10,30], gamma=0.1, verbose=True)

# train
total_step = len(train_data)

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

time_elapsed = time.time() - start_time
print('training prep completed in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
start_time = time.time()
print(f"training started at {datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')}")


for epoch in range(num_epochs):

    epoch_start_time = time.time()
    model.train()

    total_loss = 0
    total_correct = 0
    total = 0
    all_labels = []
    all_predictions = []

    for i, data in enumerate(train_data):
        images, labels_dict = data[0]['data'], data[0]['label']
        images = images.to(device).to(torch.float16)
        labels = labels_dict.squeeze(-1).long().to(device)
        # forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # backward and optimize
        optimizer.zero_grad()

        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
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
        for data in val_data:
            images, labels_dict = data[0]['data'], data[0]['label']
            images = images.to(device).to(torch.float16)
            labels = labels_dict.squeeze(-1).long().to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()
            val_all_labels.extend(labels.cpu().numpy())
            val_all_predictions.extend(predicted.cpu().numpy())

    val_epoch_loss = val_loss / len(val_data)
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
        save_model(epoch, model, optimizer, scheduler, amp, checkpoint_file)

    if interrupted:
        print(f"KeyboardInterrupt received: saving model for session {session_name} and exiting")
        break

    # check early stopping conditions, stop if necessary
    if early_stop_val_loss(val_epoch_loss):
        break
    
    epoch_complete = time.time() - epoch_start_time
    print('epoch completed in {:.0f}m {:.0f}s'.format(epoch_complete // 60, epoch_complete % 60))
    # end of epoch run (identation!)

writer.flush()
writer.close()