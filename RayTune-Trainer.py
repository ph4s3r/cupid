##########################################################################################################
# Author: Mihaly Sulyok & Peter Karacsonyi                                                               #
# Last updated: 2024 jan 17                                                                              #
# Training model                                                                                         #
# Input: h5path files                                                                                    #
# Output: trained model, optimizer, scheduler, epoch state, tensorboard data                             #
##########################################################################################################


# imports
import os
if os.name == "nt":
    import helpers.openslideimport  # on windows, openslide needs to be installed manually, check local openslideimport.py
# local
import lib
import timedinput
import dali_raytune_train as dali
import helpers.doublelogger as dl
# pip
import time
import torch
import signal
import logging
from apex import amp
from ray import tune
from pathlib import Path
from functools import partial
from datetime import datetime
from coolname import generate_slug
from ray.air import Checkpoint, session
from ray.tune.schedulers import ASHAScheduler
from sklearn.metrics import precision_recall_fscore_support


from nvidia_resnets.resnet import (
    se_resnext101_32x4d,
)

from torch.utils.tensorboard import SummaryWriter # launch with http://localhost:6006/

##################
# raytune config #
##################

gpus_per_trial = 1

config = {
    "num_epochs": 120,
    "nesterov": tune.choice([True, False]),
    "momentum": tune.uniform(0.1, 0.9),
    "lr": tune.loguniform(1e-4, 1e-1),
    "batch_size": 128,  # It is also ok to specify constant values. othwerwise max is 128 with AMP, 62 without
    "gpus_per_trial": 1
}


#####################
# configure folders #
#####################
base_dir = Path("/mnt/bigdata/placenta")
model_checkpoint_dir = base_dir / Path("training_checkpoints")
model_checkpoint_dir.mkdir(parents=True, exist_ok=True)
tiles_dir = base_dir / Path("tiles-training")

##############################################################################################################
# instantiate tensorboard summarywriter (write the run's data into random subdir with some random funny name)#
##############################################################################################################
tensorboard_session_name = generate_slug(2)
tensorboard_log_dir = base_dir / "tensorboard_data" / tensorboard_session_name
# user can customize the tensorboard folder
user_input = timedinput.timed_input("Any comment to add to the session (will be appended to the tensorboard folder)? : ")
user_input = timedinput.sanitize(user_input)
print(f"Adding comment to tensorboard data: {user_input}")
if user_input is not None:
    if user_input != '':
        tensorboard_session_name  = tensorboard_session_name + "-" + user_input
        tensorboard_log_dir = base_dir / "tensorboard_data" / Path(tensorboard_session_name)

tensorboard_log_dir.mkdir(parents=True, exist_ok=True)
writer = SummaryWriter(log_dir=tensorboard_log_dir)


#############################################
# double logging to stdout and file as well #
#############################################
dl.setLogger(tensorboard_log_dir / Path("logs.txt"))
log = logging.getLogger("spl")
start_time = time.time()
log.info(f"tensorboard session {tensorboard_session_name} started at {datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')}")

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
    log.info("Interrupt received, stopping...")
# attach signal handler
signal.signal(signal.SIGINT, signal_handler)

# def save_model(epoch, model, optimizer, scheduler, amp, checkpoint_file):
#         torch.save({
#             'epoch': epoch,
#             'model_state_dict': model.state_dict(),
#             'optimizer_state_dict': optimizer.state_dict(),
#             'scheduler_state_dict': scheduler.state_dict(),
#             'amp': amp.state_dict()
#             }, checkpoint_file)
#         log.info(f"Model successfully saved to file {checkpoint_file}")

# to update learning rate
scheduler = ASHAScheduler(
    metric="loss",
    mode="min",
    max_t=config.get("num_epochs"),
    grace_period=1,
    reduction_factor=2,
)


def trainer(data_dir):
    ########################
    # read tiles with DALI #
    ########################
    train_loader, val_loader, dataset_size = dali.dataloaders(data_dir)

    ####################
    # model definition #
    ####################

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
    # checkpoint = None
    # if 0:
    #     checkpoint = torch.load(base_dir / "training_checkpoints" / "")
    #     model.load_state_dict(checkpoint["model_state_dict"])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log.info(f"Using {device}")
    # make sure we use cudnn
    log.info("torch.backends.cudnn.enabled?: ", torch.backends.cudnn.enabled)
    # enable cudnn benchmarks
    torch.backends.cudnn.benchmark = True
    model = model.to(device)

    # loss and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=config.get('lr'), nesterov=config.get('nesterov'), momentum=0.9)

    # Automatic Mixed Precision (AMP) https://github.com/NVIDIA/apex deprecated...
    # TODO: change to https://pytorch.org/docs/stable/amp.html
    model, optimizer = amp.initialize(
            model, 
            optimizer,
            opt_level="O1", # 01: Mixed precision
            loss_scale="dynamic",
            # just all the defaults for 01 
            cast_model_type=None,
            patch_torch_functions=True,
            keep_batchnorm_fp32=None,
            master_weights=None,
        )
    # if checkpoint is not None:
    #     if checkpoint.get('amp') is not None:
    #         amp.load_state_dict(checkpoint['amp'])

    #########################
    # raytune checkpointing #
    #########################

    checkpoint = session.get_checkpoint()

    if checkpoint:
        checkpoint_state = checkpoint.to_dict()
        start_epoch = checkpoint_state["epoch"]
        model.load_state_dict(checkpoint_state["net_state_dict"])
        optimizer.load_state_dict(checkpoint_state["optimizer_state_dict"])
    else:
        start_epoch = 0

    # train
    total_step = dataset_size # full training dataset len
    val_steps = 0


    # early stop on val loss not decreasing for <patience> epochs with more than <min_delta>
    early_stop_val_loss = lib.EarlyStopping(
        min_delta=0.001,
        patience=15,
        verbose=True,
        consecutive=False
    )

    time_elapsed = time.time() - start_time
    log.info('training prep completed in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    start_time = time.time()
    training_start = datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
    log.info(f"training started at {training_start}")

    hyperparams_tensorboard = {
        "model": "se_resnext101_32x4d",
        "scheduler_type": str(scheduler), 
        "scheduler.step_size": str(scheduler.step_size),
        "scheduler.gamma": str(scheduler.gamma),
        "optimizer": str(optimizer),
        "amp._amp_state.opt_properties.options": str(amp._amp_state.opt_properties.options),
        "batch_size": str(config.get('batch_size')),
        "training_started": str(training_start)
    }

    writer.add_hparams(hyperparams_tensorboard, {})
    writer.add_text("hyperparameters", lib.pretty_json(hyperparams_tensorboard))
    writer.add_text("comment", user_input)

    for epoch in range(config.get('num_epochs')):

        epoch_start_time = time.time()

        total_loss = 0
        total_correct = 0
        total = 0
        all_labels = []
        all_predictions = []

        for i, data in enumerate(train_loader):
            running_loss = 0.0
            epoch_steps = 0

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

        epoch_loss = total_loss / total_step
        epoch_acc = total_correct / total
        # https://scikit-learn.org/0.15/modules/generated/sklearn.metrics.precision_recall_fscore_support.html
        precision, recall, f1_score, _ = precision_recall_fscore_support(all_labels, all_predictions, labels=[0,1], average='weighted')

        # write epoch learning stats to tensorboard & file
        writer.add_scalar("loss/train", loss, epoch)
        writer.add_scalar('acc/train', epoch_acc, epoch)
        writer.add_scalar('weighted_precision/train', precision, epoch)
        writer.add_scalar('weighted_recall/train', recall, epoch)
        writer.add_scalar('weighted_f1/train', f1_score, epoch)
        latest_lr = torch.optim.lr_scheduler.MultiStepLR.get_last_lr(scheduler)[-1]
        writer.add_scalar('params/learning_rate', latest_lr, epoch)

        # show stats at the end of epoch
        log.info(f"Training - Epoch {epoch+1}/{config.get('num_epochs')} Steps {i+1} - Loss: {epoch_loss:.6f}, Acc: {epoch_acc:.6f}, Precision: {precision:.6f}, Recall: {recall:.6f}, F1: {f1_score:.6f}, Last lr: {latest_lr:.8f}")
        
        running_loss += loss.item()
        epoch_steps += 1
        if i % 2000 == 1999:  # print every 2000 mini-batches
            print(
                "[%d, %5d] loss: %.3f"
                % (epoch + 1, i + 1, running_loss / epoch_steps)
            )
            running_loss = 0.0
        
        val_loss = 0
        val_correct = 0
        val_total = 0
        val_all_labels = []
        val_all_predictions = []

        with torch.no_grad():
            for data in val_loader:
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
                val_steps += 1

        val_epoch_loss = val_loss / len(val_loader)
        val_epoch_acc = val_correct / val_total
        val_precision, val_recall, val_f1_score, _ = precision_recall_fscore_support(val_all_labels, val_all_predictions, labels=[0,1], average='weighted')

        train_val_acc_diff = epoch_acc - val_epoch_acc
        train_val_loss_diff = val_loss - loss 

        # log validation stats
        writer.add_scalar("loss/val", val_epoch_loss, epoch)
        writer.add_scalar('acc/val', val_epoch_acc, epoch)
        writer.add_scalar("acc/train-val-diff", train_val_acc_diff, epoch)
        writer.add_scalar("loss/val_loss-loss-diff", train_val_loss_diff, epoch)
        writer.add_scalar('weighted_precision/val', val_precision, epoch)
        writer.add_scalar('weighted_recall/val', val_recall, epoch)
        writer.add_scalar('weighted_f1/val', val_f1_score, epoch)

        log.info(f"Validation - Epoch {epoch+1}/{config.get('num_epochs')} - Loss: {val_epoch_loss:.6f}, Acc: {val_epoch_acc:.6f}, Precision: {val_precision:.6f}, Recall: {val_recall:.6f}, F1: {val_f1_score:.6f}")

        # decay learning rate
        scheduler.step()

        # save model checkpoint and data (epoch)
        if epoch > 5 or interrupted:
            checkpoint_file = str(model_checkpoint_dir)+"/"+tensorboard_session_name+str(epoch)+".ckpt"

        checkpoint_data = {
            "epoch": epoch,
            "net_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }
        checkpoint = Checkpoint.from_dict(checkpoint_data)

        session.report(
            {"loss": val_loss / val_steps, "accuracy": epoch_acc},
            checkpoint=checkpoint,
        )

        if interrupted:
            log.info(f"KeyboardInterrupt received: saving model for session {tensorboard_session_name} and exiting")
            break

        epoch_complete = time.time() - epoch_start_time
        # end of epoch run (identation!)
        writer.flush()
        # check early stopping conditions, stop if necessary
        if early_stop_val_loss(val_epoch_loss):
            break



result = tune.run(
    partial(trainer, data_dir=tiles_dir),
    resources_per_trial={"cpu": 8, "gpu": gpus_per_trial},
    config=config,
    num_samples=10, # 1
    scheduler=scheduler,
    checkpoint_at_end=True)

# 1
# At each trial, Ray Tune will now randomly sample a combination of parameters from these search spaces. 
# It will then train a number of models in parallel and find the best performing one among these. 
# We also use the ASHAScheduler which will terminate bad performing trials early.

best_trial = result.get_best_trial("loss", "min", "last")
print(f"Best trial config: {best_trial.config}")
print(f"Best trial final validation loss: {best_trial.last_result['loss']}")
print(f"Best trial final validation accuracy: {best_trial.last_result['accuracy']}")

best_trained_model = model(best_trial.config["l1"], best_trial.config["l2"])
device = "cpu"
if torch.cuda.is_available():
    device = "cuda:0"
    if gpus_per_trial > 1:
        best_trained_model = nn.DataParallel(best_trained_model)
best_trained_model.to(device)

best_checkpoint = best_trial.checkpoint.to_air_checkpoint()
best_checkpoint_data = best_checkpoint.to_dict()

best_trained_model.load_state_dict(best_checkpoint_data["net_state_dict"])

writer.close()