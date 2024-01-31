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
import dali_raytune_train
import helpers.doublelogger as dl
# pip
import time
import torch
import signal
import logging
from ray import tune
from pathlib import Path
from datetime import datetime
from coolname import generate_slug
from ray.train import Checkpoint
from ray.air import session
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.hyperopt import HyperOptSearch
from sklearn.metrics import precision_recall_fscore_support


from nvidia_resnets.resnet import (
    se_resnext101_32x4d,
)

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


def trainer(config, data_dir = tiles_dir):
    ########################
    # read tiles with DALI #
    ########################
    train_loader, val_loader, dataset_size = dali_raytune_train.dataloaders(data_dir)

    ####################
    # model definition #
    ####################

    model = se_resnext101_32x4d(
        pretrained=True
    )

    ##################
    # begin training #
    ##################

    model.fc = torch.nn.Linear(in_features=2048, out_features=2, bias=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log.info(f"Using {device}")
    # make sure we use cudnn
    # log.info("torch.backends.cudnn.enabled?: ", torch.backends.cudnn.enabled)
    # enable cudnn benchmarks
    # torch.backends.cudnn.benchmark = True
    model = model.to(device)

    # loss and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=config.get('lr'), nesterov=config.get('nesterov'), momentum=0.9)

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

    time_elapsed = time.time() - start_time
    log.info('training prep completed in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    start_time = time.time()
    training_start = datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
    log.info(f"training started at {training_start}")

    for epoch in range(start_epoch, config.get('max_epochs')):

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

            # backward and optimize
            optimizer.zero_grad()

            # forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.step()

            running_loss += loss.item()
            epoch_steps += 1
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print(
                    "[%d, %5d] loss: %.3f"
                    % (epoch + 1, i + 1, running_loss / epoch_steps)
                )
                running_loss = 0.0

            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            total_correct += (predicted == labels).sum().item()
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())     

        epoch_loss = total_loss / total_step
        epoch_acc = total_correct / total
        precision, recall, f1_score, _ = precision_recall_fscore_support(all_labels, all_predictions, labels=[0,1], average='weighted')

        # show stats at the end of epoch
        log.info(f"Training - Epoch {epoch+1}/{config.get('max_epochs')} Steps {i+1} - Loss: {epoch_loss:.6f}, Acc: {epoch_acc:.6f}, Precision: {precision:.6f}, Recall: {recall:.6f}, F1: {f1_score:.6f}")
        
        val_loss = 0
        val_correct = 0
        val_total = 0
        val_all_labels = []
        val_all_predictions = []

        
        for data in val_loader:
            with torch.no_grad():
                images, labels_dict = data[0]['data'], data[0]['label']
                images = images.to(device).to(torch.float16)
                labels = labels_dict.squeeze(-1).long().to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.cpu().numpy()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                val_all_labels.extend(labels.cpu().numpy())
                val_all_predictions.extend(predicted.cpu().numpy())
                val_steps += 1
                

        val_epoch_loss = val_loss / len(val_loader)
        val_epoch_acc = val_correct / val_total
        val_precision, val_recall, val_f1_score, _ = precision_recall_fscore_support(val_all_labels, val_all_predictions, labels=[0,1], average='weighted')

        log.info(f"Validation - Epoch {epoch+1}/{config.get('max_epochs')} - Loss: {val_epoch_loss:.6f}, Acc: {val_epoch_acc:.6f}, Precision: {val_precision:.6f}, Recall: {val_recall:.6f}, F1: {val_f1_score:.6f}")

        checkpoint_data = {
            "epoch": epoch,
            "net_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }
        checkpoint = Checkpoint.from_dict(checkpoint_data)

        session.report(
            {"loss": val_loss / val_steps, "accuracy": val_epoch_acc},
            checkpoint=checkpoint,
        )

        if interrupted:
            log.info(f"KeyboardInterrupt received: saving model for session {tensorboard_session_name} and exiting")
            break
        # end of epoch run (identation!)


def main():

    ########################
    # raytune search space #
    ########################

    # search space
    search_space = {
        "max_epochs": 120,
        "nesterov": tune.choice([True, False]),
        "momentum": tune.uniform(0.1, 0.9),
        "lr": tune.loguniform(1e-4, 1e-1),
        "batch_size": 24,  # It is also ok to specify constant values. othwerwise max is 128 with AMP, 62 without
        "gpus_per_trial": 1
    }

    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=search_space.get("max_epochs"),
        grace_period=1,
        reduction_factor=2,
    )

    ###############
    # raytune run #
    ###############

    hyperopt_search = HyperOptSearch(search_space, metric="mean_accuracy", mode="max")

    tuner = tune.Tuner(
        trainer,
        tune_config=tune.TuneConfig(
            num_samples=10,
            search_alg=hyperopt_search,
        ),
    )

    results = tuner.fit()

    # At each trial, Ray Tune will now randomly sample a combination of parameters from these search spaces. 
    # It will then train a number of models in parallel and find the best performing one among these. 
    # We also use the ASHAScheduler which will terminate bad performing trials early.

    best_trial = results.get_best_trial("loss", "min", "last")
    print(f"Best trial config: {best_trial.config}")
    print(f"Best trial final validation loss: {best_trial.last_result['loss']}")
    print(f"Best trial final validation accuracy: {best_trial.last_result['accuracy']}")


if __name__ == "__main__":
    main()