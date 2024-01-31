##########################################################################################################
# Author: Mihaly Sulyok & Peter Karacsonyi                                                               #
# Last updated: 2024 jan 17                                                                              #
# Training model                                                                                         #
# Input: directory containing wsi-named directories with the jpg tile files                              #
# Output: model weights, optimizer, metrics (tensorboard)                                                #
##########################################################################################################


# imports
import os
if os.name == "nt":
    import helpers.openslideimport  # on windows, openslide needs to be installed manually, check local openslideimport.py
# local
import dali_raytune_train
import helpers.doublelogger as dl
# pip
import torch
import signal
import tempfile
from ray import tune, init
from pathlib import Path
from ray.air import session
from ray.train import RunConfig
from coolname import generate_slug
from ray.train import Checkpoint, get_context
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
tiles_dir = base_dir / Path("tiles-train-500")

############################################################################################################
# instantiate ray-tune session folder (write the run's data into random subdir with some random funny name)#
############################################################################################################
train_session_name = generate_slug(2)
session_dir = base_dir / "ray_sessions" / train_session_name
session_dir.mkdir(parents=True, exist_ok=True)


####################################
# KeyboardInterrupt: stop training #
####################################
global interrupted # global KeyboardInterrupt Flag
interrupted = False
def signal_handler(signum, frame):
    global interrupted
    interrupted = True
    print("Interrupt received, stopping...")
signal.signal(signal.SIGINT, signal_handler) # attach


def save_checkpoint(epoch, model, optimizer, session_dir, metrics):
    with tempfile.TemporaryDirectory(dir=str(session_dir)) as tempdir:
        if get_context().get_world_rank() == 0: # make sure only the no.1 worker manages the checkpoint
            torch.save(
                    {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                os.path.join(tempdir, "checkpoint.pt"),
            )
        session.report(
            metrics=metrics,                              # accuracy etc..
            checkpoint=Checkpoint.from_directory(tempdir) # creating a Checkpoint from tempdir
        )


###############################
# definition of the trainable #
###############################
def trainer(config, data_dir=tiles_dir):


    ########################
    # read tiles with DALI #
    ########################
    train_loader, val_loader, dataset_size = dali_raytune_train.dataloaders(tiles_dir=data_dir, batch_size=config.get('batch_size'))


    ####################
    # model definition #
    ####################
    model = se_resnext101_32x4d(
        pretrained=True
    )
    model.fc = torch.nn.Linear(in_features=2048, out_features=2, bias=True)


    ###################
    # model to device #
    ###################
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using {device}")
    model = model.to(device)

    ######################
    # loss and optimizer #
    ######################
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=config.get('lr'), nesterov=config.get('nesterov'), momentum=config.get('momentum'))

    #########################
    # raytune checkpointing #
    #########################
    # https://docs.ray.io/en/latest/train/user-guides/checkpoints.html
    checkpoint = session.get_checkpoint()

    if checkpoint:
        with checkpoint.as_directory() as checkpoint_dir:
            checkpoint_dict = torch.load(
                os.path.join(checkpoint_dir, "checkpoint.pt")
                )
            start_epoch = checkpoint_dict["epoch"] + 1
            model.load_state_dict(checkpoint_dict["model_state"])
            optimizer.load_state_dict(checkpoint_dict["optimizer_state_dict"])
    else:
        start_epoch = 0
    
    # train
    total_step = dataset_size # full training dataset len
    val_steps = 0

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
            images = images.to(device).to(torch.float32)
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
        print(f"Training - Epoch {epoch+1}/{config.get('max_epochs')} Steps {i+1} - Loss: {epoch_loss:.6f}, Acc: {epoch_acc:.6f}, Precision: {precision:.6f}, Recall: {recall:.6f}, F1: {f1_score:.6f}")
        
        val_loss = 0
        val_correct = 0
        val_total = 0
        val_all_labels = []
        val_all_predictions = []

        
        for data in val_loader:
            with torch.no_grad():
                images, labels_dict = data[0]['data'], data[0]['label']
                images = images.to(device).to(torch.float32)
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

        print(f"Validation - Epoch {epoch+1}/{config.get('max_epochs')} - Loss: {val_epoch_loss:.6f}, Acc: {val_epoch_acc:.6f}, Precision: {val_precision:.6f}, Recall: {val_recall:.6f}, F1: {val_f1_score:.6f}")

        metrics = {
            "mean_accuracy": epoch_acc,
            "loss": epoch_loss,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "val_accuracy": val_epoch_acc,
            "val_loss": val_loss / val_steps, 
            "val_precision": val_precision,
            "val_recall": val_recall,
            "val_f1_score": val_f1_score
        }

        # TODO: persistent dir not really working so needed to be done
        # https://docs.ray.io/en/latest/train/user-guides/persistent-storage.html#persistent-storage-guide
        save_checkpoint(epoch, model, optimizer, session_dir, metrics)

        if interrupted:
            print(f"KeyboardInterrupt received: quitting training session(s)")
            save_checkpoint(epoch, model, optimizer, session_dir, metrics)
            break
        # end of epoch run (identation!)


def main():

    ########################
    # raytune search space #
    ########################
    ray_search_config = {
        "max_epochs": 120,
        "nesterov": tune.choice([True, False]),
        "momentum": tune.uniform(0.8, 0.95),
        "lr": tune.loguniform(0.03, 0.04),
        "batch_size": 36
    }
    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=ray_search_config.get("max_epochs"),
        grace_period=1,
        reduction_factor=2,
    )


    ######################
    # raytune init & run #
    ######################
    init(
        resources={"cpu": 16, "gpu": 1},
        logging_level='info',
        include_dashboard=True
        )
    tuner = tune.Tuner(
        tune.with_resources(
            trainable=trainer, 
            resources={"cpu": 16, "gpu": 1}),
            param_space=ray_search_config,
            run_config=RunConfig(
                storage_path=session_dir, 
                name=train_session_name,
                log_to_file=True
                ),
            tune_config=tune.TuneConfig(
                num_samples=10,
                search_alg=HyperOptSearch(metric="mean_accuracy", mode="max"),
                scheduler=scheduler,
                max_concurrent_trials=1
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