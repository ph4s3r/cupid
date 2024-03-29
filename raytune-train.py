##########################################################################################################
# Author: Peter Karacsonyi                                                                               #
# Last updated: 2024 feb 2                                                                               #
# Training model                                                                                         #
# Input: training and validation data from a dataloader                                                  #
# Output: raytune experiment, a lot of trials, hopefully the best hyperparams then                       #
##########################################################################################################


# local
import util.utils as utils
# pip & std
import os
import torch
from pathlib import Path
from ray.air import session
from ray import tune, init, train
from coolname import generate_slug
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.hyperopt import HyperOptSearch
from ray.tune.stopper import TrialPlateauStopper
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import precision_recall_fscore_support


#####################
# configure folders #
#####################
base_dir = Path('/mnt/bigdata/datasets/camelyon-pcam')
# tiles_dir = base_dir / Path('tiles')
h5_dir = base_dir / Path('h5')


########################
# static configuration #
########################
static_config = {
    'epochs': 120,
    'batch_size': 128
}


##########################################
# instantiate ray-tune experiment folder #
##########################################
train_session_name = generate_slug(2)
session_dir = base_dir / 'ray_sessions' / train_session_name
session_dir.mkdir(parents=True, exist_ok=True)


################################
# function to save checkpoints #
################################
def save_checkpoint(epoch, model, optimizer, lr_scheduler, session_dir, metrics):
    if train.get_context().get_world_rank() is None or train.get_context().get_world_rank() == 0: # only the no.1 worker manages checkpoints
        checkpoint_data = {
            'epoch': epoch,
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': lr_scheduler.state_dict(),
        }
        if epoch > 4:
            checkpoint_data['model_state'] = model.state_dict()
            checkpoint_file = os.path.join(session_dir, f"{train_session_name}-{session.get_trial_name().split('_')[1]}-{epoch}.ckpt")
            torch.save(checkpoint_data, checkpoint_file)
            print(f'model checkpoint saved as {checkpoint_file}')
    
    session.report(
        metrics=metrics
    )




###############################
# definition of the trainable #
###############################
def trainer(ray_config, static_config=static_config, data_dir=h5_dir):
    
    assert torch.cuda.is_available(), 'GPU is required because of Pytorch-AMP'; device = 'cuda'

    # ########################
    # # read tiles with DALI #
    # ########################
    # import dataloaders.dali_raytune_train
    # train_loader, val_loader, _ = dataloaders.dali_raytune_train.dataloaders(
    #     tiles_dir=data_dir, 
    #     batch_size=ray_config.get('batch_size', static_config.get('batch_size')), 
    #     classlabels=['tumor', 'normal'],
    #     image_size = 256
    # )

    ###########
    # read h5 #
    ###########
    from dataloaders.pcam_h5_dataloader import load_pcam
    train_loader, val_loader, _ = load_pcam(
        dataset_root=data_dir, 
        batch_size=ray_config.get('batch_size', static_config.get('batch_size')), 
        shuffle=False,
        download=False
    )

    # ⭐️⭐️ AMP GradScaler
    scaler = torch.cuda.amp.GradScaler()
    
    ####################
    # model definition #
    ####################
    # from models.nvidia_resnets.resnet import se_resnext101_32x4d
    # model = se_resnext101_32x4d(
    #     pretrained=True
    # )
    # model.fc = torch.nn.Linear(in_features=2048, out_features=2, bias=True)
    # model = model.to(device)

    from torchvision.models import densenet161, DenseNet161_Weights

    model = densenet161(
        weights=DenseNet161_Weights.IMAGENET1K_V1
    )

    model.fc = torch.nn.Linear(in_features=2208, out_features=2, bias=True)
    model = model.to(device)
 

    ######################
    # loss and optimizer #
    ######################
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(), 
        lr=ray_config.get('lr'), 
        nesterov=ray_config.get('nesterov'), 
        momentum=ray_config.get('momentum')
    )

    #############################################
    # ReduceLROnPlateau learning rate scheduler #
    #############################################
    lr_scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        min_lr=1e-4, 
        patience=3,
        eps=1e-7,
        threshold=1e-4
    )

    #####################
    # checkpoint loader #
    #####################
    checkpoint = train.get_checkpoint()
    if checkpoint:
        with checkpoint.as_directory() as checkpoint_dir:
            checkpoint_dict = torch.load(
                os.path.join(utils.find_latest_file(checkpoint_dir, '*.ckpt')),
                )
            if checkpoint_dict.get('model_state', None) is not None:
                start_epoch = checkpoint_dict['epoch'] + 1
                model.load_state_dict(checkpoint_dict.get('model_state', None))
                optimizer.load_state_dict(checkpoint_dict['optimizer_state_dict'])
                lr_scheduler.load_state_dict(checkpoint_dict['scheduler_state_dict'])
    else:
        start_epoch = 0

    ############
    # training #
    ############

    for epoch in range(start_epoch, static_config.get('epochs', 120)):

        train_loss = 0
        train_total = 0
        all_labels = []
        train_correct = 0
        all_predictions = []
        train_batches_processed = 0

        for data in train_loader:
            # images, labels_dict = data[0]['data'], data[0]['label'] # dali-type data
            images, labels_dict = data[0], data[1]          # h5-type data
            images = images.to(device).to(torch.float32)
            labels = labels_dict.squeeze(-1).long().to(device)

            optimizer.zero_grad()

            # ⭐️⭐️ forward pass with AMP autocast
            with torch.autocast(device_type='cuda'):
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            # ⭐️⭐️ backward pass with gradient scaling
            scaler.scale(loss).backward()
            # ⭐️ ⭐️ opt update with scaler
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
            train_batches_processed += 1

        train_loss = train_loss / train_batches_processed
        train_acc = train_correct / train_total
        precision, recall, f1_score, _ = precision_recall_fscore_support(all_labels, all_predictions, labels=[0,1], average='weighted')
        if epoch > 1:
            curr_lr = lr_scheduler.get_last_lr()[0]
        else:
            curr_lr = ray_config.get('lr')
        
        ##############
        # validation #
        ##############
            
        val_loss = 0
        val_total = 0
        val_correct = 0
        val_all_labels = []
        val_all_predictions = []
        val_batches_processed = 0

        for data in val_loader:
            with torch.no_grad():
                # images, labels_dict = data[0]['data'], data[0]['label'] # dali-type data
                images, labels_dict = data[0], data[1]          # h5-type data
                images = images.to(device).to(torch.float32)
                labels = labels_dict.squeeze(-1).long().to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                val_all_labels.extend(labels.cpu().numpy())
                val_all_predictions.extend(predicted.cpu().numpy())
                val_batches_processed += 1
                

        val_epoch_loss = val_loss / val_batches_processed
        val_epoch_acc = val_correct / val_total
        val_precision, val_recall, val_f1_score, _ = precision_recall_fscore_support(val_all_labels, val_all_predictions, labels=[0,1], average='weighted')

        # schedule lr based on val loss
        lr_scheduler.step(val_epoch_loss)

        metrics = {
            'mean_accuracy': train_acc,
            'loss': train_loss,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'val_accuracy': val_epoch_acc,
            'val_loss': val_epoch_loss, 
            'val_precision': val_precision,
            'val_recall': val_recall,
            'val_f1_score': val_f1_score,
            'learning_rate': curr_lr
        }

        # TODO: for some reason it is logging into 2 directories...
        # https://docs.ray.io/en/latest/train/user-guides/persistent-storage.html#persistent-storage-guide
        save_checkpoint(
            epoch, 
            model, 
            optimizer, 
            lr_scheduler, 
            session_dir, 
            metrics
        )
        # end of epoch run (identation!)


def main():

    ########################
    # raytune search space #
    ########################

    ray_search_config = {
        'nesterov': False,
        'momentum': tune.uniform(0.05, 0.6),
        'lr': tune.loguniform(0.005, 0.04),
        'batch_size': tune.qrandint(16, 128, 16),
    }

    scheduler = ASHAScheduler(
        metric='val_accuracy',
        mode='max',
        max_t=static_config.get('epochs'),
        grace_period=1,
        reduction_factor=2,
    )

    current_best_params = [{
        'nesterov': False,
        'momentum': 0.5,
        'lr': 0.02,
        'batch_size': 128,
    }]

    search_alg = HyperOptSearch(
        metric='val_accuracy', 
        mode='max',
        points_to_evaluate=current_best_params
        )
    
    acc_plateau_stopper = TrialPlateauStopper(
        metric='val_accuracy',
        mode='max', # ?
        std=0.005,
        num_results=4,
        grace_period=4,
    )


    #################
    # init ray-tune #
    #################
    init(
        logging_level='info',
        include_dashboard=False
    )

    tuner = tune.Tuner(
        tune.with_resources(
            trainable=trainer, 
            resources={'cpu': 3, 'gpu': 0.20}
        ),
        param_space=ray_search_config,
        run_config=train.RunConfig(
            checkpoint_config=train.CheckpointConfig(num_to_keep=2),
            storage_path=session_dir,
            log_to_file=True,
            stop=acc_plateau_stopper,
            verbose=1,
        ),
        tune_config=tune.TuneConfig(
            num_samples=100,
            search_alg=search_alg,
            scheduler=scheduler,
            max_concurrent_trials=4
        )
    )

    ###############################
    # can resume saved experiment #
    ###############################
    experiment_path = None # '/mnt/bigdata/datasets/camelyon-pcam/ray_sessions/blue-malkoha/trainer_2024-02-09_17-41-08' # path should be where the .pkl file is
    if experiment_path is not None:
        print(f'resuming experiment from {experiment_path}')
        tuner = tune.Tuner.restore(path=experiment_path, trainable=trainer)

    ##################
    # run experiment #
    ##################
    results = tuner.fit()
    best_trial = results.get_best_result(
        metric='val_accuracy', 
        mode='min', 
        scope='all'
    )

    print(f'Best trial selected by val_accuracy: ')
    print(f'config: {best_trial.config}')
    try:
        print(f'path: {best_trial.path}')
        print(f'Best checkpoints: {best_trial.best_checkpoints}') # can get it with get_best_checkpoint
    except:
        pass
    
if __name__ == '__main__':
    main()