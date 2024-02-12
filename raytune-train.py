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
import tempfile
from pathlib import Path
from ray.train import Checkpoint
from ray import tune, init, train
from coolname import generate_slug
from ray.tune.schedulers import ASHAScheduler
from ray.tune.stopper import TrialPlateauStopper
from ray.tune.search.hyperopt import HyperOptSearch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import precision_recall_fscore_support


#####################
# configure folders #
#####################
base_dir = Path('/mnt/bigdata/datasets/camelyon-pcam')
# tiles_dir = base_dir / Path('tiles')
h5_dir = base_dir / Path('h5')
ray_dir  = base_dir / Path('ray_sessions')


########################
# static configuration #
########################
static_config = {
    'epochs': 120,
    'batch_size': 48
}


##################################
# function to save checkpoints   #
# saves into the trial directory #
##################################
def save_checkpoint(epoch, model, optimizer, lr_scheduler, metrics):
    if train.get_context().get_world_rank() is None or train.get_context().get_world_rank() == 0: # only the no.1 worker manages checkpoints
        if epoch > 4:
            checkpoint_data = {
                'epoch': epoch,
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': lr_scheduler.state_dict(),
                'model_state' : model.state_dict()
            }
            with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
                    torch.save(
                        checkpoint_data,
                        os.path.join(temp_checkpoint_dir, f"model-{epoch}.pt"),
                    )
                    checkpoint = Checkpoint.from_directory(temp_checkpoint_dir)
                    train.report(metrics, checkpoint=checkpoint)


###############################
# definition of the trainable #
###############################
def trainer(ray_config, static_config=static_config, data_dir=h5_dir):
    
    assert torch.cuda.is_available(), 'GPU is required because of Pytorch-AMP'; device = torch.device('cuda')

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
        shuffle=True
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
    # model.to(device)


    ####################
    # model definition #
    ####################
    from torchvision.models import densenet161, DenseNet161_Weights
    model = densenet161(
        weights=DenseNet161_Weights.IMAGENET1K_V1
    )
    model.fc = torch.nn.Linear(in_features=2208, out_features=2, bias=True)
    model.to(device)


    ####################
    # loss & optimizer #
    ####################
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(), 
        lr=ray_config.get('lr'), 
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


    ############################
    # manual checkpoint loader #
    ############################
    if 0:
        model_checkpoint_path = '/mnt/bigdata/datasets/camelyon-pcam/ray_sessions/independent-dazzling-chameleon-of-reward/observant-guan_0_2024-02-12_21-53-42/checkpoint_000001/model-6.pt'
        checkpoint_dict = torch.load(model_checkpoint_path)
        model.load_state_dict(checkpoint_dict['model_state'])
        optimizer.load_state_dict(checkpoint_dict['optimizer_state_dict'])
        lr_scheduler.load_state_dict(checkpoint_dict['scheduler_state_dict'])
        print(f'Loaded checkpoint from {model_checkpoint_path} successfully!')
 


    ############################################################################
    # ray checkpoint loader: will load the latest ckpt from training directory #
    # unfortunately it does not work due to no GPU is available after restore  #
    ############################################################################

    if train.get_checkpoint():
        loaded_checkpoint = train.get_checkpoint()
        with loaded_checkpoint.as_directory() as loaded_checkpoint_dir:
            latest_checkpoint_file: str | None = utils.find_latest_file(loaded_checkpoint_dir, '*.pt')
            assert latest_checkpoint_file, f"checkpoint file not found in {loaded_checkpoint_dir}"
            checkpoint_dict = torch.load(latest_checkpoint_file)
            start_epoch = checkpoint_dict['epoch'] + 1
            model.load_state_dict(checkpoint_dict.get('model_state', None))
            optimizer.load_state_dict(checkpoint_dict['optimizer_state_dict'])
            lr_scheduler.load_state_dict(checkpoint_dict['scheduler_state_dict'])
            print(f'latest checkpoint {latest_checkpoint_file} successfully loaded!')
    else:
        # print(f'checkpoint not found, starting from scratch (train.get_checkpoint() = {train.get_checkpoint()})')
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
            metrics
        )
        # end of epoch run (identation!)




def main():


    ########################
    # raytune search space #
    ########################

    ray_search_config = {
        'momentum': 0.1,
        'lr': 0.02,
        'batch_size': 512
    }

    scheduler = ASHAScheduler(
        metric='val_accuracy',
        mode='max',
        max_t=static_config.get('epochs'),
        grace_period=4,
        reduction_factor=2,
    )

    current_best_params = [{
        'momentum': 0.1,
        'lr': 0.02,
        'batch_size': 48,
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

    def trial_str_creator(trial):
        return generate_slug(2)


    #################
    # init ray-tune #
    #################
    init(
        logging_level='info',
        include_dashboard=False
    )

    #######################################################################################
    # can resume saved experiment (does not work unfortunately due to GPUs not available) #
    #######################################################################################
    resume = False
    if resume:
        experiment_path = '/mnt/bigdata/datasets/camelyon-pcam/ray_sessions/realistic-keen-gorilla-of-merriment' # path should be where the .pkl file is
        assert tune.Tuner.can_restore(experiment_path), f'FATAL: experiment cannot be restored from {experiment_path}'
        tuner = tune.Tuner.restore(
            trainable=trainer,
            param_space=ray_search_config,
            path=experiment_path,
            )
        print(f'resuming experiment from {experiment_path} ', tuner.get_results())
        print("")
    else:
        tuner = tune.Tuner(
            tune.with_resources(
                trainable=trainer,
                resources={'cpu': 16, 'gpu': 1}
            ),
            param_space=ray_search_config,
            run_config=train.RunConfig(
                checkpoint_config=train.CheckpointConfig(
                    num_to_keep=2,
                    checkpoint_score_attribute='val_accuracy',
                    checkpoint_score_order='max'
                    ),
                storage_path=ray_dir,
                log_to_file=True,
                # stop=acc_plateau_stopper,
                verbose=1,
                name=generate_slug()
            ),
            tune_config=tune.TuneConfig(
                num_samples=1,
                # search_alg=search_alg,
                scheduler=scheduler,
                max_concurrent_trials=1,
                trial_name_creator=trial_str_creator
            )
        )

    ##################
    # run experiment #
    ##################
    results = tuner.fit()
    best_trial = results.get_best_result(
        metric='val_accuracy', 
        mode='max', 
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