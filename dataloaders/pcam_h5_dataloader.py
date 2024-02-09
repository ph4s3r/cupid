from torchvision.datasets import PCAM
from torchvision.transforms import v2
import torch

t_ToTensor = v2.Compose(
    [
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True)
    ]
)

t_AUG = v2.Compose(
    [
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.RandomApply(
            transforms=[
                v2.RandomRotation(degrees=(0, 359)), 
                v2.ColorJitter(brightness=.5, hue=.3, saturation=.8, contrast=.5)
            ]
        , p=1)  
    ]
)

def load_pcam(dataset_root, batch_size = 128, shuffle = True, download = True):

    pcam_train_ds_normal = PCAM(root=dataset_root, transform=t_ToTensor, split='train', download=True)
    pcam_train_ds_aug = PCAM(root=dataset_root, transform=t_AUG, split='train', download=True)
    pcam_train_dataset = pcam_train_ds_normal + pcam_train_ds_aug
    pcam_val_dataset = PCAM(root=dataset_root, transform=t_ToTensor, split='val', download=True)

    print('pcam_train_ds_normal len: ', len(pcam_train_ds_normal))
    print('pcam_train_ds_aug len: ', len(pcam_train_ds_aug))
    print('pcam_train_dataset len: ', len(pcam_train_dataset))
    print('pcam_val_dataset len: ', len(pcam_val_dataset))

    dataset_size = len(pcam_train_dataset) + len(pcam_val_dataset)

    train_loader = torch.utils.data.DataLoader(
        pcam_train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=16
    )
    val_loader = torch.utils.data.DataLoader(
        pcam_val_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=16
    )

    return train_loader, val_loader, dataset_size