##########################################################################################################
# Author: Peter Karacsonyi                                                                               #
# Last updated: 2024 jan 20                                                                              #
# https://docs.nvidia.com/deeplearning/dali                                                              #
# Data loading functions, speedtest & visualisation                                                      #
##########################################################################################################

import copy
from pathlib import Path
import nvidia.dali.fn as fn
from nvidia.dali import pipeline_def
from nvidia.dali.plugin.base_iterator import LastBatchPolicy
from nvidia.dali.plugin.pytorch import DALIClassificationIterator

@pipeline_def
def train_pipeline(files, labels, shard_id, num_shards, image_size = 256, stick_to_shard=False, pad_last_batch=False):
    """
    training pipeline: creates class based on the folder name (dir named '0' will assign '0' class etc..)

    Args:
        shard_id (_type_): get the yth part of x
        num_shards (_type_): split to x parts
        stick_to_shard (bool, optional): Determines whether the reader should stick to a data shard instead of 
            going through the entire dataset. If decoder caching is used, it significantly reduces the amount
            of data to be cached, but might affect accuracy of the training..
        pad_last_batch (bool, optional): If set to True, pads the shard by repeating the last sample..

    """    
    ims, labels = fn.readers.file(
        files=files,
        labels=labels,
        random_shuffle=True, 
        initial_fill=768, 
        name="Reader", 
        shard_id=shard_id, 
        num_shards=num_shards, 
        stick_to_shard=stick_to_shard, 
        pad_last_batch=pad_last_batch
    )
    images = fn.decoders.image(ims, device='cpu')
    images = fn.transpose(images, perm=[2, 0, 1])
    images = fn.resize(images, size=[image_size, image_size])
    return images, labels


# def train_pipe(batch_size: int, num_threads: int, external_data):
#     """
#     training pipeline using external input iterator

#     Args:
#         num_threads (int): 16 was the best
#         external_data (_type_): iterator ref

#     """
#     pipe = Pipeline(batch_size, num_threads)
#     with pipe:
#         im_files, labels = fn.external_source(source=external_data, num_outputs=2)
#         images = fn.decoders.image(im_files, device="cpu")
#         images = fn.transpose(images, perm=[2, 0, 1])
#         pipe.set_outputs(images, labels)
#     return pipe


# class TileTrainIterator(object):
#     """
#     summary: iterator object for training pipeline: creating class labels from filename: 'a' or 'b' 
#     usage: images_dir must be a directory where subfolders with the name of the wsi exist
#     TODO: can read to GPU right away
#     https://docs.nvidia.com/deeplearning/dali/user-guide/docs/examples/general/data_loading/external_input.html
#     """    
#     def __init__(self, images_dir: str, batch_size: int, shard_id=0, num_shards=1):
#         self.images_dir = images_dir
#         self.batch_size = batch_size
#         self.files = [] 

#         # read all images
#         self.wsi_folders = [f for f in Path(images_dir).glob("*/")]
#         assert len(self.wsi_folders) > 0, f"No wsi subfolders found under {images_dir}"
#         for wsi_folder in self.wsi_folders:
#             pngs = [f for f in Path(wsi_folder).glob("*.png")]
#             self.files.extend(pngs)
#         self.data_set_len = len(self.files)
#         # sharding
#         self.files = self.files[shard_id::num_shards]
#         self.n = len(self.files)
#         assert self.n > 0, f"No png tiles found in folder(s) {self.wsi_folders}"
#         print(f"DALI: dataset size for shard {shard_id} is {self.n}")

#     def __iter__(self):
#         self.i = 0
#         shuffle(self.files)
#         return self

#     def __next__(self):
#         batch = []
#         labels = []

#         if self.i >= self.n:
#             self.__iter__()
#             raise StopIteration

#         for _ in range(self.batch_size):
#             item =  self.files[self.i % self.n]
#             batch.append(np.fromfile(item, dtype = np.uint8))
#             if 'a' in item.stem:
#                 classlabel = 0
#             elif 'b' in item.stem:
#                 classlabel = 1
#             else:
#                 raise Exception(f"Error: cannot create classlabel from {item}. tile png does not have an 'a' or 'b' in filename.")
#             labels.append(torch.tensor(classlabel,dtype=torch.int32)) # there are coordinates greater than 2^16
#             self.i += 1
#         return (batch, labels)

#     def __len__(self):
#         return self.data_set_len

#     next = __next__


def dataloaders(tiles_dir, batch_size, classlabels: list[str], image_size = 256):
    """_summary_

    Args:
        tiles_dir (_type_): _description_
        batch_size (_type_): _description_
        classlabels (list[str]): if str is found in the filename, it gets a numeric label = the index of str in the list

    Raises:
        Exception: _description_

    Returns:
        _type_: _description_
    """
    ########################
    # read tiles with DALI #
    ########################

    # we make a 80-20 split by using 5 shards (splitting the images to 5 batches: each shard number refers to 20% of the data)
    num_shards = 5
    val_shard_id = 4  # Shard 4 for validation

    files = []

    # create file and label list to pass to the pipeline for label creation
    wsi_folders = [f for f in Path(tiles_dir).glob("*/")]
    assert len(wsi_folders) > 0, f"No wsi subfolders found under {tiles_dir}"

    for wsi_folder in wsi_folders:
        filenames = [str(f) for f in Path(wsi_folder).glob("*.jpg")]
        files.extend(filenames)

    labels = copy.deepcopy(files)

    # convert all classlabels to lowercase -jic
    for i, c in enumerate(classlabels):
        classlabels[i] = c.lower()

    # gather info on class-label balance
    classlabel_counts = [0] * len(classlabels)

    for i, filename in enumerate(labels):
        filename = filename.split("/")[-1].lower()
        for label_index, label_substr in enumerate(classlabels):
            if label_substr in filename:     # if filename contains the label substr
                labels[i] = label_index # tile will be labeled with the index of string found in the array
                classlabel_counts[label_index] += 1
        if type(labels[i]) != int:
            raise Exception(f"Error: cannot create classlabel from {filename}. tile jpg does not have any of {classlabels} in filename.")            


    train_pipelines = [train_pipeline(
        files=files, 
        labels=labels,
        shard_id=shard_seq, 
        num_shards=num_shards,
        image_size=image_size,
        batch_size=batch_size,
        stick_to_shard=False,   # if True, loads only one shard per epoch, otherwise the entire dataset
        num_threads=16,
        device_id=0             # none is CPU, while 0 is GPU
    ) for shard_seq in range(num_shards)]

    val_pipeline = train_pipeline(
        files=files, 
        labels=labels,
        shard_id=val_shard_id, 
        num_shards=num_shards,
        image_size=image_size,
        batch_size=batch_size,
        stick_to_shard=False,   # if True, loads only one shard per epoch, otherwise the entire dataset
        num_threads=16,
        device_id=0             # none is CPU, while 0 is GPU
    )

    train_loader = DALIClassificationIterator(train_pipelines, reader_name="Reader", last_batch_policy=LastBatchPolicy.PARTIAL)
    val_loader   = DALIClassificationIterator(val_pipeline, reader_name="Reader", last_batch_policy=LastBatchPolicy.PARTIAL)

    dataset_size = len(labels)
    print(f"DALI loaded a dataset of size {dataset_size} successfully. ")
    print(f"classlabels: {classlabels} and tile counts respectively: {classlabel_counts} (in percentages: {[round(100 * c / dataset_size, 2) for c in classlabel_counts]})")

    return train_loader, val_loader, dataset_size