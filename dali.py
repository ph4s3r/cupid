##########################################################################################################
# Author: Peter Karacsonyi                                                                               #
# Last updated: 2024 jan 20                                                                              #
# https://docs.nvidia.com/deeplearning/dali                                                              #
# Data loading functions, speedtest & visualisation                                                      #
##########################################################################################################

import torch
import numpy as np
from pathlib import Path
from random import shuffle
import nvidia.dali.fn as fn
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from timeit import default_timer as timer
from nvidia.dali.pipeline import Pipeline

# @pipeline_def
# def train_pipeline(image_dir, shard_id, num_shards, stick_to_shard=False, pad_last_batch=False, device_id=0):
#     """
#     training pipeline: creates class based on the folder name (dir named '0' will assign '0' class etc..)

#     Args:
#         shard_id (_type_): get the yth part of x
#         num_shards (_type_): split to x parts
#         stick_to_shard (bool, optional): _description_. Defaults to False.
#         pad_last_batch (bool, optional): _description_. Defaults to False.
#         device_id (int, optional): for multi-gpu reading. Defaults to 0.

#     """    
#     jpegs, labels = fn.readers.file(
#         file_root=image_dir, 
#         random_shuffle=True, 
#         initial_fill=768, 
#         name="Reader", 
#         shard_id=shard_id, 
#         num_shards=num_shards, 
#         stick_to_shard=stick_to_shard, 
#         pad_last_batch=pad_last_batch
#     )
#     images = fn.decoders.image(jpegs, device='cpu')
#     images = fn.transpose(images, perm=[2, 0, 1])
#     return images, labels


def pipe(batch_size: int, num_threads: int, external_data, device_id=0):
    """
    pipeline using arbitrary custom external input iterator

    Args:
        num_threads (int): 16 was the best
        external_data (_type_): here comes the custom iterator
        device_id (int, optional): for multi-gpu reading. Defaults to 0.

    """    
    pipe = Pipeline(batch_size, num_threads, device_id)
    with pipe:
        im_files, labels = fn.external_source(source=external_data, num_outputs=2)
        images = fn.decoders.image(im_files, device="cpu")
        images = fn.transpose(images, perm=[2, 0, 1])
        pipe.set_outputs(images, labels)
    return pipe

def show_images(image_batch, batch_size):
    """
    visualize a batch of images (don't do too many, max 16 I guess)
    """    
    columns = 4
    rows = (batch_size + 1) // columns
    fig = plt.figure(figsize=(24, (24 // columns) * rows))
    gs = gridspec.GridSpec(rows, columns)
    
    for j in range(min(rows * columns, batch_size)):  
        img = image_batch.at(j)
        if img.dtype == np.float32:
            img = np.clip(img, 0, 1) 
        elif img.dtype == np.uint8:
            img = img / 255.0  
        img = np.transpose(img, (1, 0, 2)) 

        ax = plt.subplot(gs[j])
        ax.axis("off")
        ax.imshow(img)

    plt.show() 

def speedtest(pipeline, batch, n_threads):
    """
    can compare pipeline bandwidths

    e.g.: 
    dali.speedtest(dali_simple.mixed, batch_size, 16)
    dali.speedtest(dali_simple.train_pipeline_cpu, batch_size, 16)

    Speed: 10682.352789287172 imgs/s
    Speed: 11352.206050700624 imgs/s
    """    
    pipe = pipeline(batch_size=batch, num_threads=n_threads, device_id=0)
    pipe.build()
    # warmup
    for i in range(5):
        pipe.run()
    # test
    n_test = 20
    t_start = timer()
    for i in range(n_test):
        pipe.run()
    t = timer() - t_start
    print("Speed: {} imgs/s".format((n_test * batch)/t))


class TileTrainIterator(object):
    """
    summary: iterator object for training pipeline: creating class labels from filename: 'a' or 'b' 
    usage: images_dir must be a directory where subfolders with the name of the wsi exist
    TODO: can read to GPU right away
    https://docs.nvidia.com/deeplearning/dali/user-guide/docs/examples/general/data_loading/external_input.html
    """    
    def __init__(self, images_dir: str, batch_size: int, num_gpus=1, device_id=0):
        self.images_dir = images_dir
        self.batch_size = batch_size
        self.files = [] 

        # read all images
        self.wsi_folders = [f for f in Path(images_dir).glob("*/")]
        assert len(self.wsi_folders) > 0, f"No wsi subfolders found under {images_dir}"
        for wsi_folder in self.wsi_folders:
            pngs = [f for f in Path(wsi_folder).glob("*.png")]
            self.files.extend(pngs)
        self.data_set_len = len(self.files)
        # leaving this for possible future multigpu use-case
        self.files = self.files[self.data_set_len * device_id // num_gpus:
                                self.data_set_len * (device_id + 1) // num_gpus]
        self.n = len(self.files)
        assert self.n > 0, f"No png tiles found in folder(s) {self.wsi_folders}"
        print(f"DALI: training dataset size is {self.n}")

    def __iter__(self):
        self.i = 0
        shuffle(self.files)
        return self

    def __next__(self):
        batch = []
        labels = []

        if self.i >= self.n:
            self.__iter__()
            raise StopIteration

        for _ in range(self.batch_size):
            item =  self.files[self.i % self.n]
            batch.append(np.fromfile(item, dtype = np.uint8))
            if 'a' in item.stem:
                classlabel = 0
            elif 'b' in item.stem:
                classlabel = 1
            else:
                raise Exception(f"Error: cannot create classlabel from {item}. tile png does not have an 'a' or 'b' in filename.")
            labels.append(torch.tensor(classlabel,dtype=torch.int32)) # there are coordinates greater than 2^16
            self.i += 1
        return (batch, labels)

    def __len__(self):
        return self.data_set_len

    next = __next__


class TileInferIterator(object):
    """
    summary: iterator object for inference pipeline: reading filenames to create tile key labels 
    usage: images_dir must be a directory where png files exist
    
    TODO: can read to GPU right away
    https://docs.nvidia.com/deeplearning/dali/user-guide/docs/examples/general/data_loading/external_input.html
    """    
    def __init__(self, images_dir: str, batch_size: int, num_gpus=1, device_id=0):
        self.images_dir = images_dir
        self.batch_size = batch_size

        # read all images
        self.files = [f for f in Path(images_dir).glob("*.png")]
        self.data_set_len = len(self.files)

        # leaving this for possible future multigpu use-case
        self.files = self.files[self.data_set_len * device_id // num_gpus:
                                self.data_set_len * (device_id + 1) // num_gpus]
        self.n = len(self.files)
        assert self.n > 0, f"No png tiles found in dir {images_dir}"

    def __iter__(self):
        self.i = 0
        shuffle(self.files)
        return self

    def __next__(self):
        batch = []
        labels = []

        if self.i >= self.n:
            self.__iter__()
            raise StopIteration

        for _ in range(self.batch_size):
            item =  self.files[self.i % self.n]
            batch.append(np.fromfile(item, dtype = np.uint8))
            tilex, tiley = item.stem.split('_')[-2:]
            labels.append(torch.tensor([int(tilex), int(tiley)],dtype=torch.int32)) # there are coordinates greater than 2^16
            self.i += 1
        return (batch, labels)

    def __len__(self):
        return self.data_set_len

    next = __next__