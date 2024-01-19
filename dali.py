##########################################################################################################
# Author: Peter Karacsonyi                                                                               #
# Last updated: 2024 jan 19                                                                              #
# https://docs.nvidia.com/deeplearning/dali                                                              #
# Data loading functions, speedtest & visualisation                                                      #
##########################################################################################################

import torch
import numpy as np
from pathlib import Path
from random import shuffle
import nvidia.dali.fn as fn
import matplotlib.pyplot as plt
import nvidia.dali.types as types
from nvidia.dali import pipeline_def
from nvidia.dali.pipeline import Pipeline
import matplotlib.gridspec as gridspec
from timeit import default_timer as timer


@pipeline_def
def cpupipe(image_dir, shard_id, num_shards, stick_to_shard=False, pad_last_batch=False, device_id=0):
    jpegs, labels = fn.readers.file(
        file_root=image_dir, 
        random_shuffle=True, 
        initial_fill=768, 
        name="Reader", 
        shard_id=shard_id, 
        num_shards=num_shards, 
        stick_to_shard=stick_to_shard, 
        pad_last_batch=pad_last_batch
    )
    images = fn.decoders.image(jpegs, device='cpu')
    images = fn.transpose(images, perm=[2, 0, 1])
    return images, labels

# Define your specific DALI pipeline
@pipeline_def
def cpupipe_infer(image_dir):
    jpegs, labels = fn.readers.file(
        file_root=image_dir,
        initial_fill=768, 
        name="Infer_Reader",
    )
    images = fn.decoders.image(jpegs, device='cpu')
    images = fn.transpose(images, perm=[2, 0, 1])
    return images, labels


def show_images(image_batch, batch_size):
    columns = 4
    rows = (batch_size + 1) // columns
    fig = plt.figure(figsize=(24, (24 // columns) * rows))
    gs = gridspec.GridSpec(rows, columns)
    
    for j in range(min(rows * columns, batch_size)):  # Adjust loop to not exceed batch size
        img = image_batch.at(j)
        if img.dtype == np.float32:
            img = np.clip(img, 0, 1)  # Ensure float images are in [0, 1] range
        elif img.dtype == np.uint8:
            img = img / 255.0  # Normalize uint8 images to [0, 1] range
        img = np.transpose(img, (1, 0, 2))  # Transpose from CHW to HWC format for Matplotlib

        ax = plt.subplot(gs[j])
        ax.axis("off")
        ax.imshow(img)

    plt.show()  # Explicitly call plt.show() to display the images

def speedtest(pipeline, batch, n_threads):
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

# usage: 
# dali_simple.speedtest(dali_simple.hypipe, batch_size, 16)
# dali_simple.speedtest(dali_simple.cpupipe, batch_size, 16)

#Speed: 10682.352789287172 imgs/s
#Speed: 11352.206050700624 imgs/s

class ExternalInputIterator(object):
    """
    This iterator reads images and creates labels from filenames
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


def ExternalSourcePipeline(batch_size: int, num_threads: int, external_data, device_id=0):
    pipe = Pipeline(batch_size, num_threads, device_id)
    with pipe:
        im_files, labels = fn.external_source(source=external_data, num_outputs=2)
        images = fn.decoders.image(im_files, device="cpu")
        images = fn.transpose(images, perm=[2, 0, 1])
        pipe.set_outputs(images, labels)
    return pipe