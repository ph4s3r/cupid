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
from nvidia.dali.pipeline import Pipeline

def infer_pipe(batch_size: int, num_threads: int, external_data):
    """
    inference pipeline using custom external input iterator

    Args:
        num_threads (int): 16 was the best
        external_data (_type_): here comes the custom iterator

    """
    pipe = Pipeline(batch_size, num_threads, device_id=0)
    with pipe:
        im_files, labels = fn.external_source(source=external_data, num_outputs=2)
        images = fn.decoders.image(im_files, device="cpu")
        images = fn.transpose(images, perm=[2, 0, 1])
        pipe.set_outputs(images, labels)
    return pipe


class TileInferIterator(object):
    """
    summary: iterator object for inference pipeline: reading filenames to create tile key labels 
    usage: images_dir must be a directory where png files exist
    
    TODO: can read to GPU right away
    https://docs.nvidia.com/deeplearning/dali/user-guide/docs/examples/general/data_loading/external_input.html
    """    
    def __init__(self, images_dir: str, batch_size: int):
        self.images_dir = images_dir
        self.batch_size = batch_size

        # read all images
        self.files = [f for f in Path(images_dir).glob("*.png")]
        self.data_set_len = len(self.files)

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