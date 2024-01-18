##########################################################################################################
# Author: Peter Karacsonyi                                                                               #
# Last updated: 2024 jan 18                                                                              #
# https://docs.nvidia.com/deeplearning/dali                                                              #
# Data loading functions, speedtest & visualisation                                                      #
##########################################################################################################

import numpy as np
import nvidia.dali.fn as fn
import matplotlib.pyplot as plt
from nvidia.dali import pipeline_def
import matplotlib.gridspec as gridspec
from timeit import default_timer as timer


@pipeline_def
def cpupipe(image_dir, shard_id, num_shards, stick_to_shard=False, pad_last_batch=False):
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
        random_shuffle=True, 
        initial_fill=768, 
        name="Reader",
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