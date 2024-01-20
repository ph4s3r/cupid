import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from timeit import default_timer as timer

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
    pipe = pipeline(batch_size=batch, num_threads=n_threads)
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