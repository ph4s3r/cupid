# clinical

deep learning framework preprocessing wsis with pathml &amp; training pytorch models

# project structure

- atomrunner: reads geojson files (and wsi's for the original slide dimensions) and generates numpy.ndarrays that are the standard format of annotations of the new PathML masks, writes them out in numpy (.npy) array dump format
- pathml-preprocessing: reading wsi's (+ annotations produced by atomrunner if required) and producing h5path files with pathml
- trainer: reading h5path and training a model
- in the trainer: global means and stds can be computed with (mean, std = helpers.ds_means_stds.mean_stds(full_ds)), need to re-run the trainer with manually entering the results into v2.compose.Normalize
- pathML-BinaryThreshold.py: tool to load a wsi to find the right binary threshold

# versions

- tested only on windows: Anaconda / Python 3.9.18
- made with PathML 2.1.1 & torch 2.1.1+cu121

# concepts

- labels will be processed from filenames so
- filenames should be something like "p091-normal.tif" where e.g. dash separates an identifier of scan or patient etc and classification group (class)

- actual label examples:
    - class: 0 or 1 or other integer
    - file: <p091>

# todos

- arra az otsura ranezni mert szar (csomo ures kep van)
- prep one-tile transform checker for PathML
- run on Ubuntu
- plot learning curve
- visual inference + heatmap
- albumentations (https://albumentations.ai/ & https://pathml.readthedocs.io/en/latest/examples/link_train_hovernet.html#Data-augmentation)
- auc
- test accuracy
- feature extraction

# notes

- when loaded echino wsis, needed to use openslide backend, otherwise it will be processed with bioformats by default then pipelines will fail with not being able to process the number of channels...
- why windows?: WSL was unstable and unpredictable because of resource handling
- why py instead of jupyter notebooks?: 
    - on windows debugging did not work properly. You only need notebooks if you want to rerun parts of your code sequentially and don't want to re-execute everything, also display results nicely and in a structured way
    - there is a nice profiler set up for python in .vscode/launch.json works with snakeviz
- how to install on windows: 
    - install latest anaconda
    - create new env named pathml with python 3.9x
    - make sure that the openslide importer is working (download binary and set path)
    - install latest cuda (11.2 was used here) that is supported by the latest pytorch (they need to be in sync)
    - install torch with going to their website and grabbing the appropriate install commands

