# clinical

deep learning framework preprocessing wsis with pathml &amp; training pytorch models

# project structure

## main runners

### preprocessing 

- **pathML-PipelineRunnner**: reading wsi's (+ annotations produced by atomrunner if required) and producing h5path files with pathml

### training

- **PyTorch-Trainer**: reading h5path and training a model
- in the trainer: global means and stds can be computed with (mean, std = helpers.ds_means_stds.mean_stds(full_ds)), need to re-run the trainer with manually entering the results into v2.compose.Normalize
- **PyTorch-PCAM-Trainer**: test model training on a PCAM dataset

## auxilliary functions

- **pathML-Transforms**: tool to load a wsi and check how a set of transforms like binary threshold, morphopen, etc make the mask look like 
- **Annotator**: reads geojson files (and wsi's for the original slide dimensions) and generates numpy.ndarrays that are the standard format of annotations of the new PathML masks, writes them out in numpy (.npy) array dump format

# versions

- tested only on windows: Anaconda / Python 3.9.18
- made with PathML 2.1.1 & torch 2.1.1+cu121

# concepts

- labels can be processed from filenames as well as metadata of wsi files
- in case of filename-based label generation the filenames should be something like "p091-normal.tif" where e.g. dash separates an identifier of scan or patient etc and classification group (class)

# todos

- understand why do we still have a lot of empty tiles while masks seem to be covering tissues properly (need to revise pathml tissue-gen as well)
- introduce some dynamic learning rate decay (fix epoch-count does not work)
- run on Ubuntu
- https://pytorch.org/docs/stable/tensorboard.html learn how to push plts to tensorboard, plot histograms, means stds, few images, masks etc...
- visual inference + heatmap
- albumentations (https://albumentations.ai/ & https://pathml.readthedocs.io/en/latest/examples/link_train_hovernet.html#Data-augmentation)
- auc
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

