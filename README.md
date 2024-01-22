# clinical

deep learning framework preprocessing wsis with pathml &amp; training pytorch models

# project structure

## main runners


### preprocessing previews

- **pathML-TissueDetection-Preview.py**:    check how a mask with pathml "TissueDetection" covers an image
- **pathML-Transforms-Preview.py**:         check how a mask with pathml single transforms like binary threshold, morphopen, etc covers an image

### preprocessing 

- **pathML-placenta-TissueDetection.ipynb**: reading wsi's and producing h5path files with pathml

### saving tiles from h5 to png images

- **pathML-TileSaver.py**: read h5s and write image files

### training

- **PyTorch-Trainer**: training!
- NOTE: in the trainer: global means and stds can be computed with (mean, std = helpers.ds_means_stds.mean_stds(full_ds)), need to re-run the trainer with manually entering the results into v2.compose.Normalize

### fine-tuning / transfer learning (continue training from a checkpoint)

- **PyTorch-FineTuner**: load checkpoint and session and continue training

### evaluation

- **slide-infer.ipynb**: set model type, saved weights, session name and test data to get accuracy, roc/auc graph, top false negative images and write data to tensorboard

## auxilliary functions

- **Annotator**: (cannot process big files...) reads geojson files (and wsi's for the original slide dimensions) and generates numpy.ndarrays that are the standard format of annotations of the new PathML masks, writes them out in numpy (.npy) array dump format

# versions

- tested with conda & Python 3.9
- PathML 2.1.1 & torch 2.1.1+cu121

# labeling

- class labels are processed from wsi filenames
- tile keys are also processed from wsi filenames

# TODOs

- ray-tune
- feature extraction
- separate config from code: with e.g. Hydra or ConfigArgParse
- switch to pytorch amp from APEX amp, because it is deprecated
- study se_resnext101_32x4d more to fine-tune
- think about segmentation -> https://github.com/ph4s3r/clinical/discussions/4
- test in different env like Azure (just make sure the requirements.txt / conda env yaml is complete)
- test out different image sizes / batch

# notes

- when loaded echino wsis, needed to use openslide backend, otherwise it will be processed with bioformats by default then pipelines will fail with not being able to process the number of channels...
- don't use WSL!! it was unstable and unpredictable because of resource handling
- why py instead of jupyter notebooks?: 
    - on windows debugging did not work properly. You only need notebooks if you want to rerun parts of your code sequentially and don't want to re-execute everything, also display results nicely and in a structured way
    - there is a nice profiler set up for python in .vscode/launch.json works with snakeviz
- how to install on windows: 
    - install latest anaconda
    - create new env named pathml with python 3.9x
    - make sure that the openslide importer is working (download binary and set path)
    - install latest cuda (11.2 was used here) that is supported by the latest pytorch (they need to be in sync):
    - install torch with going to their website and grabbing the appropriate install commands

