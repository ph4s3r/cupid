# The CUPID Project: Classifying Underlying Placental Issues in Premature Infants with Deep Learning

## Synopsis:

This study aims to classify placentas using deep learning methods by utilizing digital Whole-Slide Images (WSI) of placental tissues. The classification is performed through semi-supervised learning (no annotations are used, only slide-level labeling) and aims to distinguish between two main groups of placentas: those from newborns with a defined clinical endpoint (bronchopulmonary dysplasia BPD, necrotising enterocolitis, focal intestinal perforation, intraventricular hemorrhage) and those from newborns without these endpoints. Additionally, further clinical parameters such as gestational week, maternal risk factors and possibly other variables are included in the analysis.

## Objectives:

Classification of placentas into two groups based on clinical endpoints.
Primary clinical endpoint:
bronchopulmonary dysplasia at 36 weeks PMA
Secondary clinical endpoints:
- Death at 28 days
- Presence of intraventricular hemorrhage throughout 36 weeks PMA
- Presence of necrotising enterocolitis or focal intestinal perforation throughout 36 weeks PMA

Exploration of the relationship between clinical parameters and placental classification.
Application of deep learning for classification to investigate survival rates in relation to placental classification.
There are no plans to share the data with third parties outside of the physicians involved in the project


# Project Structure

Various solutions exist for dataloading and tile-extraction depending on dataset we intend to work on

### Tile Extraction

- **tileextraction/pathml-tissueDetectionpreview.py**:      check how a mask with pathML "TissueDetection" covers an image
- **tileextraction/pathml-transformspreview.py**:           check how a mask with pathML single transforms like binary threshold, morphopen, etc covers an image
- **tileextraction/wsi-tileextractor.py**:                  reading tiff wsi files in a folder and extract jpeg tiles
- **tileextraction/wsiprocessor.py**:                       wsi processing functionality of wsi-tileextractor.py (it runs as subprocess) 
- **tileextraction/wsi-to-h5.ipynb**:                       (legacy) reading tiff wsi files with pathML in a folder and produce h5path files
- **tileextraction/h5-to-tiles.py**:                        read h5s files and extract jpeg tiles

### Dataloading

- **dataloaders/dali_...py**:                               data loading functions with Nvidia DALI, output is always a PyTorch type dataloader
- **dataloaders/pcam_h5_dataloader.py**:                    can load a PyTorch h5 dataset (from torchvision.datasets)

## Training

- **raytune-train**:                                      running ray-tune experiment to find hyperparams & train
- NOTE: in the trainer: global means and stds can be computed with (mean, std = helpers.ds_means_stds.mean_stds(full_ds)), need to re-run the trainer with manually entering the results into v2.compose.Normalize

### Inference

- **infer.ipynb**:                                    infer a whole slide image to see tile heatmap, roc/auc graph and other prediction based data

# Versions

- tested with conda & Python 3.9
- PathML 2.1.1 & torch 2.1.1+cu121

# Installation

- see requirements/... files for conda and pip
