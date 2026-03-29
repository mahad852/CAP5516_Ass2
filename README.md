# CAP5516 - Medical Image Computing: Assignment 2 #
This repository contains the implementation for Deep Learning-based Brain Tumor Segmentation Using MRI, specifically focusing on the Task01 of the BraTS challenge datasets. We employ a SegResNet architecture to perform 3D volumetric segmentation across four MRI modalities: FLAIR, T1-w, T1-gd, and T2-w.

## Installation ##
To set up the environment, ensure you have Conda installed and run the following commands:

```
# Create the environment
conda create -n cap5516_ass2 python=3.11

# Activate the environment
conda activate cap5516_ass2

# Install dependencies
pip install -r requirements.txt
```

## Running the code ##
### 1. Training (5-fold cross validation) ###
The training script, `train_5_fold.py`, performs a full 5-fold cross-validation on the `imagesTr` dataset. For each fold, it saves training metrics and the best model weights (based on the highest Dice score) into subdirectories (e.g., output_dir/0, output_dir/1, etc.).

Note: All subsequent scripts require this training step to be completed first, as they rely on the generated `output_dir`.

```
python train_5_fold.py --root /path/to/dataset --output_dir ./results --epochs 20 --val_interval 1
```

Arguments:
* `--root`: Path to the `Task01_BrainTumour` folder
* `--output_dir`: Directory where weights and logs will be saved.
* `--epochs`: Number of training epochs (default: 20).
* `--val_interval`: Interval between validation rounds.

### 2. Evaluation ###
The `test_5_fold.py` script performs inference on each fold's validation set using the saved best weights. It calculates and prints the Dice Score and Hausdorff Distance (95%) for the following regions:
* Whole Tumor (WT)
* Tumor Core (TC)
* Enhancing Tumor (ET)

```
python test_5_fold.py --root /path/to/dataset --output_dir ./results
```

### Visualization & Plotting ###
#### Save Segmentation Masks ####
To generate the qualitative results seen in the report, use `save_examples.py`. This script saves MRI slices overlaid with ground truth and predicted segmentation masks. 
```
python save_examples.py --root /path/to/dataset --output_dir ./results --fold 0 --modality 0
```

* `--fold`: Specify which fold's model to use for inference.
* `--modality`: MRI modality index (default is 0, typically FLAIR).

#### Plot Training Progress ####
The `plot_dice_graph.py` script aggregates the training logs from the `output_dir` and generates a plot of the average Dice score (aggregate and by class) over the training epochs.

```
python plot_dice_graph.py --output_dir ./results
```

## Results Summary ##
Our model achieves the following average performance across 5 folds:

| Region | Dice Score | Hausdorff Distance (95%) |
| --- | --- | --- |
| Whole Tumor | 0.77 | 9.47 |
| Tumor Core | 0.87 | 8.62 |
| Tumor Core | 0.76 | 6.05 |