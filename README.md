# Semantic Segmentation with Sparse Point Supervision (LoveDA)

## Overview

This project investigates semantic segmentation of remote sensing imagery using **sparse point annotations** instead of full pixel-wise masks.

Traditional segmentation models require dense annotations, which are expensive and time-consuming to obtain.  
This work demonstrates how to:

- Simulate sparse point labels from full masks
- Train using **Partial Cross Entropy Loss**
- Evaluate full-mask performance (mIoU)
- Analyze how supervision density affects performance

Dataset used: **LoveDA (Urban/Rural domains)**

---

## Setup Instructions

### 1. Create Virtual Environment

```bash
python -m venv venv
```

Activate:

**Windows**
```bash
venv\Scripts\activate
```

**Mac/Linux**
```bash
source venv/bin/activate
```

---

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Dataset Setup (LoveDA)

Download LoveDA dataset from its official source.

Place it in the following structure:

```
data/raw/LoveDA/
  Train/
    Urban/
      images_png/
      masks_png/
    Rural/
      images_png/
      masks_png/
  Val/
    Urban/
      images_png/
      masks_png/
    Rural/
      images_png/
      masks_png/
```

⚠ Do NOT commit dataset files to git.

---

## Running Training

### Quick Dev Run (Fast CPU Test)

```bash
python train.py --config configs/dev.json
```

This runs:
- img_size = 128
- epochs = 2
- small K
- uniform sampling

---

### Full Experiment Run

```bash
python train.py --config configs/exp.json
```

---

## Running Experiments

### Experiment A: Point Density (K Sweep)

```bash
python -m src.run_experiments
```

This evaluates:

- K = {10, 50, 200, 1000}
- uniform sampling

---

### Generate Plots

```bash
python -m src.plot_results
```

Plots will be saved to:

```
reports/
```

---

## Method Summary

### Sparse Point Simulation

For each training mask:

- Select K labeled pixels
- Set all other pixels to IGNORE_INDEX
- Support:
  - uniform sampling
  - class-balanced sampling

---

### Partial Cross Entropy Loss

Loss is computed **only on labeled pixels**:

- Unlabeled pixels are ignored
- Loss is averaged over labeled pixels only
- Edge-safe if no labeled pixels

---

### Model

- UNet
- ResNet34 encoder (ImageNet pretrained)
- Output: (B, C, H, W)

---

## Evaluation

Validation uses full masks to compute:

- Mean Intersection over Union (mIoU)
- Pixel Accuracy

---

## Results

See:

```
reports/report.md
```

Includes:

- mIoU vs K plots
- Sampling strategy comparison
- Qualitative visualizations

---

## Reproducibility

All runs save:

- config.json
- checkpoints
- sample predictions

Under:

```
runs/<timestamp>/
```

---

## Notes

- Designed to run on CPU (reduced image sizes for feasibility)
- For GPU training, increase image size and epochs
- Dataset not included in repository

---

## Author

Felix Ojiambo  
Remote Sensing Sparse Supervision Project
```
