# DeepHRD Heatmap Generation Tutorial

This tutorial provides a complete step-by-step guide for generating heatmaps and sequential frame animations from DeepHRD predictions.

## Overview

The pipeline consists of four main steps:
1. **Preprocessing**: Tile whole slide images (WSI) at different resolutions
2. **Prediction**: Run DeepHRD model to get tile-level predictions
3. **Static Visualization**: Generate single heatmap overlays
4. **Sequential Animation**: Create frame-by-frame progressive heatmaps

## Prerequisites

### Clone Repository

```bash
# Clone the DeepHRD repository
git clone https://github.com/alexandrovteam/DeepHRD.git
cd DeepHRD
```

### Environment Setup

```bash
# Create conda environment from requirements file
conda create --name deephrd --file requirements.txt
conda activate deephrd
```

### Download Pre-trained Models

```bash
# Download DeepHRD model weights
wget -r ftp://alexandrovlab-ftp.ucsd.edu/pub/tools/DeepHRD
```

### Required Files
- Whole slide images (.svs format)
- DeepHRD trained models (downloaded above or custom trained)

## Step 1: Preprocessing with Tile Extraction
### 1.1 Running Preprocessing

```python
import preprocessing_with_tile_data_overlap as preproc

# For 5x tiles
preproc.preprocess_images(
    project="BRCA",
    projectPath="/path/to/slides/",
    max_cpu=16,
    save_top_tiles=True,
    overlap=0  # Set to 0 for no overlap between tiles
)

# For 20x tiles
# Please modify 
# preprocessing_with_tile_data_overlap.py 
# line 1451 to RESOLUTION = '20x'
preproc.preprocess_images(
    project="BRCA", 
    projectPath="/path/to/slides/",
    max_cpu=16,
    save_top_tiles=True,
    overlap=0
)
```

## Step 2: Running Predictions

### 2.1 Create Metadata File (OPTIONAL FOR HEATMAP GENERATION)

[OPTIONAL FOR HEATMAP GENERATION] 

Create a tab-separated file with slide information (needed for DeepHRD_predict.py, could be skipped for run_predictions.py):

```tsv
slide	patient	label	partition
001.svs	P001	1	test
002.svs	P002	0	test
003.svs	P003	1	test
```

### 2.2 Run DeepHRD Predictions (5x and 20x)

Use `run_predictions.py` for both 5x and 20x tiles

```bash
python run_predictions.py \
  --tiles /path/to/preprocessed/tiles/5x/ \
  --models /path/to/models/BRCA_FFPE/ \
  --output /path/to/predictions/ \
  --resolution 5x \
  --batch_size 128 \
  --BN_reps 10
```

```bash
python run_predictions.py \
  --tiles /path/to/preprocessed/tiles/20x/ \
  --models /path/to/models/BRCA_FFPE/ \
  --output /path/to/predictions/ \
  --resolution 20x \
  --batch_size 128 \
  --BN_reps 10
```


### 2.3 Prediction Output Format

The predictions are saved as TSV files with the following format (the features are placeholder as they are not used for heatmap generation):
```
tile_path	slide_number	x_coord	y_coord	hrd_probability features...
/path/001/tile_x0_y0.png	001	0	0	0.823   0.0
/path/001/tile_x1024_y0.png	001	1024	0	0.456   0.0
```

## Step 3: Static Heatmap Visualization

### 3.1 Using use_easymsi_viz_core_robust.py

This script creates single heatmap overlays with automatic parameter detection:

```bash
python use_easymsi_viz_core_robust.py \
  --slide_dir /path/to/slides/BRCA \
  --pred_5x /path/to/predictions_5x.tsv \
  --pred_20x /path/to/predictions_20x.tsv \
  --name_map /path/to/slideNumberToSampleName.txt \
  --output_dir /path/to/visualizations/ \
  --report_csv visualization_report.csv
```

## Step 4: Sequential Frame Animation

### 4.1 Using create_sequential_frames_multi_resolution.py

This jupyter notebook `sequential_to_vedio.ipynb` creates sequential frames from the heatmaps.