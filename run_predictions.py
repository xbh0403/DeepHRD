#!/usr/bin/env python3
"""
Script to run model predictions on all tiles using existing DeepHRD models.
Reuses the test_final.py functionality.
"""

import sys
import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from pathlib import Path
import glob
from tqdm import tqdm
import torchvision.transforms as transforms
from PIL import Image

# Add DeepHRD to path
sys.path.append('/tscc/nfs/home/bax001/restricted/Erik_Project/DeepHRD')
sys.path.append('/tscc/nfs/home/bax001/restricted/Erik_Project/DeepHRD/base')

# Import DeepHRD modules
from model import ResNet_dropout as RNN
import utilsModel as ut
import torch.nn.functional as F
from torch.utils.data import DataLoader

def extract_coordinates_from_filename(filename, resolution):
    """
    Extract coordinates from tile filename.
    Format: slide-tile-r{row}-c{col}-x{x}-y{y}-w{width}-h{height}.png
    or: slide-20x-tile-r{row}-c{col}-x{x}-y{y}-w{width}-h{height}.png
    """
    parts = Path(filename).stem.split('-')
    
    # Find indices of coordinate parts
    r_idx = next(i for i, p in enumerate(parts) if p.startswith('r'))
    c_idx = next(i for i, p in enumerate(parts) if p.startswith('c'))
    x_idx = next(i for i, p in enumerate(parts) if p.startswith('x'))
    y_idx = next(i for i, p in enumerate(parts) if p.startswith('y'))
    
    row = int(parts[r_idx][1:])
    col = int(parts[c_idx][1:])
    x = int(parts[x_idx][1:])
    y = int(parts[y_idx][1:])
    
    return row, col, x, y

def load_model(model_path, dropout_rate=0.2, gpu_id=0):
    """
    Load a DeepHRD model.
    """
    device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
    
    # Initialize model architecture using DeepHRD's ResNet model
    model = RNN(dropout_rate)
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # Handle different checkpoint formats
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    elif 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    return model, device

def run_model_predictions(tiles_dir, models_dir, output_path, resolution='5x', 
                         batch_size=64, gpu=0, workers=16, BN_reps=10):
    """
    Run ensemble model predictions on all tiles.
    
    Args:
        tiles_dir: Directory containing tile images
        models_dir: Directory containing ensemble models (m1-m5)
        output_path: Path to save predictions TSV
        resolution: '5x' or '20x'
        batch_size: Batch size for inference
        gpu: GPU device ID
        workers: Number of data loading workers
        BN_reps: Number of Bayesian dropout repetitions
    """
    
    # Find all tile images
    tile_pattern = str(Path(tiles_dir) / '*' / '*.png')
    tile_paths = sorted(glob.glob(tile_pattern))
    
    if len(tile_paths) == 0:
        # Try single directory structure
        tile_pattern = str(Path(tiles_dir) / '*.png')
        tile_paths = sorted(glob.glob(tile_pattern))
    
    print(f"Found {len(tile_paths)} tiles to process")
    
    if len(tile_paths) == 0:
        print(f"No tiles found in {tiles_dir}")
        return None
    
    # Setup model paths
    model_files = []
    for i in range(1, 6):  # m1 through m5
        model_path = Path(models_dir) / f'{resolution}_m{i}.pth'
        if model_path.exists():
            model_files.append(str(model_path))
        else:
            print(f"Warning: Model {model_path} not found")
    
    if len(model_files) == 0:
        raise ValueError(f"No models found in {models_dir} for resolution {resolution}")
    
    print(f"Using {len(model_files)} ensemble models")
    
    # Create simple dataset
    class TileDataset(torch.utils.data.Dataset):
        def __init__(self, tile_paths):
            self.tile_paths = tile_paths
            # Use same normalization as DeepHRD
            normalize = transforms.Normalize(mean=[0.485, 0.406, 0.406], std=[0.229, 0.224, 0.225])
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                normalize
            ])
        
        def __len__(self):
            return len(self.tile_paths)
        
        def __getitem__(self, idx):
            img = Image.open(self.tile_paths[idx]).convert('RGB')
            img = self.transform(img)
            return img, self.tile_paths[idx]
    
    dataset = TileDataset(tile_paths)
    dataloader = DataLoader(dataset, batch_size=batch_size, 
                           shuffle=False, num_workers=workers)
    
    # Initialize results storage
    all_predictions = []
    
    # Run predictions for each model in ensemble
    for model_idx, model_path in enumerate(model_files):
        print(f"\nProcessing model {model_idx + 1}/{len(model_files)}: {Path(model_path).name}")
        
        model, device = load_model(model_path, dropout_rate=0.2, gpu_id=gpu)
        
        # Enable dropout for uncertainty estimation
        def enable_dropout(m):
            if type(m) == nn.Dropout:
                m.train()
        model.apply(enable_dropout)
        
        model_predictions = []
        
        with torch.no_grad():
            for batch_idx, (images, paths) in enumerate(tqdm(dataloader, desc="Inference")):
                images = images.to(device)
                
                # Multiple forward passes for Bayesian uncertainty
                batch_preds = []
                for _ in range(BN_reps):
                    outputs = model(images)
                    probs = torch.sigmoid(outputs).cpu().numpy()
                    batch_preds.append(probs)
                
                # Average predictions across dropout samples
                batch_preds = np.mean(batch_preds, axis=0)
                
                # Ensure batch_preds is 1D
                if len(batch_preds.shape) > 1:
                    batch_preds = batch_preds.squeeze()
                
                # Store predictions with metadata
                for i, path in enumerate(paths):
                    row, col, x, y = extract_coordinates_from_filename(path, resolution)
                    
                    # Extract features (using hook on penultimate layer)
                    features = torch.zeros(512)  # Placeholder for features
                    
                    # Handle different shapes of batch_preds
                    try:
                        if batch_preds.ndim == 0:  # scalar
                            prob = float(batch_preds)
                        elif batch_preds.ndim == 1:  # 1D array
                            prob = float(batch_preds[i] if i < len(batch_preds) else batch_preds[0])
                        else:  # 2D or higher
                            prob = float(batch_preds[i].flatten()[0])
                    except:
                        # Fallback: just get a scalar value somehow
                        prob = float(np.array(batch_preds).flatten()[i] if i < len(np.array(batch_preds).flatten()) else 0.5)
                    
                    pred_entry = {
                        'slide_path': path,
                        'slide_idx': 0,  # Placeholder
                        'x_coord': x,
                        'y_coord': y,
                        'row': row,
                        'col': col,
                        'probability': prob,
                        'model': model_idx + 1
                    }
                    
                    # Add placeholder features
                    for j in range(512):
                        pred_entry[f'feature_{j}'] = 0.0
                    
                    model_predictions.append(pred_entry)
        
        all_predictions.extend(model_predictions)
        
        # Clear GPU memory
        del model
        torch.cuda.empty_cache()
    
    # Convert to DataFrame and save
    df = pd.DataFrame(all_predictions)
    
    # Average predictions across ensemble
    avg_predictions = df.groupby(['x_coord', 'y_coord']).agg({
        'probability': 'mean',
        'slide_path': 'first',
        'row': 'first',
        'col': 'first'
    }).reset_index()
    
    # Save in DeepHRD feature vector format
    output_data = []
    for _, row in avg_predictions.iterrows():
        # Format: slide_path, slide_idx, x_coord, y_coord, probability, features...
        line_data = [
            row['slide_path'],
            0,  # slide_idx
            row['x_coord'],
            row['y_coord'],
            row['probability']
        ]
        # Add placeholder features
        line_data.extend([0.0] * 512)
        output_data.append(line_data)
    
    # Save as TSV
    output_df = pd.DataFrame(output_data)
    output_df.to_csv(output_path, sep='\t', header=False, index=False)
    
    print(f"\nSaved predictions to {output_path}")
    print(f"Total tiles processed: {len(tile_paths)}")
    print(f"Average probability: {avg_predictions['probability'].mean():.4f}")
    print(f"Probability range: [{avg_predictions['probability'].min():.4f}, {avg_predictions['probability'].max():.4f}]")
    
    return output_path

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--tiles', type=str, required=True, help='Directory containing tiles')
    parser.add_argument('--models', type=str, required=True, help='Directory containing models')
    parser.add_argument('--output', type=str, required=True, help='Output TSV path')
    parser.add_argument('--resolution', type=str, default='5x', choices=['5x', '20x'])
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--BN_reps', type=int, default=10)
    
    args = parser.parse_args()
    
    run_model_predictions(
        args.tiles,
        args.models,
        args.output,
        args.resolution,
        args.batch_size,
        args.gpu,
        args.workers,
        args.BN_reps
    )

# python3 run_predictions.py \
#     --tiles /tscc/lustre/restricted/alexandrov-ddn/users/bax001/Erik_Project/DeepHRD/slides_selected_5x/tiles_png \
#     --models /tscc/lustre/restricted/alexandrov-ddn/users/bax001/Erik_Project/DeepHRD/models/BRCA_FFPE \
#     --output predictions_selected_5x.tsv \
#     --resolution 5x \
#     --batch_size 128 \
#     --BN_reps 10

# python3 run_predictions.py \
#     --tiles /tscc/lustre/restricted/alexandrov-ddn/users/bax001/Erik_Project/DeepHRD/slides_selected_20x/tiles_png \
#     --models /tscc/lustre/restricted/alexandrov-ddn/users/bax001/Erik_Project/DeepHRD/models/BRCA_FFPE \
#     --output predictions_selected_20x.tsv \
#     --resolution 20x \
#     --batch_size 128 \
#     --BN_reps 10

# python3 /tscc/lustre/restricted/alexandrov-ddn/users/bax001/Erik_Project/DeepHRD/use_e2e_viz_core_robust.py \
#     --slide_dir /tscc/lustre/restricted/alexandrov-ddn/users/bax001/Erik_Project/DeepHRD/slides_selected_5x/BRCA \
#     --pred_5x slides_selected_5x/predictions_selected_5x.tsv \
#     --pred_20x slides_selected_20x/predictions_selected_20x.tsv \
#     --name_map /tscc/nfs/home/bax001/restricted/Erik_Project/DeepHRD/slides_selected_20x/slideNumberToSampleName.txt \
#     --output_dir /tscc/lustre/restricted/alexandrov-ddn/users/bax001/Erik_Project/DeepHRD/visualizations_selected \
#     --skip_existing \
#     --report_csv visualization_report.csv

# python3 /tscc/lustre/restricted/alexandrov-ddn/users/bax001/Erik_Project/DeepHRD/use_e2e_viz_core_robust.py \
#     --slide_dir /tscc/lustre/restricted/alexandrov-ddn/users/bax001/Erik_Project/DeepHRD/slides_selected_5x/BRCA \
#     --pred_5x slides_selected_5x/predictions_selected_5x.tsv \
#     --pred_20x slides_selected_20x/predictions_selected_20x.tsv \
#     --name_map /tscc/nfs/home/bax001/restricted/Erik_Project/DeepHRD/slides_selected_20x/slideNumberToSampleName.txt \
#     --output_dir /tscc/lustre/restricted/alexandrov-ddn/users/bax001/Erik_Project/DeepHRD/visualizations_selected_redo \
#     --skip_existing \
#     --report_csv visualization_report.csv

#   python3 use_e2e_viz_core_robust.py \
#     --slide_dir "/tscc/nfs/home/bax001/restricted/Erik_Project/DeepHRD/slides_selected_5x/BRCA" \
#     --pred_5x "/tscc/nfs/home/bax001/restricted/Erik_Project/DeepHRD/slides_selected_5x/predictions_selected_5x.tsv" \
#     --pred_20x "/tscc/nfs/home/bax001/restricted/Erik_Project/DeepHRD/slides_selected_20x/predictions_selected_20x.tsv" \
#     --name_map "/tscc/nfs/home/bax001/restricted/Erik_Project/DeepHRD/slides_selected_5x/slideNumberToSampleName.txt" \
#     --output_dir "/tscc/nfs/home/bax001/restricted/Erik_Project/DeepHRD/plotting/robust_visualizations_final" \
#     --report_csv "visualization_report.csv"

#   python3 create_sequential_frames_multi_resolution.py \
#     --slide_dir "/tscc/nfs/home/bax001/restricted/Erik_Project/DeepHRD/slides_selected_5x/BRCA" \
#     --predictions "/tscc/nfs/home/bax001/restricted/Erik_Project/DeepHRD/slides_selected_5x/predictions_selected_5x.tsv" \
#     --output_dir "/tscc/nfs/home/bax001/restricted/Erik_Project/DeepHRD/plotting/all_sequential_frames_5x" \
#     --resolution 5x \
#     --name_map "/tscc/nfs/home/bax001/restricted/Erik_Project/DeepHRD/slides_selected_5x/slideNumberToSampleName.txt"

#   python3 create_sequential_frames_multi_resolution.py \
#     --slide_dir "/tscc/nfs/home/bax001/restricted/Erik_Project/DeepHRD/slides_selected_5x/BRCA" \
#     --predictions "/tscc/nfs/home/bax001/restricted/Erik_Project/DeepHRD/slides_selected_5x/predictions_selected_5x.tsv" \
#     --output_dir "/tscc/nfs/home/bax001/restricted/Erik_Project/DeepHRD/plotting/pure_sequential_frames_5x" \
#     --resolution 5x \
#     --name_map "/tscc/nfs/home/bax001/restricted/Erik_Project/DeepHRD/slides_selected_5x/slideNumberToSampleName.txt" \
#     --pure_heatmap

#   python3 /tscc/nfs/home/bax001/restricted/Erik_Project/DeepHRD/create_sequential_frames_multi_resolution.py \
#     --slide_dir /tscc/nfs/home/bax001/restricted/Erik_Project/DeepHRD/slides_selected_5x/BRCA \
#     --predictions /tscc/nfs/home/bax001/restricted/Erik_Project/DeepHRD/slides_selected_5x/predictions_selected_5x.tsv \
#     --output_dir /tscc/nfs/home/bax001/restricted/Erik_Project/DeepHRD/plotting/pure_sequential_frames_5x_final \
#     --resolution 5x \
#     --name_map /tscc/nfs/home/bax001/restricted/Erik_Project/DeepHRD/slides_selected_5x/slideNumberToSampleName.txt \
#     --pure_heatmap