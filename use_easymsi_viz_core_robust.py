#!/usr/bin/env python3

"""
Robust version of E2E_VIZ core visualization for DeepHRD predictions.
Includes error handling and generates a CSV report of processing status.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
import matplotlib.pyplot as plt
from PIL import Image
import openslide
from scipy.ndimage import gaussian_filter
import traceback
from datetime import datetime

# Add E2E_VIZ to path
sys.path.insert(0, '/tscc/nfs/home/bax001/project/E2E_VIZ')


def create_attention_heatmap(
    attention_weights: np.ndarray,
    tile_coordinates: pd.DataFrame,
    slide_dimensions: tuple,
    tile_size: int = 256,
    downsample_factor: int = 32,
    smooth: bool = False,
    smooth_sigma: float = 0.0,
    normalize: bool = False
) -> np.ndarray:
    """
    Direct copy of E2E_VIZ create_attention_heatmap function with error handling.
    """
    # Validate inputs
    if attention_weights.ndim != 1:
        raise ValueError(f"Attention weights must be 1D array, got ndim {attention_weights.ndim}")
    
    # Handle empty coordinates
    if len(tile_coordinates) == 0:
        # Return empty heatmap
        heatmap_width = slide_dimensions[0] // downsample_factor
        heatmap_height = slide_dimensions[1] // downsample_factor
        return np.full((heatmap_height, heatmap_width), np.nan, dtype=np.float32)
    
    # Convert coordinates to numpy array
    if isinstance(tile_coordinates, pd.DataFrame):
        if 'x' in tile_coordinates.columns and 'y' in tile_coordinates.columns:
            coords = tile_coordinates[['x', 'y']].values
        else:
            # If columns are missing, return empty heatmap
            heatmap_width = slide_dimensions[0] // downsample_factor
            heatmap_height = slide_dimensions[1] // downsample_factor
            return np.full((heatmap_height, heatmap_width), np.nan, dtype=np.float32)
    else:
        coords = np.array(tile_coordinates)
    
    if len(coords) != len(attention_weights):
        raise ValueError(f"Number of coordinates ({len(coords)}) must match attention weights ({len(attention_weights)})")
    
    # Normalize attention weights to [0, 1] only if requested
    if normalize and attention_weights.max() > attention_weights.min():
        attention_norm = (attention_weights - attention_weights.min()) / (attention_weights.max() - attention_weights.min())
    else:
        attention_norm = attention_weights.copy()
    
    # Calculate heatmap dimensions
    heatmap_width = slide_dimensions[0] // downsample_factor
    heatmap_height = slide_dimensions[1] // downsample_factor
    
    # Initialize heatmap with NaN (transparent background)
    heatmap = np.full((heatmap_height, heatmap_width), np.nan, dtype=np.float32)
    
    for idx, (x, y) in enumerate(coords):
        attention_value = attention_norm[idx]
        
        # Calculate position in heatmap
        x_start = int(x) // downsample_factor
        y_start = int(y) // downsample_factor
        x_end = int(x + tile_size) // downsample_factor
        y_end = int(y + tile_size) // downsample_factor
        
        # Clip to heatmap bounds
        x_end = min(x_end, heatmap_width)
        y_end = min(y_end, heatmap_height)
        
        # Set attention values as solid squares
        heatmap[y_start:y_end, x_start:x_end] = attention_value
    
    # Apply smoothing if requested
    if smooth and smooth_sigma > 0:
        heatmap = gaussian_filter(heatmap, sigma=smooth_sigma)
    
    # print(f"Created heatmap: shape={heatmap.shape}, range=[{np.nanmin(heatmap):.3f}, {np.nanmax(heatmap):.3f}]")
    return heatmap


def create_slide_thumbnail(slide_path, size=(2000, 2000)):
    """Create high-res thumbnail from slide."""
    try:
        slide = openslide.OpenSlide(str(slide_path))
        thumbnail = slide.get_thumbnail(size)
        slide.close()
        return thumbnail
    except Exception as e:
        print(f"Error creating thumbnail: {e}")
        # Return a blank image
        return Image.new('RGB', size, (255, 255, 255))


def overlay_heatmap_on_slide(
    heatmap: np.ndarray,
    slide_thumbnail: Image.Image,
    colormap: str = 'viridis',
    alpha: float = 0.6,
    vmin: float = None,
    vmax: float = None
) -> Image.Image:
    """
    Overlay heatmap on slide thumbnail.
    
    Parameters:
    -----------
    heatmap : np.ndarray
        Heatmap array with values to visualize
    slide_thumbnail : Image.Image
        Slide thumbnail image
    colormap : str
        Matplotlib colormap name
    alpha : float
        Transparency for overlay (0=transparent, 1=opaque)
    vmin : float, optional
        Minimum value for color normalization. If None, uses heatmap min.
    vmax : float, optional
        Maximum value for color normalization. If None, uses heatmap max.
    """
    from matplotlib import cm
    from matplotlib.colors import Normalize
    from scipy.ndimage import zoom
    
    # Convert thumbnail to array
    thumb_array = np.array(slide_thumbnail)
    
    # Check if heatmap is all NaN
    if np.all(np.isnan(heatmap)):
        return slide_thumbnail
    
    # Resize heatmap to match thumbnail using nearest neighbor to preserve sharp boundaries
    zoom_y = thumb_array.shape[0] / heatmap.shape[0]
    zoom_x = thumb_array.shape[1] / heatmap.shape[1]
    heatmap_resized = zoom(heatmap, (zoom_y, zoom_x), order=0)  # order=0 for nearest neighbor
    
    # Apply colormap
    cmap = cm.get_cmap(colormap)
    
    # Handle NaN values for normalization
    valid_mask = ~np.isnan(heatmap_resized)
    
    # Use provided vmin/vmax or calculate from data
    if vmin is None or vmax is None:
        if np.any(valid_mask):
            calc_vmin = np.nanmin(heatmap_resized)
            calc_vmax = np.nanmax(heatmap_resized)
        else:
            calc_vmin, calc_vmax = 0, 1
        
        if vmin is None:
            vmin = calc_vmin
        if vmax is None:
            vmax = calc_vmax
    
    norm = Normalize(vmin=vmin, vmax=vmax)
    heatmap_colored = cmap(norm(np.where(np.isnan(heatmap_resized), 0, heatmap_resized)))
    heatmap_colored = (heatmap_colored[:, :, :3] * 255).astype(np.uint8)
    
    # Create overlay - only apply heatmap where it's not NaN
    overlay = thumb_array.copy()
    overlay[valid_mask] = (alpha * heatmap_colored[valid_mask] + (1 - alpha) * thumb_array[valid_mask]).astype(np.uint8)
    
    return Image.fromarray(overlay)


def parse_deephrd_predictions(pred_file, slide_number=None):
    """Parse DeepHRD predictions into E2E_VIZ format."""
    try:
        df = pd.read_csv(pred_file, sep='\t', header=None)
        
        coords = []
        probs = []
        
        for _, row in df.iterrows():
            if len(row) < 5:
                continue
            
            tile_path = row[0]
            
            # Filter by slide number if specified
            if slide_number and f'/{slide_number}/' not in str(tile_path):
                continue
            
            x = int(row[2])
            y = int(row[3])
            prob = float(row[4])
            
            coords.append({'x': x, 'y': y})
            probs.append(prob)
        
        # Create DataFrame for coordinates
        tile_coordinates = pd.DataFrame(coords)
        
        # Create numpy array for attention weights
        attention_weights = np.array(probs)
        
        return attention_weights, tile_coordinates
    except Exception as e:
        print(f"Error parsing predictions: {e}")
        return np.array([]), pd.DataFrame()


def get_tile_parameters(slide_number, resolution, base_dir=None):
    """
    Automatically determine tile parameters based on objective power.
    
    Returns:
        tuple: (tile_size, downsample_factor)
    """
    # Try to find objectiveInfo.txt in various locations
    possible_paths = []
    if base_dir:
        possible_paths.append(Path(base_dir) / "objectiveInfo.txt")
    
    # Common locations
    possible_paths.extend([
        Path("/tscc/nfs/home/bax001/restricted/Erik_Project/DeepHRD/slides_selected_5x/objectiveInfo.txt"),
        Path("/tscc/nfs/home/bax001/restricted/Erik_Project/DeepHRD/slides_selected_20x/objectiveInfo.txt"),
        Path("./objectiveInfo.txt")
    ])
    
    objective_power = None
    for obj_path in possible_paths:
        if obj_path.exists():
            try:
                with open(obj_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split('\t')
                        if len(parts) == 2 and parts[0] == slide_number:
                            objective_power = int(parts[1])
                            print(f"  Found objective power {objective_power}x for slide {slide_number}")
                            break
                if objective_power:
                    break
            except:
                continue
    
    # Default to 20x if not found
    if objective_power is None:
        print(f"  Warning: Could not find objective power for slide {slide_number}, defaulting to 20x")
        objective_power = 20
    
    # Determine parameters based on objective power and resolution
    if resolution == '5x':
        if objective_power == 40:
            tile_size = 2048  # 256 * 8
            downsample_factor = 64
        elif objective_power == 20:
            tile_size = 1024  # 256 * 4
            downsample_factor = 32
        else:  # 10x
            tile_size = 512   # 256 * 2
            downsample_factor = 16
    elif resolution == '20x':
        if objective_power == 40:
            tile_size = 512   # 256 * 2
            downsample_factor = 8
        elif objective_power == 20:
            tile_size = 256   # 256 * 1
            downsample_factor = 8
        else:  # 10x - not supported for 20x
            print(f"  Warning: 20x resolution not supported at 10x objective, using defaults")
            tile_size = 256
            downsample_factor = 8
    else:
        # Default fallback
        tile_size = 256
        downsample_factor = 8
    
    return tile_size, downsample_factor


def visualize_deephrd_with_e2e_core(
    slide_path,
    pred_5x_path,
    pred_20x_path,
    slide_number,
    output_dir,
    skip_existing=True,
    base_dir=None
):
    """Use E2E_VIZ-style functions to create visualizations with error handling."""
    
    slide_name = Path(slide_path).stem
    print(f"Processing {slide_name}...")
    
    # Check if output already exists
    output_path = Path(output_dir) / f'{slide_name}_deephrd_visualization.png'
    if skip_existing and output_path.exists():
        print(f"Skipping {slide_name} - output already exists")
        return {
            'slide': slide_name,
            'status': 'skipped',
            'error': 'Already exists',
            'tiles_5x': 0,
            'tiles_20x': 0,
            'output_path': str(output_path)
        }
    
    try:
        # Get slide dimensions
        slide = openslide.OpenSlide(str(slide_path))
        slide_dimensions = slide.dimensions
        slide.close()
        print(f"Slide dimensions: {slide_dimensions}")
        
        # Parse predictions
        attention_5x, coords_5x = parse_deephrd_predictions(pred_5x_path, slide_number)
        attention_20x, coords_20x = parse_deephrd_predictions(pred_20x_path, slide_number)
        
        print(f"Found {len(attention_5x)} 5x and {len(attention_20x)} 20x predictions")
        
        # Check if we have any predictions
        if len(attention_5x) == 0 and len(attention_20x) == 0:
            return {
                'slide': slide_name,
                'status': 'error',
                'error': 'No predictions found',
                'tiles_5x': 0,
                'tiles_20x': 0,
                'output_path': ''
            }
        
        # Create slide thumbnail
        print("Creating slide thumbnail...")
        thumbnail = create_slide_thumbnail(slide_path)
        
        # Get automatic parameters for 5x
        tile_size_5x, downsample_5x = get_tile_parameters(slide_number, '5x', base_dir)
        print(f"  Using 5x parameters: tile_size={tile_size_5x}, downsample={downsample_5x}")
        
        # Create 5x heatmap
        print("Creating 5x heatmap...")
        if len(attention_5x) > 0:
            heatmap_5x = create_attention_heatmap(
                attention_weights=attention_5x,
                tile_coordinates=coords_5x,
                slide_dimensions=slide_dimensions,
                tile_size=tile_size_5x,
                downsample_factor=downsample_5x,
                smooth=False,
                smooth_sigma=1.5,
                normalize=False
            )
        else:
            # Create empty heatmap
            heatmap_5x = np.full((slide_dimensions[1]//downsample_5x, slide_dimensions[0]//downsample_5x), np.nan, dtype=np.float32)
        
        # Get automatic parameters for 20x
        tile_size_20x, downsample_20x = get_tile_parameters(slide_number, '20x', base_dir)
        print(f"  Using 20x parameters: tile_size={tile_size_20x}, downsample={downsample_20x}")
        
        # Create 20x heatmap
        print("Creating 20x heatmap...")
        if len(attention_20x) > 0:
            heatmap_20x = create_attention_heatmap(
                attention_weights=attention_20x,
                tile_coordinates=coords_20x,
                slide_dimensions=slide_dimensions,
                tile_size=tile_size_20x,
                downsample_factor=downsample_20x,
                smooth=False,
                smooth_sigma=1.0,
                normalize=False
            )
        else:
            # Create empty heatmap
            heatmap_20x = np.full((slide_dimensions[1]//downsample_20x, slide_dimensions[0]//downsample_20x), np.nan, dtype=np.float32)
        
        # Create overlays
        print("Creating overlays...")
        overlay_5x = overlay_heatmap_on_slide(
            heatmap=heatmap_5x,
            slide_thumbnail=thumbnail,
            colormap='seismic',
            alpha=0.5
        )
        
        overlay_20x = overlay_heatmap_on_slide(
            heatmap=heatmap_20x,
            slide_thumbnail=thumbnail,
            colormap='seismic',
            alpha=0.5
        )
        
        # Save outputs
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save individual components
        thumbnail.save(output_dir / f'{slide_name}_thumbnail.png')
        overlay_5x.save(output_dir / f'{slide_name}_5x_overlay.png')
        overlay_20x.save(output_dir / f'{slide_name}_20x_overlay.png')
        
        print(f"Successfully saved visualization to {output_path}")
        
        return {
            'slide': slide_name,
            'status': 'success',
            'error': '',
            'tiles_5x': len(attention_5x),
            'tiles_20x': len(attention_20x),
            'mean_5x': float(attention_5x.mean()) if len(attention_5x) > 0 else np.nan,
            'mean_20x': float(attention_20x.mean()) if len(attention_20x) > 0 else np.nan,
            'output_path': str(output_path)
        }
        
    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)}"
        print(f"Error processing {slide_name}: {error_msg}")
        print(traceback.format_exc())
        
        return {
            'slide': slide_name,
            'status': 'error',
            'error': error_msg,
            'tiles_5x': 0,
            'tiles_20x': 0,
            'output_path': ''
        }


def main():
    parser = argparse.ArgumentParser(description='Robust E2E_VIZ core functions for DeepHRD visualization')
    parser.add_argument('--slide_dir', type=str, required=True, help='Directory containing WSI slides')
    parser.add_argument('--pred_5x', type=str, required=True, help='Path to 5x predictions TSV')
    parser.add_argument('--pred_20x', type=str, required=True, help='Path to 20x predictions TSV')
    parser.add_argument('--name_map', type=str, required=True, help='Path to slideNumberToSampleName.txt')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
    parser.add_argument('--skip_existing', action='store_true', help='Skip slides that already have visualizations')
    parser.add_argument('--report_csv', type=str, default='visualization_report.csv', help='Path to save processing report')
    
    args = parser.parse_args()
    
    # Load slide mapping
    slide_mapping = {}
    try:
        with open(args.name_map, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split('\t')
                    if len(parts) == 2:
                        slide_mapping[parts[1]] = parts[0]
    except Exception as e:
        print(f"Error loading slide mapping: {e}")
        print("Proceeding without slide number mapping...")
    
    # Process each slide
    slide_files = sorted([f for f in Path(args.slide_dir).glob('*.svs')])
    
    if len(slide_files) == 0:
        print(f"No .svs files found in {args.slide_dir}")
        return
    
    print(f"Found {len(slide_files)} slides to process")
    
    # Results tracking
    results = []
    
    for idx, slide_path in enumerate(slide_files, 1):
        slide_name = slide_path.stem
        print(f"\n[{idx}/{len(slide_files)}] Processing {slide_name}...")
        
        # Find slide number
        slide_number = None
        for name_part, number in slide_mapping.items():
            if name_part in str(slide_path.name):
                slide_number = number
                break
        
        if not slide_number and len(slide_mapping) > 0:
            print(f"No mapping found for {slide_name}, using slide name as identifier...")
            slide_number = slide_name
        
        # Determine base_dir from slide path or predictions path
        base_dir = Path(args.slide_dir).parent if args.slide_dir else None
        
        result = visualize_deephrd_with_e2e_core(
            slide_path=slide_path,
            pred_5x_path=args.pred_5x,
            pred_20x_path=args.pred_20x,
            slide_number=slide_number,
            output_dir=args.output_dir,
            skip_existing=args.skip_existing,
            base_dir=base_dir
        )
        
        results.append(result)
    
    # Save report
    report_df = pd.DataFrame(results)
    report_path = Path(args.output_dir) / args.report_csv
    report_df.to_csv(report_path, index=False)
    
    # Print summary
    print("\n" + "="*60)
    print("PROCESSING SUMMARY")
    print("="*60)
    
    success_count = len(report_df[report_df['status'] == 'success'])
    error_count = len(report_df[report_df['status'] == 'error'])
    skipped_count = len(report_df[report_df['status'] == 'skipped'])
    
    print(f"Total slides processed: {len(slide_files)}")
    print(f"  Successful: {success_count}")
    print(f"  Errors: {error_count}")
    print(f"  Skipped (existing): {skipped_count}")
    
    if error_count > 0:
        print("\nSlides with errors:")
        error_df = report_df[report_df['status'] == 'error']
        for _, row in error_df.iterrows():
            print(f"  - {row['slide']}: {row['error']}")
    
    print(f"\nReport saved to: {report_path}")
    print(f"Visualizations saved to: {args.output_dir}")
    
    # Also save a detailed error log
    if error_count > 0:
        error_log_path = Path(args.output_dir) / 'error_log.txt'
        with open(error_log_path, 'w') as f:
            f.write(f"Error Log - {datetime.now()}\n")
            f.write("="*60 + "\n\n")
            for _, row in error_df.iterrows():
                f.write(f"Slide: {row['slide']}\n")
                f.write(f"Error: {row['error']}\n")
                f.write("-"*40 + "\n\n")
        print(f"Error log saved to: {error_log_path}")


if __name__ == "__main__":
    main()