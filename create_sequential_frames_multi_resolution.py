#!/usr/bin/env python3
"""
Create sequential frames for all slides in a directory - supports both 5x and 20x resolutions
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from PIL import Image
import openslide
import argparse

from use_easymsi_viz_core_robust import (
    create_attention_heatmap,
    create_slide_thumbnail,
    overlay_heatmap_on_slide,
    parse_deephrd_predictions,
    get_tile_parameters
)


def create_progressive_frames(
    slide_path,
    predictions_file,
    output_dir,
    slide_number,
    resolution='5x',  # Add resolution parameter
    tile_size=None,  # Will be set based on resolution
    downsample_factor=None,  # Will be set based on resolution
    thumbnail_size=(2000, 2000),
    colormap='seismic',
    alpha=0.5,
    max_frames=None,
    pure_heatmap=True  # Create pure heatmap without labels
):
    """
    Create sequential frames by progressively revealing attention weights.
    Supports both 5x and 20x resolutions.
    """
    
    # Set parameters based on resolution
    # If not manually specified, get automatic parameters based on objective power
    if tile_size is None or downsample_factor is None:
        # Try to get base_dir from slide path
        base_dir = Path(slide_path).parent.parent if slide_path else None
        auto_tile_size, auto_downsample = get_tile_parameters(slide_number, resolution, base_dir)
        
        if tile_size is None:
            tile_size = auto_tile_size
        if downsample_factor is None:
            downsample_factor = auto_downsample
    
    # Validate resolution
    if resolution not in ['5x', '20x']:
        raise ValueError(f"Unsupported resolution: {resolution}. Use '5x' or '20x'")
    
    print(f"  Resolution: {resolution}, tile_size: {tile_size}, downsample: {downsample_factor}")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Parse all predictions at once
    print(f"  Parsing predictions for slide {slide_number}...")
    full_attention_weights, tile_coordinates = parse_deephrd_predictions(
        predictions_file, slide_number
    )
    
    if len(full_attention_weights) == 0:
        print(f"  No predictions found for slide {slide_number}")
        return False
    
    print(f"  Found {len(full_attention_weights)} predictions")
    
    # Calculate global min/max for consistent normalization across all frames
    global_vmin = full_attention_weights.min()
    global_vmax = full_attention_weights.max()
    print(f"  Global normalization range: [{global_vmin:.3f}, {global_vmax:.3f}]")
    
    # Limit frames if requested
    if max_frames and len(full_attention_weights) > max_frames:
        print(f"  Limiting to {max_frames} frames (sampling evenly)")
        indices = np.linspace(0, len(full_attention_weights)-1, max_frames, dtype=int)
        sampled_attention = full_attention_weights[indices]
        sampled_coordinates = tile_coordinates.iloc[indices].reset_index(drop=True)
    else:
        sampled_attention = full_attention_weights
        sampled_coordinates = tile_coordinates
    
    # Get slide dimensions and create thumbnail
    print("  Loading slide and creating thumbnail...")
    slide = openslide.OpenSlide(str(slide_path))
    slide_dimensions = slide.dimensions
    slide.close()
    
    thumbnail = create_slide_thumbnail(slide_path, thumbnail_size)
    
    # Save frame 0 - just the thumbnail
    print("  Saving frame 0 (thumbnail only)...")
    if pure_heatmap:
        # Save pure thumbnail without any labels
        thumbnail.save(output_dir / 'frame_0000.png')
    else:
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        ax.set_title(f"Slide {slide_number} - Original ({resolution})", fontsize=14)
        ax.axis('off')
        ax.imshow(thumbnail)
        plt.tight_layout()
        plt.savefig(output_dir / 'frame_0000.png', dpi=200, bbox_inches='tight')
        plt.close()
    
    # Generate progressive frames
    print(f"  Generating {len(sampled_attention)} frames...")
    
    for i in range(len(sampled_attention)):
        # Use only the subset of data up to current index
        subset_attention = sampled_attention[:i+1]
        subset_coordinates = sampled_coordinates.iloc[:i+1]
        
        # Create heatmap with current subset
        heatmap = create_attention_heatmap(
            attention_weights=subset_attention,
            tile_coordinates=subset_coordinates,
            slide_dimensions=slide_dimensions,
            tile_size=tile_size,
            downsample_factor=downsample_factor,
            smooth=False,
            normalize=False
        )
        
        # Create overlay with fixed normalization
        overlay = overlay_heatmap_on_slide(
            heatmap=heatmap,
            slide_thumbnail=thumbnail,
            colormap=colormap,
            alpha=alpha,
            vmin=global_vmin,
            vmax=global_vmax
        )
        
        # Save frame
        if pure_heatmap:
            # Pure heatmap - no labels, text, or colorbar
            # Save directly as image without matplotlib decorations
            overlay_array = np.array(overlay)
            Image.fromarray(overlay_array).save(output_dir / f'frame_{i+1:04d}.png')
        else:
            # Original version with labels
            fig, ax = plt.subplots(1, 1, figsize=(10, 10))
            ax.set_title(f"{resolution} Progressive Heatmap: {i+1}/{len(sampled_attention)} tiles", fontsize=14)
            ax.axis('off')
            ax.imshow(overlay)
            
            # Add colorbar
            cmap = matplotlib.colormaps.get(colormap) if hasattr(matplotlib, 'colormaps') else cm.get_cmap(colormap)
            norm = Normalize(vmin=0, vmax=1)
            sm = cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('HRD Probability', rotation=270, labelpad=15)
            
            # Add current tile info (the last one added)
            current_prob = subset_attention[-1]
            current_x = subset_coordinates.iloc[-1]['x']
            current_y = subset_coordinates.iloc[-1]['y']
            info_text = f'{resolution} Tile {i+1}: prob={current_prob:.3f}, pos=({current_x},{current_y})'
            ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=10,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            plt.tight_layout()
            frame_path = output_dir / f'frame_{i+1:04d}.png'
            plt.savefig(frame_path, dpi=200, bbox_inches='tight')
            plt.close()
        
        # Progress update
        if (i + 1) % 20 == 0:
            print(f"    Saved frame {i+1}/{len(sampled_attention)}")
    
    # Save final comparison
    print("  Creating final comparison...")
    
    # Final overlay with all tiles (use full data, not sampled)
    final_heatmap = create_attention_heatmap(
        attention_weights=full_attention_weights,
        tile_coordinates=tile_coordinates,
        slide_dimensions=slide_dimensions,
        tile_size=tile_size,
        downsample_factor=downsample_factor,
        smooth=False,
        normalize=False
    )
    
    final_overlay = overlay_heatmap_on_slide(
        heatmap=final_heatmap,
        slide_thumbnail=thumbnail,
        colormap=colormap,
        alpha=alpha,
        vmin=global_vmin,
        vmax=global_vmax
    )
    
    if pure_heatmap:
        # Save pure final overlay without comparison
        final_overlay.save(output_dir / 'final_overlay.png')
    else:
        # Original comparison view
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        ax1.set_title("Original Slide Thumbnail", fontsize=14)
        ax1.axis('off')
        ax1.imshow(thumbnail)
        
        ax2.set_title(f"Complete {resolution} Heatmap ({len(full_attention_weights)} tiles)", fontsize=14)
        ax2.axis('off')
        ax2.imshow(final_overlay)
        
        cmap = matplotlib.colormaps.get(colormap) if hasattr(matplotlib, 'colormaps') else cm.get_cmap(colormap)
        sm = cm.ScalarMappable(cmap=cmap, norm=Normalize(vmin=0, vmax=1))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax2, fraction=0.046, pad=0.04)
        cbar.set_label('HRD Probability', rotation=270, labelpad=15)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'final_comparison.png', dpi=200, bbox_inches='tight')
        plt.close()
    
    print(f"  All frames saved to {output_dir}")
    print(f"  Total frames: {len(sampled_attention) + 1}")
    
    return True


def process_all_slides(
    slide_dir,
    predictions_file,
    output_base_dir,
    resolution='5x',
    name_map_file=None,
    max_frames_per_slide=None,
    custom_tile_size=None,
    custom_downsample=None,
    pure_heatmap=True
):
    """
    Process all slides in a directory for a given resolution
    """
    slide_dir = Path(slide_dir)
    output_base_dir = Path(output_base_dir)
    
    # Load slide name mapping if provided
    slide_mapping = {}
    if name_map_file and Path(name_map_file).exists():
        with open(name_map_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split('\t')
                    if len(parts) == 2:
                        # Map from slide name to number
                        slide_mapping[parts[1]] = parts[0]
        print(f"Loaded {len(slide_mapping)} slide mappings")
        print(f"Mappings: {slide_mapping}")
    
    # Find all slide files
    slide_files = sorted(list(slide_dir.glob('*.svs')))
    
    if not slide_files:
        print(f"No .svs files found in {slide_dir}")
        return
    
    print(f"Found {len(slide_files)} slides to process at {resolution} resolution")
    print("=" * 60)
    
    # Process each slide
    successful = 0
    failed = 0
    
    for idx, slide_path in enumerate(slide_files, 1):
        slide_name = slide_path.stem
        print(f"\n[{idx}/{len(slide_files)}] Processing {slide_name} at {resolution}...")
        
        # Find slide number from mapping
        slide_number = None
        
        # Check exact match first (without extension)
        for slide_base_name, number in slide_mapping.items():
            if slide_base_name in slide_name:
                slide_number = number
                break
        
        # If no mapping found, try to extract number from filename
        if not slide_number:
            # Try to extract pattern like "001" or similar
            import re
            match = re.search(r'(\d{3})', slide_name)
            if match:
                slide_number = match.group(1)
            else:
                # Use the slide name itself
                slide_number = slide_name
                print(f"  WARNING: No mapping found for {slide_name}, using name as ID")
        
        print(f"  Using slide number: {slide_number}")
        
        # Create output directory for this slide and resolution
        output_dir = output_base_dir / f"slide_{slide_number}_{resolution}_frames"
        
        try:
            success = create_progressive_frames(
                slide_path=slide_path,
                predictions_file=predictions_file,
                output_dir=output_dir,
                slide_number=slide_number,
                resolution=resolution,
                tile_size=custom_tile_size,
                downsample_factor=custom_downsample,
                colormap='seismic',
                max_frames=max_frames_per_slide,
                pure_heatmap=pure_heatmap
            )
            
            if success:
                successful += 1
            else:
                failed += 1
                
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    # Print summary
    print("\n" + "=" * 60)
    print(f"PROCESSING COMPLETE ({resolution})")
    print("=" * 60)
    print(f"Total slides: {len(slide_files)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Output saved to: {output_base_dir}")


def main():
    parser = argparse.ArgumentParser(description='Create sequential heatmap frames for slides at different resolutions')
    parser.add_argument('--slide_dir', type=str, required=True,
                       help='Directory containing WSI slides')
    parser.add_argument('--predictions', type=str, required=True,
                       help='Path to predictions TSV file')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for frames')
    parser.add_argument('--resolution', type=str, choices=['5x', '20x'], default='5x',
                       help='Resolution of tiles (5x or 20x)')
    parser.add_argument('--name_map', type=str, default=None,
                       help='Path to slide number mapping file')
    parser.add_argument('--max_frames', type=int, default=None,
                       help='Maximum frames per slide (for testing/speed)')
    parser.add_argument('--tile_size', type=int, default=None,
                       help='Custom tile size (default: 1024 for 5x, 256 for 20x)')
    parser.add_argument('--downsample', type=int, default=None,
                       help='Custom downsample factor (default: 64 for 5x, 8 for 20x)')
    parser.add_argument('--pure_heatmap', action='store_true',
                       help='Create pure heatmap frames without labels, text, or colorbars')
    parser.add_argument('--with_labels', dest='pure_heatmap', action='store_false',
                       help='Include labels, text, and colorbars in frames')
    parser.set_defaults(pure_heatmap=True)  # Default to pure heatmap
    
    args = parser.parse_args()
    
    process_all_slides(
        slide_dir=args.slide_dir,
        predictions_file=args.predictions,
        output_base_dir=args.output_dir,
        resolution=args.resolution,
        name_map_file=args.name_map,
        max_frames_per_slide=args.max_frames,
        custom_tile_size=args.tile_size,
        custom_downsample=args.downsample,
        pure_heatmap=args.pure_heatmap
    )


if __name__ == '__main__':
    main()