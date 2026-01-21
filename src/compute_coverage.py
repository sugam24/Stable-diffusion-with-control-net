"""
Compute per-class percentage area coverage from satellite image annotation masks.

Usage:
    1. Set DEFAULT_MASK_PATH below and run: python compute_coverage.py
    2. Or pass image path as argument: python compute_coverage.py <mask_image_path>
    3. Or import and call: from compute_coverage import compute_coverage
    
Example:
    python compute_coverage.py ../dataset/Annotation/annotated_masks/output_117.png
"""

import numpy as np
from PIL import Image
import argparse
import sys
import os


# ============================================================
# SET YOUR MASK IMAGE PATH HERE (change this as needed)
# ============================================================
DEFAULT_MASK_PATH = "../dataset/Annotation/annotated_masks/output_338.png"
# ============================================================


# Satellite imagery class color mapping (RGB values)
CLASS_COLORS = {
    (128, 0, 0): "River",              # Red
    (0, 0, 128): "Residential Area",   # Dark Blue
    (0, 128, 0): "Road",               # Green
    (128, 128, 0): "Forest",           # Yellow
    (0, 128, 128): "Unused Land",      # Light Blue/Cyan
    (128, 0, 128): "Agricultural Area", # Pink/Magenta
}


def compute_coverage(mask_path: str) -> dict:
    """
    Compute per-class coverage percentage for an annotation mask.
    
    Args:
        mask_path: Path to the annotation mask (PNG)
        
    Returns:
        Dictionary with class names and their coverage percentages
    """
    mask = Image.open(mask_path)
    mask_array = np.array(mask)
    
    # Handle different mask formats
    if len(mask_array.shape) == 2:
        mask_array = np.stack([mask_array] * 3, axis=-1)
    elif mask_array.shape[2] == 4:
        mask_array = mask_array[:, :, :3]
    
    total_pixels = mask_array.shape[0] * mask_array.shape[1]
    pixels_flat = mask_array.reshape(-1, 3)
    
    # Initialize all classes to 0%
    coverage = {name: 0.0 for name in CLASS_COLORS.values()}
    
    # Get unique colors and count pixels
    unique_colors = np.unique(pixels_flat, axis=0)
    
    for color in unique_colors:
        color_tuple = tuple(color)
        if color_tuple in CLASS_COLORS:
            class_name = CLASS_COLORS[color_tuple]
            matches = np.all(pixels_flat == color, axis=1)
            pixel_count = np.sum(matches)
            coverage[class_name] = round((pixel_count / total_pixels) * 100, 2)
    
    return coverage


def main():
    parser = argparse.ArgumentParser(
        description="Compute per-class percentage area coverage from annotation mask"
    )
    parser.add_argument(
        "mask_path",
        type=str,
        nargs="?",
        default=DEFAULT_MASK_PATH,
        help="Path to the annotation mask image (PNG). If not provided, uses DEFAULT_MASK_PATH."
    )
    
    args = parser.parse_args()
    
    # Resolve relative path from script directory if needed
    mask_path = args.mask_path
    if not os.path.isabs(mask_path):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        mask_path = os.path.join(script_dir, mask_path)
    
    try:
        coverage = compute_coverage(mask_path)
        
        print("\n" + "=" * 50)
        print("  AREA COVERAGE (%)")
        print("=" * 50)
        
        # Sort by percentage descending
        sorted_items = sorted(coverage.items(), key=lambda x: -x[1])
        
        for class_name, percentage in sorted_items:
            bar_length = int(percentage / 2)
            bar = "█" * bar_length + "░" * (50 - bar_length)
            print(f"  {class_name:<20s}: {percentage:6.2f}% |{bar[:25]}|")
        
        print("=" * 50 + "\n")
        
    except FileNotFoundError:
        print(f"Error: File not found: {args.mask_path}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
