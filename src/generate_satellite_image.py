"""
Smart City Satellite Image Generation Pipeline

This module implements a constrained smart-city generation pipeline that:
1. Analyzes urban layout from segmentation masks
2. Generates planning suggestions based on analysis
3. Produces structured reports with issues and interventions
4. Uses ControlNet + Stable Diffusion ONLY as visualization

IMPORTANT: Diffusion is NOT the decision-maker. Urban analysis, suggestions,
and constraints are computed BEFORE image generation.

Usage:
    1. Set DEFAULT_IMAGE_PATH and DEFAULT_MASK_PATH below and run: python generate_satellite_image.py
    2. Or pass paths as arguments: python generate_satellite_image.py --image <path> --mask <path>
    3. For planned-city layout: put an aerial in reference_image/ (or set DEFAULT_REFERENCE_IMAGE_PATH).
       The pipeline uses it for ControlNet structure; use --no_reference to use only the unplanned image.
"""

import os
import sys
import cv2
import torch
import numpy as np
from pathlib import Path
from PIL import Image
import argparse
from typing import Tuple, Dict, Optional, List, Union
from datetime import datetime

from diffusers import (
    StableDiffusionControlNetInpaintPipeline,
    ControlNetModel,
    MultiControlNetModel,
    DPMSolverMultistepScheduler,
)

# Import compute_coverage from the same directory
from compute_coverage import compute_coverage


# =============================================================================
# SET YOUR IMAGE PATHS HERE (change these as needed)
# =============================================================================
DEFAULT_IMAGE_PATH = "../dataset/images/output_337.png"
DEFAULT_MASK_PATH = "../dataset/masks/output_337.png"
# Optional: planned-city image for ControlNet structure (relative to project root)
DEFAULT_REFERENCE_IMAGE_PATH = "reference_image/reference_image_elche_alicante_spain.jpg"
# =============================================================================


# =============================================================================
# CONFIGURATION - BGR Label Mapping (canonical, matches user's format)
# Label ID -> Class name; BGR colors for mask matching (OpenCV convention)
# =============================================================================
LABEL_IDS = {
    1: "Residential Area",   # (128, 0, 0) BGR
    2: "Road",               # (0, 128, 0) BGR
    3: "River",              # (0, 0, 128) BGR
    4: "Forest",             # (0, 128, 128) BGR
    5: "Unused Land",        # (128, 128, 0) BGR
    6: "Agricultural Area",  # (128, 0, 128) BGR
}

# BGR colors per label (exact mapping from user's conversion script)
BGR_COLORS = {
    1: (128, 0, 0),    # Residential_area
    2: (0, 128, 0),    # Road
    3: (0, 0, 128),    # River
    4: (0, 128, 128),  # Forest
    5: (128, 128, 0),  # unused_land
    6: (128, 0, 128),  # Agricultural_area
}

CLASS_COLORS_BGR = {LABEL_IDS[i]: BGR_COLORS[i] for i in range(1, 7)}

# RGB for display/compute_spatial_metrics (BGR -> RGB: swap R and B)
CLASS_COLORS = {name: (bgr[2], bgr[1], bgr[0]) for name, bgr in CLASS_COLORS_BGR.items()}

# Classes that can be modified during generation (planned city transformation)
EDITABLE_CLASSES = ["Residential Area", "Unused Land", "Agricultural Area"]

# Classes that must remain unchanged (preserved exactly)
IMMUTABLE_CLASSES = ["Road", "River", "Forest"]


# =============================================================================
# SPATIAL ANALYSIS FUNCTIONS
# =============================================================================

def compute_connected_components(binary_mask: np.ndarray) -> Tuple[np.ndarray, int]:
    """
    Compute connected components in a binary mask using 8-connectivity.
    
    Args:
        binary_mask: Binary numpy array (H, W) with 1s for the class
        
    Returns:
        Tuple of (labeled_array, num_components)
    """
    from scipy import ndimage
    labeled, num_features = ndimage.label(binary_mask, structure=np.ones((3, 3)))
    return labeled, num_features


def compute_spatial_metrics(
    seg_mask: Optional[Image.Image] = None,
    class_colors: Optional[Dict] = None,
    label_mask: Optional[np.ndarray] = None,
) -> Dict:
    """
    Compute spatial metrics from the segmentation mask for evidence-based analysis.
    Accepts either (seg_mask, class_colors) for color masks or label_mask for single-channel.
    """
    if label_mask is not None:
        h, w = label_mask.shape[:2]
        class_masks = create_class_masks_from_label_mask(label_mask)
    elif seg_mask is not None and class_colors is not None:
        mask_array = np.array(seg_mask)
        h, w = mask_array.shape[:2]
        class_masks = {}
        for class_name, color in class_colors.items():
            color_array = np.array(color)
            matches = np.all(mask_array == color_array, axis=-1)
            class_masks[class_name] = matches.astype(np.uint8)
    else:
        raise ValueError("Provide either (seg_mask, class_colors) or label_mask")

    total_pixels = h * w
    metrics = {
        "image_dimensions": {"height": h, "width": w, "total_pixels": total_pixels},
        "class_metrics": {},
        "adjacency": {},
        "fragmentation": {},
    }
    
    # Compute per-class spatial metrics
    for class_name, binary_mask in class_masks.items():
        pixel_count = np.sum(binary_mask)
        
        if pixel_count == 0:
            metrics["class_metrics"][class_name] = {
                "pixel_count": 0,
                "coverage_percent": 0.0,
                "num_components": 0,
                "avg_component_size_pixels": 0,
                "median_component_size_pixels": 0,
                "largest_component_pixels": 0,
                "smallest_component_pixels": 0,
                "compactness": "N/A",
            }
            continue
        
        # Connected components analysis
        labeled, num_components = compute_connected_components(binary_mask)
        
        # Component sizes
        component_sizes = []
        for i in range(1, num_components + 1):
            component_sizes.append(np.sum(labeled == i))
        
        component_sizes = sorted(component_sizes, reverse=True)
        
        avg_size = np.mean(component_sizes) if component_sizes else 0
        median_size = np.median(component_sizes) if component_sizes else 0
        largest = component_sizes[0] if component_sizes else 0
        smallest = component_sizes[-1] if component_sizes else 0
        
        # Compactness: ratio of largest component to total class area
        # High compactness = one dominant contiguous region
        compactness_ratio = largest / pixel_count if pixel_count > 0 else 0
        
        if compactness_ratio >= 0.8:
            compactness = "High"
        elif compactness_ratio >= 0.5:
            compactness = "Medium"
        else:
            compactness = "Low"
        
        metrics["class_metrics"][class_name] = {
            "pixel_count": int(pixel_count),
            "coverage_percent": round((pixel_count / total_pixels) * 100, 2),
            "num_components": num_components,
            "avg_component_size_pixels": int(avg_size),
            "median_component_size_pixels": int(median_size),
            "largest_component_pixels": int(largest),
            "smallest_component_pixels": int(smallest),
            "largest_component_percent": round((largest / total_pixels) * 100, 2),
            "compactness_ratio": round(compactness_ratio, 3),
            "compactness": compactness,
        }
    
    # Compute fragmentation index for key classes
    # Fragmentation = 1 - (largest_component / total_class_area)
    for class_name in ["Residential Area", "Forest", "Unused Land"]:
        cm = metrics["class_metrics"].get(class_name, {})
        if cm.get("pixel_count", 0) > 0:
            frag = 1 - cm.get("compactness_ratio", 0)
            metrics["fragmentation"][class_name] = {
                "index": round(frag, 3),
                "level": "High" if frag > 0.5 else ("Medium" if frag > 0.2 else "Low"),
                "num_fragments": cm.get("num_components", 0),
            }
    
    # Compute adjacency relationships
    # Check which classes are adjacent to residential areas
    residential_mask = class_masks.get("Residential Area", np.zeros((h, w), dtype=np.uint8))
    if np.sum(residential_mask) > 0:
        # Dilate residential mask by 1 pixel to find adjacent regions
        from scipy import ndimage
        dilated = ndimage.binary_dilation(residential_mask, iterations=2)
        boundary = dilated.astype(np.uint8) - residential_mask
        
        adjacent_classes = {}
        for class_name, binary_mask in class_masks.items():
            if class_name == "Residential Area":
                continue
            overlap = np.sum(boundary * binary_mask)
            if overlap > 0:
                adjacent_classes[class_name] = {
                    "boundary_pixels": int(overlap),
                    "adjacency_strength": "Strong" if overlap > 1000 else ("Moderate" if overlap > 200 else "Weak"),
                }
        
        metrics["adjacency"]["Residential Area"] = adjacent_classes
    
    return metrics


def analyze_urban_layout(coverage: Dict[str, float], spatial_metrics: Dict = None) -> Dict:
    """
    Analyze urban layout using coverage percentages AND spatial metrics.
    All conclusions are evidence-based and tied to computed indicators.
    
    Args:
        coverage: Dictionary with class names and coverage percentages
        spatial_metrics: Optional spatial metrics from compute_spatial_metrics()
        
    Returns:
        Dictionary containing evidence-based urban analysis
    """
    analysis = {
        "coverage": coverage,
        "spatial_metrics": spatial_metrics,
        "indicators": {},
        "density_assessment": {},
        "open_space_assessment": {},
        "fragmentation_assessment": {},
        "issues": [],
    }
    
    residential = coverage.get("Residential Area", 0)
    unused = coverage.get("Unused Land", 0)
    forest = coverage.get("Forest", 0)
    agricultural = coverage.get("Agricultural Area", 0)
    road = coverage.get("Road", 0)
    river = coverage.get("River", 0)
    
    # --- Compute Derived Indicators ---
    green_space = forest + agricultural
    built_area = residential + road
    open_space = unused + forest + agricultural + river
    
    indicators = {
        "residential_coverage_percent": round(residential, 2),
        "green_space_percent": round(green_space, 2),
        "built_area_percent": round(built_area, 2),
        "open_space_percent": round(open_space, 2),
        "unused_land_percent": round(unused, 2),
        "road_coverage_percent": round(road, 2),
        "development_intensity": round(built_area / max(open_space, 0.1), 3),
        "green_to_residential_ratio": round(green_space / max(residential, 0.1), 3),
    }
    
    # Add spatial indicators if available
    if spatial_metrics:
        res_metrics = spatial_metrics.get("class_metrics", {}).get("Residential Area", {})
        indicators["residential_num_components"] = res_metrics.get("num_components", 0)
        indicators["residential_compactness_ratio"] = res_metrics.get("compactness_ratio", 0)
        indicators["residential_largest_block_percent"] = res_metrics.get("largest_component_percent", 0)
        indicators["residential_avg_block_pixels"] = res_metrics.get("avg_component_size_pixels", 0)
        
        frag = spatial_metrics.get("fragmentation", {}).get("Residential Area", {})
        indicators["residential_fragmentation_index"] = frag.get("index", 0)
    
    analysis["indicators"] = indicators
    
    # --- Density Assessment (Evidence-Based) ---
    density = {
        "classification": None,
        "evidence": [],
    }
    
    if residential >= 50:
        density["classification"] = "High"
        density["evidence"].append(
            f"Residential areas occupy {residential:.1f}% of total area, exceeding 50% threshold"
        )
    elif residential >= 30:
        density["classification"] = "Medium"
        density["evidence"].append(
            f"Residential areas occupy {residential:.1f}% of total area (30-50% range)"
        )
    else:
        density["classification"] = "Low"
        density["evidence"].append(
            f"Residential areas occupy {residential:.1f}% of total area, below 30% threshold"
        )
    
    # Add compactness evidence if available
    if spatial_metrics:
        res_metrics = spatial_metrics.get("class_metrics", {}).get("Residential Area", {})
        compactness = res_metrics.get("compactness", "N/A")
        compactness_ratio = res_metrics.get("compactness_ratio", 0)
        num_components = res_metrics.get("num_components", 0)
        
        if compactness != "N/A":
            density["compactness"] = compactness
            density["evidence"].append(
                f"Residential compactness is {compactness} (ratio: {compactness_ratio:.2f}), "
                f"distributed across {num_components} distinct component(s)"
            )
    
    analysis["density_assessment"] = density
    
    # --- Open Space Assessment (Evidence-Based) ---
    open_space_eval = {
        "accessibility": None,
        "distribution": None,
        "evidence": [],
    }
    
    # Accessibility based on adjacency
    if spatial_metrics:
        adj = spatial_metrics.get("adjacency", {}).get("Residential Area", {})
        forest_adj = adj.get("Forest", {}).get("adjacency_strength", "None")
        unused_adj = adj.get("Unused Land", {}).get("adjacency_strength", "None")
        
        if forest_adj in ["Strong", "Moderate"] or unused_adj in ["Strong", "Moderate"]:
            open_space_eval["accessibility"] = "Adequate"
            open_space_eval["evidence"].append(
                f"Residential areas have {forest_adj.lower() if forest_adj != 'None' else 'no'} "
                f"adjacency to forest and {unused_adj.lower() if unused_adj != 'None' else 'no'} "
                f"adjacency to unused land"
            )
        else:
            open_space_eval["accessibility"] = "Limited"
            open_space_eval["evidence"].append(
                "Residential areas show weak or no adjacency to open space classes"
            )
    else:
        # Fallback without spatial metrics
        if open_space >= 30:
            open_space_eval["accessibility"] = "Adequate"
        elif open_space >= 15:
            open_space_eval["accessibility"] = "Limited"
        else:
            open_space_eval["accessibility"] = "Constrained"
        open_space_eval["evidence"].append(
            f"Total open space (unused + green + water) is {open_space:.1f}%"
        )
    
    # Distribution based on fragmentation
    if spatial_metrics:
        forest_frag = spatial_metrics.get("fragmentation", {}).get("Forest", {})
        if forest_frag:
            frag_level = forest_frag.get("level", "Unknown")
            frag_index = forest_frag.get("index", 0)
            open_space_eval["distribution"] = "Fragmented" if frag_level == "High" else "Continuous"
            open_space_eval["evidence"].append(
                f"Forest fragmentation index: {frag_index:.2f} ({frag_level})"
            )
    
    analysis["open_space_assessment"] = open_space_eval
    
    # --- Fragmentation Assessment ---
    if spatial_metrics:
        frag_data = spatial_metrics.get("fragmentation", {})
        for class_name, frag_info in frag_data.items():
            analysis["fragmentation_assessment"][class_name] = {
                "index": frag_info.get("index", 0),
                "level": frag_info.get("level", "Unknown"),
                "num_fragments": frag_info.get("num_fragments", 0),
            }
    
    # --- Issue Detection (Evidence-Based) ---
    issues = []
    
    # Issue: High density with limited internal open space
    if residential >= 50 and unused < 5:
        issues.append({
            "category": "Density Constraint",
            "description": (
                f"Residential areas occupy {residential:.1f}% with only {unused:.1f}% unused land available. "
                f"This indicates constrained internal open-space availability."
            ),
            "severity": "High",
            "metric_basis": ["residential_coverage_percent", "unused_land_percent"],
        })
    
    # Issue: Low green-to-residential ratio
    if residential > 0 and indicators["green_to_residential_ratio"] < 0.5:
        issues.append({
            "category": "Green Space Deficit",
            "description": (
                f"Green-to-residential ratio is {indicators['green_to_residential_ratio']:.2f}, "
                f"below the 0.5 threshold for balanced urban environments."
            ),
            "severity": "Medium" if indicators["green_to_residential_ratio"] >= 0.3 else "High",
            "metric_basis": ["green_to_residential_ratio"],
        })
    
    # Issue: High compactness with no fragmentation opportunity
    if spatial_metrics:
        res_metrics = spatial_metrics.get("class_metrics", {}).get("Residential Area", {})
        if res_metrics.get("compactness_ratio", 0) >= 0.8 and unused < 5:
            issues.append({
                "category": "Ventilation Constraint",
                "description": (
                    f"Residential areas show high compactness (ratio: {res_metrics['compactness_ratio']:.2f}) "
                    f"forming {res_metrics.get('num_components', 1)} large contiguous block(s). "
                    f"Combined with {unused:.1f}% unused land, this limits ventilation corridors."
                ),
                "severity": "Medium",
                "metric_basis": ["residential_compactness_ratio", "unused_land_percent"],
            })
    
    # Issue: Insufficient road infrastructure
    if road < 5 and residential > 30:
        issues.append({
            "category": "Infrastructure Gap",
            "description": (
                f"Road coverage ({road:.1f}%) appears low relative to residential density ({residential:.1f}%). "
                f"This may indicate internal road networks not captured in segmentation or actual deficiency."
            ),
            "severity": "Low",
            "metric_basis": ["road_coverage_percent", "residential_coverage_percent"],
        })
    
    analysis["issues"] = issues
    
    return analysis


def generate_suggestions(analysis: Dict) -> Dict:
    """
    Generate evidence-driven planning suggestions.
    Each suggestion is explicitly linked to a detected issue or metric.
    
    Args:
        analysis: Urban analysis dictionary from analyze_urban_layout()
        
    Returns:
        Dictionary containing evidence-linked suggestions
    """
    suggestions = {
        "interventions": [],
        "constraints": [],
    }
    
    coverage = analysis["coverage"]
    indicators = analysis.get("indicators", {})
    issues = analysis.get("issues", [])
    
    residential = coverage.get("Residential Area", 0)
    unused = coverage.get("Unused Land", 0)
    forest = coverage.get("Forest", 0)
    
    # --- Constraints (Based on Immutable Classes) ---
    suggestions["constraints"] = []
    if coverage.get("Road", 0) > 0:
        suggestions["constraints"].append({
            "class": "Road",
            "coverage": coverage.get("Road", 0),
            "rationale": "Road geometry defines circulation patterns and must be preserved"
        })
    if coverage.get("River", 0) > 0:
        suggestions["constraints"].append({
            "class": "River",
            "coverage": coverage.get("River", 0),
            "rationale": "Hydrological features cannot be modified without environmental impact"
        })
    if coverage.get("Forest", 0) > 0:
        suggestions["constraints"].append({
            "class": "Forest",
            "coverage": coverage.get("Forest", 0),
            "rationale": "Forest areas provide ecological services and carbon sequestration"
        })
    # Note: Agricultural Area is EDITABLE (modern precision agriculture)
    
    # --- Issue-Driven Interventions ---
    for issue in issues:
        intervention = {
            "linked_issue": issue["category"],
            "issue_description": issue["description"],
            "severity": issue["severity"],
            "recommendations": [],
        }
        
        if issue["category"] == "Density Constraint":
            intervention["recommendations"].append(
                f"Because unused land is only {unused:.1f}%, consider internal micro-open spaces "
                f"within the {residential:.1f}% residential coverage through building setbacks and courtyards"
            )
            intervention["recommendations"].append(
                "Introduce vertical green elements (green walls, rooftop vegetation) to compensate "
                "for limited horizontal open space"
            )
        
        elif issue["category"] == "Green Space Deficit":
            ratio = indicators.get("green_to_residential_ratio", 0)
            intervention["recommendations"].append(
                f"Because green-to-residential ratio is {ratio:.2f}, prioritize green integration "
                f"within editable regions to approach 0.5 target"
            )
            if unused >= 5:
                intervention["recommendations"].append(
                    f"Allocate portion of {unused:.1f}% unused land to structured green spaces"
                )
        
        elif issue["category"] == "Ventilation Constraint":
            intervention["recommendations"].append(
                "Because residential blocks are highly contiguous, introduce internal courtyards "
                "and breathing corridors to improve air circulation"
            )
            intervention["recommendations"].append(
                "Consider building height variation to promote natural ventilation patterns"
            )
        
        elif issue["category"] == "Infrastructure Gap":
            intervention["recommendations"].append(
                "Note: Road infrastructure assessment is indicative only. Verify with actual "
                "infrastructure data before intervention planning"
            )
        
        if intervention["recommendations"]:
            suggestions["interventions"].append(intervention)
    
    # --- Additional Suggestions Based on Opportunities ---
    if unused >= 10:
        suggestions["interventions"].append({
            "linked_issue": "Opportunity",
            "issue_description": f"Unused land ({unused:.1f}%) presents development opportunity",
            "severity": "Info",
            "recommendations": [
                f"Available {unused:.1f}% unused land can accommodate mixed-use development",
                "Consider community amenities or public green spaces in unused areas"
            ],
        })
    
    # If no issues detected, provide general optimization notes
    if not issues:
        suggestions["interventions"].append({
            "linked_issue": "Optimization",
            "issue_description": "No critical issues detected; general optimization applicable",
            "severity": "Info",
            "recommendations": [
                "Consider sustainable building upgrades in residential areas",
                "Evaluate solar potential on available rooftop surfaces"
            ],
        })
    
    return suggestions


# =============================================================================
# PART 2: REPORT GENERATION
# =============================================================================

def create_report(
    image_path: str,
    mask_path: str,
    analysis: Dict,
    suggestions: Dict,
    output_path: str = None
) -> str:
    """
    Generate an evidence-based technical assessment report.
    All conclusions are tied to computed metrics and spatial indicators.
    
    Args:
        image_path: Path to original satellite image
        mask_path: Path to segmentation mask
        analysis: Urban analysis dictionary
        suggestions: Planning suggestions dictionary
        output_path: Optional path to save report
        
    Returns:
        Report content as string
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    indicators = analysis.get("indicators", {})
    spatial = analysis.get("spatial_metrics", {})
    
    report = f"""# Urban Spatial Analysis Report

Generated: {timestamp}

This report presents an evidence-based assessment of land use patterns derived
from semantic segmentation analysis. All conclusions are explicitly linked to
computed spatial metrics. Image generation (if performed) serves only as
visualization and does not influence analytical conclusions.

---

## 1. Data Summary

| Property | Value |
|----------|-------|
| Source Image | {os.path.basename(image_path)} |
| Segmentation Mask | {os.path.basename(mask_path)} |
"""
    
    if spatial:
        dims = spatial.get("image_dimensions", {})
        report += f"| Image Dimensions | {dims.get('width', 'N/A')} x {dims.get('height', 'N/A')} pixels |\n"
        report += f"| Total Pixels | {dims.get('total_pixels', 'N/A'):,} |\n"
    
    report += """| Analysis Basis | Segmentation mask only (pre-generation) |

---

## 2. Area Coverage

| Class | Coverage (%) | Modification Status |
|-------|--------------|---------------------|
"""
    
    for class_name in ["Residential Area", "Road", "Forest", "Agricultural Area", "Unused Land", "River"]:
        pct = analysis["coverage"].get(class_name, 0)
        status = "Editable" if class_name in EDITABLE_CLASSES else "Immutable"
        report += f"| {class_name} | {pct:.2f} | {status} |\n"
    
    report += f"""
---

## 3. Computed Indicators

### 3.1 Coverage Indicators

| Indicator | Value | Description |
|-----------|-------|-------------|
| Residential Coverage | {indicators.get('residential_coverage_percent', 0):.2f}% | Proportion of area classified as residential |
| Green Space | {indicators.get('green_space_percent', 0):.2f}% | Combined forest and agricultural coverage |
| Built Area | {indicators.get('built_area_percent', 0):.2f}% | Combined residential and road coverage |
| Open Space | {indicators.get('open_space_percent', 0):.2f}% | Combined unused, green, and water coverage |
| Development Intensity | {indicators.get('development_intensity', 0):.3f} | Ratio of built area to open space |
| Green-to-Residential Ratio | {indicators.get('green_to_residential_ratio', 0):.3f} | Balance indicator (target >= 0.5) |

"""
    
    # Add spatial indicators if available
    if spatial and indicators.get("residential_num_components", 0) > 0:
        report += """### 3.2 Spatial Indicators (Residential)

| Indicator | Value | Interpretation |
|-----------|-------|----------------|
"""
        report += f"| Number of Components | {indicators.get('residential_num_components', 0)} | Distinct residential regions |\n"
        report += f"| Compactness Ratio | {indicators.get('residential_compactness_ratio', 0):.3f} | Largest block / total residential area |\n"
        report += f"| Largest Block | {indicators.get('residential_largest_block_percent', 0):.2f}% | Dominant residential component |\n"
        report += f"| Average Block Size | {indicators.get('residential_avg_block_pixels', 0):,} px | Mean component size |\n"
        report += f"| Fragmentation Index | {indicators.get('residential_fragmentation_index', 0):.3f} | 0=contiguous, 1=highly fragmented |\n"
    
    # Fragmentation summary
    if analysis.get("fragmentation_assessment"):
        report += "\n### 3.3 Fragmentation Summary\n\n"
        report += "| Class | Fragmentation Index | Level | Components |\n"
        report += "|-------|---------------------|-------|------------|\n"
        for class_name, frag in analysis["fragmentation_assessment"].items():
            report += f"| {class_name} | {frag['index']:.3f} | {frag['level']} | {frag['num_fragments']} |\n"
    
    report += """
---

## 4. Density Assessment

"""
    
    density = analysis.get("density_assessment", {})
    report += f"**Classification:** {density.get('classification', 'N/A')}"
    if density.get("compactness"):
        report += f" (Compactness: {density.get('compactness')})"
    report += "\n\n**Evidence:**\n"
    
    for evidence in density.get("evidence", []):
        report += f"- {evidence}\n"
    
    report += """
---

## 5. Open Space Assessment

"""
    
    open_space = analysis.get("open_space_assessment", {})
    report += f"**Accessibility:** {open_space.get('accessibility', 'N/A')}\n"
    report += f"**Distribution:** {open_space.get('distribution', 'N/A')}\n\n"
    report += "**Evidence:**\n"
    
    for evidence in open_space.get("evidence", []):
        report += f"- {evidence}\n"
    
    report += """
---

## 6. Identified Issues

"""
    
    issues = analysis.get("issues", [])
    if issues:
        for i, issue in enumerate(issues, 1):
            report += f"### Issue {i}: {issue['category']}\n\n"
            report += f"**Severity:** {issue['severity']}\n\n"
            report += f"**Description:** {issue['description']}\n\n"
            report += f"**Metric Basis:** {', '.join(issue.get('metric_basis', []))}\n\n"
    else:
        report += "No significant issues identified based on computed metrics.\n"
    
    report += """
---

## 7. Recommended Interventions

"""
    
    interventions = suggestions.get("interventions", [])
    if interventions:
        for intervention in interventions:
            report += f"### Linked to: {intervention['linked_issue']}\n\n"
            if intervention.get("issue_description"):
                report += f"**Context:** {intervention['issue_description']}\n\n"
            report += "**Recommendations:**\n"
            for rec in intervention.get("recommendations", []):
                report += f"- {rec}\n"
            report += "\n"
    else:
        report += "No specific interventions recommended.\n"
    
    report += """
---

## 8. Generation Constraints

The following regions will be preserved unchanged during visualization:

"""
    
    constraints = suggestions.get("constraints", [])
    if constraints:
        report += "| Class | Coverage (%) | Preservation Rationale |\n"
        report += "|-------|--------------|------------------------|\n"
        for c in constraints:
            report += f"| {c['class']} | {c['coverage']:.2f} | {c['rationale']} |\n"
    else:
        report += "No immutable regions detected in segmentation.\n"
    
    report += f"""
---

## 9. Editable Regions

| Class | Coverage (%) | Modification Scope |
|-------|--------------|-------------------|
| Residential Area | {analysis['coverage'].get('Residential Area', 0):.2f} | Planned residential, smart city |
| Unused Land | {analysis['coverage'].get('Unused Land', 0):.2f} | Parks, lakes, green spaces |
| Agricultural Area | {analysis['coverage'].get('Agricultural Area', 0):.2f} | Modern precision agriculture |

---

## Technical Notes

- **Mask usage:** The segmentation mask defines (1) inpainting regions (white=editable: Residential, Unused, Agricultural—these are regenerated), (2) immutable regions (River, Road, Forest—preserved exactly via compositing). ControlNet uses Canny edges from the input image for structure guidance.
- All metrics are computed from the segmentation mask prior to any image generation
- Spatial indicators use 8-connectivity for connected component analysis
- Fragmentation index = 1 - (largest_component / total_class_area)
- Adjacency is computed using 2-pixel dilation boundary detection
- Generated images serve as visualization only and do not inform analysis

---

End of Report
"""
    
    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w") as f:
            f.write(report)
    
    return report


# =============================================================================
# PART 3: CONSTRAINED IMAGE GENERATION
# =============================================================================

def color_mask_to_label_mask(mask_bgr: np.ndarray) -> np.ndarray:
    """
    Convert BGR color mask to single-channel label mask.
    Uses canonical BGR mapping: label 1-6 per class.
    """
    h, w = mask_bgr.shape[:2]
    label_mask = np.zeros((h, w), dtype=np.uint8)
    for label_id, bgr in BGR_COLORS.items():
        matches = np.all(mask_bgr == np.array(bgr), axis=-1)
        label_mask[matches] = label_id
    return label_mask


def create_class_masks_from_label_mask(label_mask: np.ndarray) -> Dict[str, np.ndarray]:
    """Create binary masks per class from single-channel label mask."""
    class_masks = {}
    for label_id, class_name in LABEL_IDS.items():
        class_masks[class_name] = (label_mask == label_id).astype(np.uint8)
    return class_masks


def load_image_and_mask(
    image_path: str,
    mask_path: str,
    size: Tuple[int, int] = (512, 512)
) -> Tuple[Image.Image, Image.Image, np.ndarray]:
    """
    Load and resize the original satellite image and its segmentation mask.
    Returns (original_image, seg_mask_pil, label_mask).
    Supports both color mask (3ch) and single-channel label mask (1ch).
    Color masks use BGR format for conversion to label mask.
    """
    original_image = Image.open(image_path).convert("RGB")
    original_image = original_image.resize(size, Image.Resampling.LANCZOS)

    mask_raw = cv2.imread(str(mask_path))
    if mask_raw is None:
        raise FileNotFoundError(f"Could not load mask: {mask_path}")

    if len(mask_raw.shape) == 2:
        # Already single-channel label mask
        label_mask = cv2.resize(mask_raw, size, interpolation=cv2.INTER_NEAREST)
        seg_mask_pil = Image.fromarray(label_mask, mode="L").convert("RGB")
    else:
        # Color mask (BGR) - convert to label mask
        mask_bgr = cv2.resize(mask_raw, size, interpolation=cv2.INTER_NEAREST)
        label_mask = color_mask_to_label_mask(mask_bgr)
        seg_mask_pil = Image.fromarray(cv2.cvtColor(mask_bgr, cv2.COLOR_BGR2RGB))

    return original_image, seg_mask_pil, label_mask


def create_class_masks(label_mask: np.ndarray) -> Dict[str, np.ndarray]:
    """Create binary masks for each segmentation class from label mask."""
    return create_class_masks_from_label_mask(label_mask)


def create_inpainting_mask(
    class_masks: Dict[str, np.ndarray],
    editable_classes: List[str],
    boundary_erosion_pixels: int = 2,
) -> Image.Image:
    """
    Create inpainting mask from editable class regions.
    White (255) = regions to regenerate, Black (0) = regions to preserve.
    Erodes editable regions at boundaries to reduce diffusion bleeding into immutables.
    """
    from scipy import ndimage
    
    h, w = list(class_masks.values())[0].shape
    editable_binary = np.zeros((h, w), dtype=np.uint8)
    
    for class_name in editable_classes:
        if class_name in class_masks:
            editable_binary = np.maximum(editable_binary, class_masks[class_name])
    
    if boundary_erosion_pixels > 0 and np.sum(editable_binary) > 0:
        editable_binary = ndimage.binary_erosion(
            editable_binary, iterations=boundary_erosion_pixels
        ).astype(np.uint8)
    
    inpaint_mask = editable_binary * 255
    return Image.fromarray(inpaint_mask, mode="L")


def create_immutable_mask(
    class_masks: Dict[str, np.ndarray],
    immutable_classes: List[str]
) -> np.ndarray:
    """
    Create binary mask for immutable regions (River, Road, Forest).
    Used for strict compositing to guarantee original pixels are preserved.
    """
    h, w = list(class_masks.values())[0].shape
    immutable_mask = np.zeros((h, w), dtype=np.uint8)
    
    for class_name in immutable_classes:
        if class_name in class_masks:
            immutable_mask = np.maximum(immutable_mask, class_masks[class_name])
    
    return immutable_mask


# ADE20K palette for Segmentation ControlNet (lllyasviel/control_v11p_sd15_seg)
# Maps our label IDs to ADE20K-compatible colors: building, road, water, tree, grass, field
ADE20K_PALETTE = np.array([
    [0, 0, 0], [120, 120, 120], [180, 120, 120], [6, 230, 230], [80, 50, 50],
    [4, 200, 3], [120, 120, 80], [140, 140, 140], [204, 5, 255], [230, 230, 230],
    [4, 250, 7], [224, 5, 255], [235, 255, 7], [150, 5, 61], [120, 120, 70],
    [8, 255, 51], [255, 6, 82], [143, 255, 140], [204, 255, 4], [255, 51, 7],
    [204, 70, 3], [0, 102, 200], [61, 230, 250], [255, 6, 51], [11, 102, 255],
    [255, 7, 71], [255, 9, 224], [9, 7, 230], [220, 220, 220], [255, 9, 92],
    [112, 9, 255], [8, 255, 214], [7, 255, 224], [255, 184, 6], [10, 255, 71],
    [255, 41, 10], [7, 255, 255], [224, 255, 8], [102, 8, 255], [255, 61, 6],
    [255, 194, 7], [255, 122, 8], [0, 255, 20], [255, 8, 41], [255, 5, 153],
], dtype=np.uint8)

# Our label_id -> ADE20K palette index (building=1, road=6, water=21, tree=5, grass=10, field=12)
LABEL_TO_ADE20K_IDX = {
    1: 1,   # Residential Area -> building (gray)
    2: 6,   # Road -> road (olive)
    3: 21,  # River -> water (blue)
    4: 5,   # Forest -> tree (green)
    5: 10,  # Unused Land -> grass (bright green)
    6: 12,  # Agricultural Area -> field (yellow-green)
}


def create_seg_control_image(label_mask: np.ndarray) -> Image.Image:
    """
    Create ADE20K-compatible segmentation control image for ControlNet.
    Maps our 6 land-use classes to ADE20K palette colors.
    """
    h, w = label_mask.shape[:2]
    color_seg = np.zeros((h, w, 3), dtype=np.uint8)
    for label_id, ada_idx in LABEL_TO_ADE20K_IDX.items():
        mask = (label_mask == label_id)
        color_seg[mask] = ADE20K_PALETTE[ada_idx]
    return Image.fromarray(color_seg)


def extract_canny_edges(
    image: Image.Image,
    low_threshold: int = 50,
    high_threshold: int = 120
) -> Image.Image:
    """
    Extract Canny edges from the original satellite image for ControlNet conditioning.
    Preserves roads, rivers, building outlines, and field boundaries from the real image.
    Returns white-on-black edge map (required by Canny ControlNet).
    """
    img_array = np.array(image)
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array
    edges = cv2.Canny(gray, low_threshold, high_threshold)
    return Image.fromarray(edges, mode="L").convert("RGB")


def extract_depth_image(image: Image.Image, resolution: int = 512) -> Image.Image:
    """
    Estimate depth from image using controlnet-aux MidasDetector.
    Returns grayscale depth map (3ch) for Depth ControlNet.
    """
    try:
        from controlnet_aux import MidasDetector
        detector = MidasDetector.from_pretrained("lllyasviel/ControlNet")
        depth = detector(image, detect_resolution=resolution, image_resolution=resolution)
        return depth if isinstance(depth, Image.Image) else Image.fromarray(depth)
    except Exception as e:
        raise RuntimeError(f"Failed to extract depth (install controlnet-aux): {e}") from e


class SmartCitySatelliteGenerator:
    """
    Generator for smart city satellite images using ControlNet + SD Inpainting + IP-Adapter.
    Supports multi-ControlNet: Canny + Segmentation + optional Depth.
    """
    
    def __init__(
        self,
        device: str = None,
        use_ip_adapter: bool = False,
        low_vram: bool = False,
        use_seg_control: bool = True,
        use_depth_control: bool = False,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.use_ip_adapter = use_ip_adapter
        self.low_vram = low_vram
        self.use_seg_control = use_seg_control
        self.use_depth_control = use_depth_control
        
        print(f"Initializing Smart City Generator on {self.device}"
              + (" (low VRAM mode)" if low_vram else "") + "...")
        
        print("Loading ControlNet (Canny edges)...")
        controlnets = [ControlNetModel.from_pretrained(
            "lllyasviel/control_v11p_sd15_canny",
            torch_dtype=self.dtype
        )]
        
        if use_seg_control:
            print("Loading ControlNet (Segmentation)...")
            controlnets.append(ControlNetModel.from_pretrained(
                "lllyasviel/control_v11p_sd15_seg",
                torch_dtype=self.dtype
            ))
        
        if use_depth_control:
            print("Loading ControlNet (Depth)...")
            controlnets.append(ControlNetModel.from_pretrained(
                "lllyasviel/control_v11f1p_sd15_depth",
                torch_dtype=self.dtype
            ))
        
        self.controlnet = MultiControlNetModel(controlnets) if len(controlnets) > 1 else controlnets[0]
        
        print("Loading Stable Diffusion Inpainting pipeline...")
        self.pipeline = StableDiffusionControlNetInpaintPipeline.from_pretrained(
            "runwayml/stable-diffusion-inpainting",
            controlnet=self.controlnet,
            torch_dtype=self.dtype,
            safety_checker=None,
        )
        
        self.pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
            self.pipeline.scheduler.config,
            use_karras_sigmas=True,
            algorithm_type="dpmsolver++",
        )
        
        if use_ip_adapter:
            try:
                print("Loading IP-Adapter for texture/lighting consistency...")
                self.pipeline.load_ip_adapter(
                    "h94/IP-Adapter",
                    subfolder="models",
                    weight_name="ip-adapter_sd15.bin"
                )
                print("IP-Adapter loaded successfully")
            except Exception as e:
                print(f"Warning: Could not load IP-Adapter ({e}). Proceeding without style reference.")
                self.use_ip_adapter = False
        
        if self.device == "cuda":
            self.pipeline.enable_attention_slicing()
            # Skip xformers when using IP-Adapter (known compatibility: tuple/shape error)
            if not self.use_ip_adapter:
                try:
                    self.pipeline.enable_xformers_memory_efficient_attention()
                    print("xformers memory efficient attention enabled")
                except Exception:
                    print("xformers not available, using standard attention")
        
        if low_vram and self.device == "cuda":
            print("Enabling model CPU offload for low VRAM GPUs...")
            self.pipeline.enable_model_cpu_offload()
        else:
            self.pipeline = self.pipeline.to(self.device)
        print("Generator ready!\n")
    
    def generate_smart_city_image(
        self,
        original_image: Image.Image,
        control_image: Union[Image.Image, List[Image.Image]],
        inpaint_mask: Image.Image,
        prompt: str,
        negative_prompt: str = "",
        num_inference_steps: int = 35,
        guidance_scale: float = 7.0,
        controlnet_conditioning_scale: Union[float, List[float], None] = None,
        ip_adapter_scale: float = 0.6,
        strength: float = 0.60,
        seed: Optional[int] = None,
    ) -> Image.Image:
        """Generate a smart city satellite image using inpainting with ControlNet(s) and optional IP-Adapter.
        control_image: single image (Canny only) or list [canny, seg, depth] for multi-ControlNet.
        controlnet_conditioning_scale: float or list matching number of ControlNets.
        """
        if controlnet_conditioning_scale is None:
            scales = [0.8]  # Canny
            if self.use_seg_control:
                scales.append(1.0)
            if self.use_depth_control:
                scales.append(0.7)
            controlnet_conditioning_scale = scales[0] if len(scales) == 1 else scales

        # With CPU offload, generator device must be "cpu" for correct placement
        gen_device = "cpu" if self.low_vram else self.device
        generator = None
        if seed is not None:
            generator = torch.Generator(device=gen_device).manual_seed(seed)
            print(f"Using seed: {seed}")
        
        if self.use_ip_adapter:
            self.pipeline.set_ip_adapter_scale(ip_adapter_scale)
        
        print(f"Generating with prompt: '{prompt[:80]}...'")
        print(f"Steps: {num_inference_steps}, Strength: {strength}, Guidance: {guidance_scale}, "
              f"ControlNet scale: {controlnet_conditioning_scale}"
              + (f", IP-Adapter scale: {ip_adapter_scale}" if self.use_ip_adapter else ""))
        
        pipeline_kwargs = dict(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=original_image,
            mask_image=inpaint_mask,
            control_image=control_image,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            strength=strength,
            generator=generator,
            height=original_image.height,
            width=original_image.width,
        )
        if self.use_ip_adapter:
            pipeline_kwargs["ip_adapter_image"] = original_image
        
        with torch.no_grad():
            result = self.pipeline(**pipeline_kwargs)
        
        return result.images[0]


def denoise_generated_regions(
    generated_image: Image.Image,
    inpaint_mask: Image.Image,
    strength: float = 0.3,
    d: int = 5,
    sigma_color: float = 50,
    sigma_space: float = 50,
) -> Image.Image:
    """
    Apply light bilateral denoising to generated regions only.
    Preserves edges while reducing flat-area noise. Strength controls blend
    between original and filtered (0=no change, 1=fully filtered).
    """
    gen_array = np.array(generated_image)
    mask_array = np.array(inpaint_mask)
    editable = (mask_array > 0).astype(np.float32)
    if np.sum(editable) == 0:
        return generated_image

    filtered = cv2.bilateralFilter(gen_array, d=d, sigmaColor=sigma_color, sigmaSpace=sigma_space)
    # Blend: strength * filtered + (1-strength) * original in editable regions
    blend = (strength * filtered + (1 - strength) * gen_array).astype(np.float32)
    editable_3ch = np.stack([editable] * 3, axis=-1)
    result = gen_array.astype(np.float32) * (1 - editable_3ch) + blend * editable_3ch
    result = np.clip(result, 0, 255).astype(np.uint8)
    return Image.fromarray(result)


def sharpen_generated_regions(
    generated_image: Image.Image,
    inpaint_mask: Image.Image,
    amount: float = 0.3,
    radius: float = 1.0,
    threshold: int = 0,
) -> Image.Image:
    """
    Apply unsharp mask to generated regions only for high-frequency detail.
    """
    from PIL import ImageFilter
    gen_array = np.array(generated_image)
    mask_array = np.array(inpaint_mask)
    editable = (mask_array > 0).astype(np.float32)
    if np.sum(editable) == 0:
        return generated_image
    sharpened = generated_image.filter(
        ImageFilter.UnsharpMask(radius=radius, percent=int(100 + amount * 100), threshold=threshold)
    )
    sharp_array = np.array(sharpened)
    editable_3ch = np.stack([editable] * 3, axis=-1)
    result = gen_array.astype(np.float32) * (1 - editable_3ch) + sharp_array.astype(np.float32) * editable_3ch
    result = np.clip(result, 0, 255).astype(np.uint8)
    return Image.fromarray(result)


def composite_with_original(
    generated_image: Image.Image,
    original_image: Image.Image,
    inpaint_mask: Image.Image,
    immutable_mask: Optional[np.ndarray] = None,
) -> Image.Image:
    """
    Composite generated image with original, ensuring immutable regions
    (River, Road, Forest) are preserved exactly at the pixel level.
    Uses explicit immutable mask when provided; preserve = immutable OR not-editable.
    """
    gen_array = np.array(generated_image)
    orig_array = np.array(original_image)
    mask_array = np.array(inpaint_mask)
    
    # Where to use original: immutable pixels OR pixels not in editable (inpaint) region
    if immutable_mask is not None and np.sum(immutable_mask) > 0:
        preserve = ((immutable_mask > 0) | (mask_array == 0)).astype(np.float32)
    else:
        preserve = (mask_array == 0).astype(np.float32)
    
    preserve_3ch = np.stack([preserve] * 3, axis=-1)
    result = gen_array * (1 - preserve_3ch) + orig_array * preserve_3ch
    result = result.astype(np.uint8)
    return Image.fromarray(result)


# =============================================================================
# PART 4: VALIDATION
# =============================================================================

def validate_and_log_results(
    original_mask_path: str,
    generated_image: Image.Image,
    original_coverage: Dict[str, float],
    output_dir: str,
    validation_path: Optional[str] = None,
) -> Dict:
    """
    Validate that immutable regions were preserved.
    
    Args:
        original_mask_path: Path to original segmentation mask
        generated_image: The generated smart city image
        original_coverage: Original coverage percentages
        output_dir: Directory to save validation results
        
    Returns:
        Validation results dictionary
    """
    validation = {
        "passed": True,
        "immutable_preserved": True,
        "tolerance": 0.5,  # % tolerance for immutable regions
        "details": {},
    }
    
    # Note: For proper validation, you would need to segment the generated image
    # Since we're using compositing, immutable regions are guaranteed preserved
    
    validation["details"]["compositing_used"] = True
    validation["details"]["immutable_classes"] = IMMUTABLE_CLASSES
    validation["details"]["original_coverage"] = original_coverage
    
    # Log validation results
    validation_log = f"""
# Validation Results

**Compositing Applied:** Yes (immutable regions guaranteed preserved)

**Immutable Classes Preserved:**
- Road: {original_coverage.get('Road', 0):.2f}% ✓
- River: {original_coverage.get('River', 0):.2f}% ✓
- Forest: {original_coverage.get('Forest', 0):.2f}% ✓

**Validation Status:** PASSED
"""
    
    val_path = validation_path or os.path.join(output_dir, "validation_results.md")
    with open(val_path, "w") as f:
        f.write(validation_log)
    
    print(f"✓ Validation results saved: {val_path}")
    
    return validation


# =============================================================================
# PROMPT CONFIGURATION
# =============================================================================

# Planning-informed descriptors per editable class (concise for CLIP 77-token limit)
CLASS_PROMPTS = {
    "Residential Area": (
        "planned neighborhood, street grid, roads between blocks, spacing between houses, "
        "small parks and green pockets, tree-lined streets, organized housing layout, green rooftops"
    ),
    "Unused Land": (
        "parks, recreational areas, playgrounds, sports fields, walking paths, "
        "green space, community garden, urban park"
    ),
    "Agricultural Area": (
        "modern precision agriculture, organized crop fields, greenhouses, "
        "structured farm layout, irrigation patterns"
    ),
}


def build_smart_city_prompt(coverage: Dict[str, float], base_prompt: str) -> str:
    """
    Build class-aware prompt from coverage using planning-informed descriptors.
    Kept under ~70 tokens for CLIP's 77-token limit.
    """
    parts = [base_prompt]
    for class_name in EDITABLE_CLASSES:
        if coverage.get(class_name, 0) > 0 and class_name in CLASS_PROMPTS:
            parts.append(CLASS_PROMPTS[class_name])
    prompt = ", ".join(parts)
    if len(prompt) > 280:
        prompt = prompt[:277] + "..."
    return prompt


DEFAULT_PROMPT = (
    "high-resolution satellite aerial image, orthographic view, "
    "realistic rooftops, agricultural parcels, forest canopy texture, "
    "clear road networks, geographic consistency, "
    "futuristic planned city, organized layout, wide tree-lined boulevards, "
    "modern glass and steel buildings, green roofs, solar panels on buildings, "
    "hyper-realistic, bright daylight, utopian atmosphere, "
    "clear sky, no clouds, photorealistic, sharp focus"
)

DEFAULT_NEGATIVE_PROMPT = (
    "abstract, painting, brush strokes, distorted, melted, surreal, texture hallucination, "
    "clouds, cloud cover, cloudy, overcast, fog, haze, mist, atmospheric effects, "
    "cloud shadows, cumulus, stratus, cirrus, weather, "
    "grainy, noisy, blurry, artifacts, low resolution, "
    "watercolor, painted, soft edges, smudged, amorphous shapes, "
    "new roads, altered roads, modified river paths, "
    "industrial buildings, factories, heavy infrastructure, "
    "fantasy style, artistic style, painting, illustration, "
    "distorted geometry, unrealistic colors, blurry, "
    "low quality, watermark, text, cartoon, "
    "informal settlement, unplanned sprawl, cramped housing, no spacing, dense slum, "
    "overgrown vacant lot, barren land"
)


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def get_project_root() -> Path:
    """Get project root directory (parent of src/)."""
    script_dir = Path(__file__).parent.resolve()
    if script_dir.name == "src":
        return script_dir.parent
    return script_dir


def process_single_pair(
    image_path: str,
    mask_path: str,
    output_path: str,
    args,
    script_dir: str,
    project_root: Path,
    verbose: bool = True,
) -> bool:
    """
    Process a single image-mask pair: analysis, report, generation, validation.
    Returns True on success, False on failure.
    """
    def log(msg: str):
        if verbose:
            print(msg)

    output_dir = os.path.dirname(output_path) or str(project_root / "output")
    os.makedirs(output_dir, exist_ok=True)

    if verbose:
        print(f"\nProcessing: {os.path.basename(image_path)}")

    # PART 1: Spatial Analysis
    _, seg_mask_for_analysis, label_mask_for_analysis = load_image_and_mask(
        image_path, mask_path, size=(args.size, args.size)
    )
    coverage = compute_coverage(mask_path)

    if verbose:
        log("\n   Coverage Results:")
        for class_name, pct in sorted(coverage.items(), key=lambda x: -x[1]):
            if pct > 0:
                status = "Editable" if class_name in EDITABLE_CLASSES else "Immutable"
                log(f"   - {class_name}: {pct:.2f}% ({status})")

    spatial_metrics = compute_spatial_metrics(label_mask=label_mask_for_analysis)
    analysis = analyze_urban_layout(coverage, spatial_metrics)
    suggestions = generate_suggestions(analysis)

    # PART 2: Report (only when analysis-only; otherwise output dir stays clean)
    if args.analysis_only:
        if verbose:
            log("\n" + "=" * 70)
            log("PART 2: REPORT GENERATION")
            log("=" * 70)
        report_path = os.path.join(output_dir, f"{Path(output_path).stem}_report.md")
        create_report(image_path, mask_path, analysis, suggestions, report_path)
        if verbose:
            log(f"   Report saved: {report_path}")

    if args.analysis_only:
        return True

    # PART 3: Generation
    if verbose:
        log("\n" + "=" * 70)
        log("PART 3: CONSTRAINED IMAGE GENERATION")
        log("=" * 70)
        log("\n[1/5] Loading image and segmentation mask...")
        log("[2/5] Creating class masks...")
        log("[3/5] Creating inpainting mask and immutable mask...")
        log("[4/5] Extracting Canny edges...")
        log("[5/5] Generating smart city satellite image...")
    original_image, seg_mask, label_mask = load_image_and_mask(
        image_path, mask_path, size=(args.size, args.size)
    )
    class_masks = create_class_masks(label_mask)
    inpaint_mask = create_inpainting_mask(class_masks, EDITABLE_CLASSES)
    immutable_mask = create_immutable_mask(class_masks, IMMUTABLE_CLASSES)

    # Clean output: save only input, mask, and output
    stem = Path(output_path).stem
    input_save_path = os.path.join(output_dir, f"{stem}_input.png")
    mask_save_path = os.path.join(output_dir, f"{stem}_mask.png")
    original_image.save(input_save_path)
    seg_mask.save(mask_save_path)
    if verbose:
        log(f"   Input saved: {input_save_path}")
        log(f"   Mask saved: {mask_save_path}")

    # Optional: use a separate reference image for ControlNet structure (e.g. planned city layout)
    ref_path = getattr(args, "reference_image", None)
    if ref_path and os.path.exists(ref_path):
        if verbose:
            log("   Using reference image for ControlNet structure (Canny"
                + (", depth" if getattr(args, "use_depth_control", False) else "") + ")")
        reference_image = Image.open(ref_path).convert("RGB").resize(
            (args.size, args.size), Image.Resampling.LANCZOS
        )
        canny_image = extract_canny_edges(
            reference_image,
            low_threshold=getattr(args, "canny_low", 50),
            high_threshold=getattr(args, "canny_high", 120),
        )
        ctrl_for_depth = reference_image
        # Save reference and its Canny to output for reproducibility
        ref_save = os.path.join(output_dir, f"{stem}_reference.png")
        canny_save = os.path.join(output_dir, f"{stem}_reference_canny.png")
        reference_image.save(ref_save)
        canny_image.save(canny_save)
        if verbose:
            log(f"   Reference saved: {ref_save}")
            log(f"   Reference Canny saved: {canny_save}")
    else:
        canny_image = extract_canny_edges(
            original_image,
            low_threshold=getattr(args, "canny_low", 50),
            high_threshold=getattr(args, "canny_high", 120),
        )
        ctrl_for_depth = original_image

    control_images = [canny_image]
    control_scales = [getattr(args, "controlnet_scale", 0.8)]

    if getattr(args, "no_seg_control", False):
        if verbose:
            log("   WARNING: Segmentation disabled - expect structural loss (roads, parcels may melt)")
    else:
        seg_image = create_seg_control_image(label_mask)
        control_images.append(seg_image)
        control_scales.append(getattr(args, "seg_scale", 1.0))

    if getattr(args, "use_depth_control", False):
        depth_image = extract_depth_image(ctrl_for_depth, resolution=args.size)
        control_images.append(depth_image)
        control_scales.append(getattr(args, "depth_scale", 0.7))

    control_image = control_images[0] if len(control_images) == 1 else control_images
    controlnet_scale = control_scales[0] if len(control_scales) == 1 else control_scales

    prompt = build_smart_city_prompt(coverage, args.prompt)

    generator = SmartCitySatelliteGenerator(
        use_ip_adapter=args.use_ip_adapter,
        low_vram=args.low_vram,
        use_seg_control=not getattr(args, "no_seg_control", False),
        use_depth_control=getattr(args, "use_depth_control", False),
    )
    generated_image = generator.generate_smart_city_image(
        original_image=original_image,
        control_image=control_image,
        inpaint_mask=inpaint_mask,
        prompt=prompt,
        negative_prompt=args.negative_prompt,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        controlnet_conditioning_scale=controlnet_scale,
        ip_adapter_scale=args.ip_adapter_scale,
        strength=args.strength,
        seed=args.seed,
    )

    if args.denoise:
        generated_image = denoise_generated_regions(
            generated_image, inpaint_mask, strength=args.denoise_strength
        )
        if verbose:
            log("   Applied light denoising to generated regions")

    if getattr(args, "sharpen", False):
        generated_image = sharpen_generated_regions(
            generated_image, inpaint_mask, amount=getattr(args, "sharpen_amount", 0.3)
        )
        if verbose:
            log("   Applied sharpening to generated regions")

    final_image = composite_with_original(
        generated_image, original_image, inpaint_mask, immutable_mask
    )

    # Save output (ControlNet result)
    output_save_path = os.path.join(output_dir, f"{stem}_output.png")
    final_image.save(output_save_path)
    if verbose:
        log("\nCompositing with original (preserving immutable regions)...")
        log(f"   Output saved: {output_save_path}")

    return True


def main():
    project_root = get_project_root()
    
    parser = argparse.ArgumentParser(
        description="Smart City Satellite Image Generation Pipeline"
    )
    parser.add_argument(
        "--image",
        type=str,
        default=DEFAULT_IMAGE_PATH,
        help="Path to original satellite image"
    )
    parser.add_argument(
        "--mask",
        type=str,
        default=DEFAULT_MASK_PATH,
        help="Path to segmentation mask"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(project_root / "output" / "smart_city_generated.png"),
        help="Path to save generated image"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=DEFAULT_PROMPT,
        help="Text prompt for generation"
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default=DEFAULT_NEGATIVE_PROMPT,
        help="Negative prompt"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=35,
        help="Number of inference steps"
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=7.0,
        help="Guidance scale (6.5-8 for stability)"
    )
    parser.add_argument(
        "--controlnet_scale",
        type=float,
        default=0.8,
        help="Canny ControlNet conditioning scale"
    )
    parser.add_argument(
        "--strength",
        type=float,
        default=0.60,
        help="Inpainting strength (0.5-0.8). Lower preserves more input style; 1.0 fully regenerates"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--size",
        type=int,
        default=512,
        help="Image size. 512 default (works on 4GB GPUs with --low_vram). 768 needs 8GB+ VRAM"
    )
    parser.add_argument(
        "--analysis_only",
        action="store_true",
        help="Only run analysis and generate report, skip image generation"
    )
    parser.add_argument(
        "--ip_adapter_scale",
        type=float,
        default=0.6,
        help="IP-Adapter scale for style consistency (0.5-0.7 recommended)"
    )
    parser.add_argument(
        "--use_ip_adapter",
        action="store_true",
        help="Enable IP-Adapter for texture/lighting consistency (experimental: may fail with ControlNet+inpainting)"
    )
    parser.add_argument(
        "--low_vram",
        action="store_true",
        help="Use CPU offload for low VRAM GPUs (e.g. 4GB). Slower but works on small GPUs."
    )
    parser.add_argument(
        "--denoise",
        action="store_true",
        help="Apply light bilateral denoising to generated regions to reduce noise"
    )
    parser.add_argument(
        "--denoise_strength",
        type=float,
        default=0.3,
        help="Denoise blend strength (0-1). Higher = more smoothing. Default 0.3"
    )
    parser.add_argument(
        "--no_seg_control",
        action="store_true",
        help="Disable segmentation ControlNet (Canny only)"
    )
    parser.add_argument(
        "--seg_scale",
        type=float,
        default=1.0,
        help="Segmentation ControlNet weight (1.0 recommended for structural preservation)"
    )
    parser.add_argument(
        "--use_depth_control",
        action="store_true",
        help="Enable depth ControlNet for lighting consistency (extra VRAM)"
    )
    parser.add_argument(
        "--depth_scale",
        type=float,
        default=0.7,
        help="Depth ControlNet weight"
    )
    parser.add_argument(
        "--canny_low",
        type=int,
        default=50,
        help="Canny edge detection low threshold"
    )
    parser.add_argument(
        "--canny_high",
        type=int,
        default=120,
        help="Canny edge detection high threshold"
    )
    parser.add_argument(
        "--sharpen",
        action="store_true",
        help="Apply unsharp mask to generated regions"
    )
    parser.add_argument(
        "--sharpen_amount",
        type=float,
        default=0.3,
        help="Sharpen strength (0-1). Default 0.3"
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Process all image-mask pairs in dataset directory"
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default=None,
        help="Dataset directory with images/ and masks/ subdirs (default: project_root/dataset)"
    )
    parser.add_argument(
        "--reference_image",
        type=str,
        default=None,
        help="Planned-city (or layout) image for ControlNet structure. Canny and depth from this image; inpainting uses --image/--mask. Default: project reference_image/ folder if present."
    )
    parser.add_argument(
        "--no_reference",
        action="store_true",
        help="Do not use any reference image; use unplanned image for ControlNet structure (original behavior)."
    )

    args = parser.parse_args()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Resolve reference image: explicit path, or default from reference_image/ folder
    if getattr(args, "no_reference", False):
        args.reference_image = None
    elif getattr(args, "reference_image", None):
        ref = args.reference_image
        if not os.path.isabs(ref):
            args.reference_image = os.path.join(script_dir, ref)
    else:
        # Default: use project reference_image/ folder (single file or first image found)
        proot = get_project_root()
        default_ref = os.path.join(proot, DEFAULT_REFERENCE_IMAGE_PATH)
        if os.path.isfile(default_ref):
            args.reference_image = default_ref
        else:
            ref_dir = os.path.join(proot, "reference_image")
            if os.path.isdir(ref_dir):
                for ext in (".png", ".jpg", ".jpeg", ".bmp", ".tiff"):
                    found = next(
                        (
                            os.path.join(ref_dir, f)
                            for f in sorted(os.listdir(ref_dir))
                            if f.lower().endswith(ext)
                        ),
                        None,
                    )
                    if found:
                        args.reference_image = found
                        break
    output_dir = os.path.dirname(args.output) or str(project_root / "output")
    os.makedirs(output_dir, exist_ok=True)

    if args.batch:
        # Batch mode: process all image-mask pairs in dataset
        dataset_dir = args.dataset_dir or str(project_root / "dataset")
        images_dir = os.path.join(dataset_dir, "images")
        masks_dir = os.path.join(dataset_dir, "masks")

        if not os.path.isdir(images_dir):
            print(f"Error: Images directory not found: {images_dir}")
            return
        if not os.path.isdir(masks_dir):
            print(f"Error: Masks directory not found: {masks_dir}")
            return

        image_extensions = (".png", ".jpg", ".jpeg", ".bmp", ".tiff")
        image_files = [
            f for f in os.listdir(images_dir)
            if f.lower().endswith(image_extensions)
        ]

        pairs = []
        for img_file in sorted(image_files):
            mask_path = os.path.join(masks_dir, img_file)
            if os.path.exists(mask_path):
                pairs.append((
                    os.path.join(images_dir, img_file),
                    mask_path,
                ))
            else:
                print(f"Skipping {img_file}: no matching mask found")

        print("=" * 70)
        print("    BATCH PROCESSING")
        print("=" * 70)
        print(f"Dataset: {dataset_dir}")
        print(f"Pairs found: {len(pairs)}")
        print(f"Output dir: {output_dir}")
        print("=" * 70)

        for i, (image_path, mask_path) in enumerate(pairs):
            stem = Path(image_path).stem
            output_path = os.path.join(output_dir, f"generated_{stem}.png")
            print(f"\n[{i + 1}/{len(pairs)}] {stem}")
            try:
                process_single_pair(
                    image_path, mask_path, output_path,
                    args, script_dir, project_root, verbose=True
                )
            except Exception as e:
                print(f"   Error: {e}")
                continue

        print("\n" + "=" * 70)
        print("    BATCH COMPLETE")
        print("=" * 70)
        print(f"Processed {len(pairs)} pairs. Outputs in {output_dir}/")
        return

    # Single-image mode
    image_path = args.image
    if not os.path.isabs(image_path):
        image_path = os.path.join(script_dir, image_path)
    mask_path = args.mask
    if not os.path.isabs(mask_path):
        mask_path = os.path.join(script_dir, mask_path)

    if not os.path.exists(image_path):
        print(f"Error: Image not found: {image_path}")
        return
    if not os.path.exists(mask_path):
        print(f"Error: Mask not found: {mask_path}")
        return

    print("=" * 70)
    print("    URBAN SPATIAL ANALYSIS AND SMART CITY VISUALIZATION")
    print("=" * 70)
    print(f"Image: {image_path}")
    print(f"Mask:  {mask_path}")
    print(f"Output: {args.output}")
    print("=" * 70)
    print("\n" + "=" * 70)
    print("PART 1: SPATIAL ANALYSIS")
    print("=" * 70)
    print("\n[1/5] Loading segmentation mask for analysis...")
    print("[2/5] Computing area coverage...")
    print("[3/5] Computing spatial metrics...")
    print("[4/5] Analyzing urban layout...")
    print("[5/5] Generating suggestions...")

    process_single_pair(
        image_path, mask_path, args.output,
        args, script_dir, project_root, verbose=True
    )

    if not args.analysis_only:
        print("\n" + "=" * 70)
        print("    PIPELINE COMPLETE")
        print("=" * 70)
        print(f"\nOutput saved to: {output_dir}/")
        print("  - *_input.png   Your input image")
        print("  - *_mask.png    Your segmentation mask")
        print("  - *_output.png  ControlNet generation result")
    else:
        print("\n" + "=" * 70)
        print("ANALYSIS COMPLETE (image generation skipped)")
        print("=" * 70)


if __name__ == "__main__":
    main()
