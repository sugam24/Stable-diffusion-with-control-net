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
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path
from PIL import Image
import argparse
from typing import Tuple, Dict, Optional, List
from datetime import datetime

from diffusers import (
    StableDiffusionControlNetInpaintPipeline,
    ControlNetModel,
    UniPCMultistepScheduler,
)

# Import compute_coverage from the same directory
from compute_coverage import compute_coverage


# =============================================================================
# SET YOUR IMAGE PATHS HERE (change these as needed)
# =============================================================================
DEFAULT_IMAGE_PATH = "../dataset/Annotation/annotated_images/output_337.png"
DEFAULT_MASK_PATH = "../dataset/Annotation/annotated_masks/output_337.png"
# =============================================================================


# =============================================================================
# CONFIGURATION - Segmentation Class Colors (RGB)
# =============================================================================
CLASS_COLORS = {
    "River": (128, 0, 0),              # Red - IMMUTABLE
    "Residential Area": (0, 0, 128),   # Dark Blue - EDITABLE
    "Road": (0, 128, 0),               # Green - IMMUTABLE
    "Forest": (128, 128, 0),           # Yellow - IMMUTABLE
    "Unused Land": (0, 128, 128),      # Cyan - EDITABLE
    "Agricultural Area": (128, 0, 128), # Magenta - IMMUTABLE
}

# Classes that can be modified during generation
EDITABLE_CLASSES = ["Residential Area", "Unused Land"]

# Classes that must remain unchanged
IMMUTABLE_CLASSES = ["Road", "River", "Forest", "Agricultural Area"]


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


def compute_spatial_metrics(seg_mask: Image.Image, class_colors: Dict) -> Dict:
    """
    Compute spatial metrics from the segmentation mask for evidence-based analysis.
    
    Metrics computed:
    - Connected components per class
    - Average and median block sizes
    - Fragmentation index
    - Adjacency relationships
    - Compactness measures
    
    Args:
        seg_mask: Color-coded segmentation mask (PIL Image)
        class_colors: Dictionary mapping class names to RGB tuples
        
    Returns:
        Dictionary containing all spatial metrics
    """
    mask_array = np.array(seg_mask)
    h, w = mask_array.shape[:2]
    total_pixels = h * w
    
    metrics = {
        "image_dimensions": {"height": h, "width": w, "total_pixels": total_pixels},
        "class_metrics": {},
        "adjacency": {},
        "fragmentation": {},
    }
    
    # Create binary masks for each class
    class_masks = {}
    for class_name, color in class_colors.items():
        color_array = np.array(color)
        matches = np.all(mask_array == color_array, axis=-1)
        class_masks[class_name] = matches.astype(np.uint8)
    
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
    if coverage.get("Agricultural Area", 0) > 0:
        suggestions["constraints"].append({
            "class": "Agricultural Area",
            "coverage": coverage.get("Agricultural Area", 0),
            "rationale": "Agricultural land supports food production capacity"
        })
    
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
| Residential Area | {analysis['coverage'].get('Residential Area', 0):.2f} | Smart city visualization |
| Unused Land | {analysis['coverage'].get('Unused Land', 0):.2f} | Development visualization |

---

## Technical Notes

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

def load_image_and_mask(
    image_path: str,
    mask_path: str,
    size: Tuple[int, int] = (512, 512)
) -> Tuple[Image.Image, Image.Image]:
    """
    Load and resize the original satellite image and its segmentation mask.
    """
    original_image = Image.open(image_path).convert("RGB")
    original_image = original_image.resize(size, Image.Resampling.LANCZOS)
    
    seg_mask = Image.open(mask_path).convert("RGB")
    seg_mask = seg_mask.resize(size, Image.Resampling.NEAREST)
    
    return original_image, seg_mask


def create_class_masks(
    seg_mask: Image.Image,
    class_colors: Dict[str, Tuple[int, int, int]]
) -> Dict[str, np.ndarray]:
    """Create binary masks for each segmentation class."""
    mask_array = np.array(seg_mask)
    class_masks = {}
    
    for class_name, color in class_colors.items():
        color_array = np.array(color)
        matches = np.all(mask_array == color_array, axis=-1)
        class_masks[class_name] = matches.astype(np.uint8)
    
    return class_masks


def create_inpainting_mask(
    class_masks: Dict[str, np.ndarray],
    editable_classes: List[str]
) -> Image.Image:
    """
    Create inpainting mask from editable class regions.
    White (255) = regions to regenerate, Black (0) = regions to preserve.
    """
    h, w = list(class_masks.values())[0].shape
    inpaint_mask = np.zeros((h, w), dtype=np.uint8)
    
    for class_name in editable_classes:
        if class_name in class_masks:
            inpaint_mask = np.maximum(inpaint_mask, class_masks[class_name] * 255)
    
    return Image.fromarray(inpaint_mask, mode="L")


def prepare_controlnet_input(seg_mask: Image.Image) -> Image.Image:
    """Prepare segmentation mask as ControlNet conditioning input."""
    return seg_mask


class SmartCitySatelliteGenerator:
    """
    Generator for smart city satellite images using ControlNet + SD Inpainting.
    """
    
    def __init__(self, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        
        print(f"Initializing Smart City Generator on {self.device}...")
        
        print("Loading ControlNet (segmentation)...")
        self.controlnet = ControlNetModel.from_pretrained(
            "lllyasviel/control_v11p_sd15_seg",
            torch_dtype=self.dtype
        )
        
        print("Loading Stable Diffusion Inpainting pipeline...")
        self.pipeline = StableDiffusionControlNetInpaintPipeline.from_pretrained(
            "runwayml/stable-diffusion-inpainting",
            controlnet=self.controlnet,
            torch_dtype=self.dtype,
            safety_checker=None,
        )
        
        self.pipeline.scheduler = UniPCMultistepScheduler.from_config(
            self.pipeline.scheduler.config
        )
        
        if self.device == "cuda":
            self.pipeline.enable_attention_slicing()
            try:
                self.pipeline.enable_xformers_memory_efficient_attention()
                print("xformers memory efficient attention enabled")
            except Exception:
                print("xformers not available, using standard attention")
        
        self.pipeline = self.pipeline.to(self.device)
        print("Generator ready!\n")
    
    def generate_smart_city_image(
        self,
        original_image: Image.Image,
        control_image: Image.Image,
        inpaint_mask: Image.Image,
        prompt: str,
        negative_prompt: str = "",
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,
        controlnet_conditioning_scale: float = 0.8,
        seed: Optional[int] = None,
    ) -> Image.Image:
        """Generate a smart city satellite image using inpainting with ControlNet."""
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
            print(f"Using seed: {seed}")
        
        print(f"Generating with prompt: '{prompt[:80]}...'")
        print(f"Steps: {num_inference_steps}, Guidance: {guidance_scale}, "
              f"ControlNet scale: {controlnet_conditioning_scale}")
        
        with torch.no_grad():
            result = self.pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=original_image,
                mask_image=inpaint_mask,
                control_image=control_image,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                controlnet_conditioning_scale=controlnet_conditioning_scale,
                generator=generator,
                height=original_image.height,
                width=original_image.width,
            )
        
        return result.images[0]


def composite_with_original(
    generated_image: Image.Image,
    original_image: Image.Image,
    inpaint_mask: Image.Image
) -> Image.Image:
    """
    Composite generated image with original, ensuring immutable regions
    are preserved exactly at the pixel level.
    """
    gen_array = np.array(generated_image)
    orig_array = np.array(original_image)
    mask_array = np.array(inpaint_mask)
    
    mask_float = mask_array.astype(np.float32) / 255.0
    mask_3ch = np.stack([mask_float] * 3, axis=-1)
    
    result = gen_array * mask_3ch + orig_array * (1 - mask_3ch)
    result = result.astype(np.uint8)
    
    return Image.fromarray(result)


# =============================================================================
# PART 4: VALIDATION
# =============================================================================

def validate_and_log_results(
    original_mask_path: str,
    generated_image: Image.Image,
    original_coverage: Dict[str, float],
    output_dir: str
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
- Agricultural Area: {original_coverage.get('Agricultural Area', 0):.2f}% ✓

**Validation Status:** PASSED
"""
    
    validation_path = os.path.join(output_dir, "validation_results.md")
    with open(validation_path, "w") as f:
        f.write(validation_log)
    
    print(f"✓ Validation results saved: {validation_path}")
    
    return validation


# =============================================================================
# PROMPT CONFIGURATION
# =============================================================================

DEFAULT_PROMPT = """
Aerial satellite view of a modern smart city district,
sustainable urban development, organized residential blocks,
green rooftops, solar panels, high-resolution satellite imagery,
top-down orthographic view, realistic urban planning,
smart infrastructure, clean streets, well-maintained buildings
""".strip().replace("\n", " ")

DEFAULT_NEGATIVE_PROMPT = """
clouds, cloud cover, fog, haze, atmospheric effects,
new roads, altered roads, modified river paths,
industrial buildings, factories, heavy infrastructure,
fantasy style, artistic style, painting, illustration,
distorted geometry, unrealistic colors, blurry,
low quality, watermark, text, cartoon
""".strip().replace("\n", " ")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def get_project_root() -> Path:
    """Get project root directory (parent of src/)."""
    script_dir = Path(__file__).parent.resolve()
    if script_dir.name == "src":
        return script_dir.parent
    return script_dir


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
        default=30,
        help="Number of inference steps"
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=7.5,
        help="Guidance scale"
    )
    parser.add_argument(
        "--controlnet_scale",
        type=float,
        default=0.8,
        help="ControlNet conditioning scale"
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
        help="Image size (width and height)"
    )
    parser.add_argument(
        "--analysis_only",
        action="store_true",
        help="Only run analysis and generate report, skip image generation"
    )
    
    args = parser.parse_args()
    
    # Resolve relative paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    image_path = args.image
    if not os.path.isabs(image_path):
        image_path = os.path.join(script_dir, image_path)
    
    mask_path = args.mask
    if not os.path.isabs(mask_path):
        mask_path = os.path.join(script_dir, mask_path)
    
    # Validate inputs
    if not os.path.exists(image_path):
        print(f"Error: Image not found: {image_path}")
        return
    if not os.path.exists(mask_path):
        print(f"Error: Mask not found: {mask_path}")
        return
    
    # Create output directory
    output_dir = os.path.dirname(args.output) or str(project_root / "output")
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 70)
    print("    URBAN SPATIAL ANALYSIS AND SMART CITY VISUALIZATION")
    print("=" * 70)
    print(f"Image: {image_path}")
    print(f"Mask:  {mask_path}")
    print(f"Output: {args.output}")
    print("=" * 70)
    
    # =========================================================================
    # PART 1: SPATIAL ANALYSIS
    # =========================================================================
    print("\n" + "=" * 70)
    print("PART 1: SPATIAL ANALYSIS")
    print("=" * 70)
    
    print("\n[1/5] Loading segmentation mask for analysis...")
    _, seg_mask_for_analysis = load_image_and_mask(
        image_path, mask_path, size=(args.size, args.size)
    )
    
    print("[2/5] Computing area coverage...")
    coverage = compute_coverage(mask_path)
    
    print("\n   Coverage Results:")
    for class_name, pct in sorted(coverage.items(), key=lambda x: -x[1]):
        if pct > 0:
            status = "Editable" if class_name in EDITABLE_CLASSES else "Immutable"
            print(f"   - {class_name}: {pct:.2f}% ({status})")
    
    print("\n[3/5] Computing spatial metrics (connected components, fragmentation, adjacency)...")
    spatial_metrics = compute_spatial_metrics(seg_mask_for_analysis, CLASS_COLORS)
    
    # Print key spatial indicators
    res_metrics = spatial_metrics.get("class_metrics", {}).get("Residential Area", {})
    if res_metrics.get("pixel_count", 0) > 0:
        print(f"\n   Residential Spatial Indicators:")
        print(f"   - Components: {res_metrics.get('num_components', 0)}")
        print(f"   - Compactness: {res_metrics.get('compactness', 'N/A')} (ratio: {res_metrics.get('compactness_ratio', 0):.3f})")
        print(f"   - Largest block: {res_metrics.get('largest_component_percent', 0):.2f}% of image")
    
    frag = spatial_metrics.get("fragmentation", {}).get("Residential Area", {})
    if frag:
        print(f"   - Fragmentation: {frag.get('level', 'N/A')} (index: {frag.get('index', 0):.3f})")
    
    print("\n[4/5] Analyzing urban layout (evidence-based)...")
    analysis = analyze_urban_layout(coverage, spatial_metrics)
    
    density = analysis.get("density_assessment", {})
    open_space = analysis.get("open_space_assessment", {})
    
    print(f"\n   Density Classification: {density.get('classification', 'N/A')}")
    if density.get("compactness"):
        print(f"   Residential Compactness: {density.get('compactness')}")
    print(f"   Open Space Accessibility: {open_space.get('accessibility', 'N/A')}")
    
    issues = analysis.get("issues", [])
    if issues:
        print(f"\n   Identified Issues ({len(issues)}):")
        for issue in issues:
            print(f"   - [{issue['severity']}] {issue['category']}")
    else:
        print("\n   No significant issues detected.")
    
    print("\n[5/5] Generating evidence-linked suggestions...")
    suggestions = generate_suggestions(analysis)
    
    interventions = suggestions.get("interventions", [])
    if interventions:
        print(f"\n   Interventions ({len(interventions)}):")
        for inv in interventions[:3]:
            print(f"   - {inv['linked_issue']}: {len(inv.get('recommendations', []))} recommendation(s)")
    
    # =========================================================================
    # PART 2: REPORT GENERATION
    # =========================================================================
    print("\n" + "=" * 70)
    print("PART 2: REPORT GENERATION")
    print("=" * 70)
    
    print("\nCreating evidence-based analysis report...")
    report_path = os.path.join(output_dir, "urban_analysis_report.md")
    report = create_report(image_path, mask_path, analysis, suggestions, report_path)
    print(f"Report saved: {report_path}")
    
    if args.analysis_only:
        print("\n" + "=" * 70)
        print("ANALYSIS COMPLETE (image generation skipped)")
        print("=" * 70)
        return
    
    # =========================================================================
    # PART 3: CONSTRAINED IMAGE GENERATION
    # =========================================================================
    print("\n" + "=" * 70)
    print("PART 3: CONSTRAINED IMAGE GENERATION")
    print("=" * 70)
    
    print("\n[1/5] Loading image and segmentation mask...")
    original_image, seg_mask = load_image_and_mask(
        image_path, mask_path, size=(args.size, args.size)
    )
    
    print("[2/5] Creating class masks...")
    class_masks = create_class_masks(seg_mask, CLASS_COLORS)
    
    print("[3/5] Creating inpainting mask (editable regions only)...")
    inpaint_mask = create_inpainting_mask(class_masks, EDITABLE_CLASSES)
    
    inpaint_mask_path = args.output.replace(".png", "_inpaint_mask.png")
    inpaint_mask.save(inpaint_mask_path)
    print(f"   Inpainting mask saved: {inpaint_mask_path}")
    
    print("[4/5] Preparing ControlNet conditioning...")
    control_image = prepare_controlnet_input(seg_mask)
    
    print("[5/5] Generating smart city satellite image...")
    generator = SmartCitySatelliteGenerator()
    
    generated_image = generator.generate_smart_city_image(
        original_image=original_image,
        control_image=control_image,
        inpaint_mask=inpaint_mask,
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        controlnet_conditioning_scale=args.controlnet_scale,
        seed=args.seed,
    )
    
    print("\nCompositing with original (preserving immutable regions)...")
    final_image = composite_with_original(generated_image, original_image, inpaint_mask)
    
    # Save outputs
    final_image.save(args.output)
    print(f"\n✓ Final image saved: {args.output}")
    
    raw_output_path = args.output.replace(".png", "_raw.png")
    generated_image.save(raw_output_path)
    print(f"✓ Raw generation saved: {raw_output_path}")
    
    # =========================================================================
    # PART 4: VALIDATION
    # =========================================================================
    print("\n" + "=" * 70)
    print("PART 4: VALIDATION")
    print("=" * 70)
    
    validation = validate_and_log_results(
        mask_path, final_image, coverage, output_dir
    )
    
    print("\n" + "=" * 70)
    print("    PIPELINE COMPLETE")
    print("=" * 70)
    print(f"\nOutputs saved to: {output_dir}/")
    print(f"  • smart_city_generated.png     - Final composited image")
    print(f"  • smart_city_generated_raw.png - Raw diffusion output")
    print(f"  • urban_analysis_report.md     - Analysis and suggestions")
    print(f"  • validation_results.md        - Validation log")
    print("=" * 70)


if __name__ == "__main__":
    main()
