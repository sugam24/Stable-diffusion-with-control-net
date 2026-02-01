# Complete Project Documentation

## Smart City Satellite Image Generation Pipeline

This document records **everything** implemented in this project for documentation and future reference. No detail is omitted.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Original Problems and Context](#2-original-problems-and-context)
3. [Architecture and Pipeline Flow](#3-architecture-and-pipeline-flow)
4. [Configuration and Constants](#4-configuration-and-constants)
5. [Prompts](#5-prompts)
6. [Features Implemented](#6-features-implemented)
7. [Command-Line Arguments](#7-command-line-arguments)
8. [Run Examples](#8-run-examples)
9. [Output Files](#9-output-files)
10. [File Structure](#10-file-structure)
11. [Dependencies](#11-dependencies)

---

## 1. Project Overview

### What This Project Does

- **Input:** A satellite image of an unplanned urban area and its semantic segmentation mask
- **Output:** A generated image of the same area visualized as a "smart city" with planned layout, modern architecture, green spaces, and organized infrastructure
- **Method:** ControlNet + Stable Diffusion Inpainting with optional reference images, LoRA, and quality tuning

### Key Principle

**Diffusion is NOT the decision-maker.** Urban analysis, suggestions, and constraints are computed BEFORE image generation from the segmentation mask. Image generation serves only as visualization.

### Main Script

- `src/generate_satellite_image.py` – Single image generation, batch processing, analysis, and report generation

### Supporting Scripts

- `src/compute_coverage.py` – Coverage analysis from segmentation mask
- `src/test_dataset_size.py` – Dataset utilities

---

## 2. Original Problems and Context

### Problem 1: ControlNet vs Prompt Conflict

**Issue:** The prompt ("smart/planned city") conflicted with the ControlNet guidance from an unplanned input image.

- ControlNet preserves the structure of its input. An unplanned reference image (chaotic roads, irregular building clusters) forces the model to keep that chaos.
- The prompt cannot overcome ControlNet when conditioning is strong. The model tried to fit "smart city" style into an unplanned layout.
- Result: Buildings appeared as undefined, blocky, brown structures—more like a dense shantytown than a futuristic city.

**Solution:** Use a separate **reference image** of a planned city (e.g., Barcelona Eixample, modern urban center) for ControlNet structure (Canny and depth), while the unplanned image and mask define what to edit and composite.

### Problem 2: Muddy / Low-Detail Output

**Issue:** Macro layout was correct, but micro details were absent. City blocks looked like blurry, brown texture patches rather than distinct buildings.

**Causes:**

1. **Resolution bottleneck:** At 512x768, individual buildings are only a few pixels. The model lacked pixel space for sharp edges, windows, or roofs.
2. **ControlNet over-guidance:** ControlNet weight 1.0 forced the model to follow every messy line from the chaotic input.
3. **Weak prompting:** A generic "smart city" prompt didn't specify structure, materials, or lighting. The model defaulted to brown/grey texture.
4. **Sampler choice:** Euler a (ancestral) adds random noise at every step, producing a softer, painterly look unsuitable for sharp architecture.

**Solutions Implemented:**

1. **Hires Fix:** Second pass at 2x resolution with denoising strength 0.45 to add crisp detail
2. **Lower ControlNet scales:** Canny 0.7, Seg 0.75, Depth 0.6; `control_guidance_end=0.7` so ControlNet stops at 70% and the model refines in the final 30%
3. **Sharper prompts:** Barcelona Eixample style, crisp white architecture, blue glass roofs, sharp edges, 8k, distinct buildings
4. **Scheduler:** DPM++ 2M Karras (default) or standard Euler; Euler a removed for architecture

### Problem 3: Model Knowledge Gap

**Issue:** Standard Stable Diffusion is trained on portraits and landscapes, not satellite/top-down views. It doesn't know what a "smart city house" looks like from above.

**Solution:** Support for `--model` (custom checkpoint) and `--lora` (satellite/aerial LoRA from Civitai).

---

## 3. Architecture and Pipeline Flow

### High-Level Flow

```
Input: image + mask
    │
    ├──► PART 1: Spatial Analysis
    │       - compute_coverage(mask)
    │       - compute_spatial_metrics(label_mask)
    │       - analyze_urban_layout(coverage, spatial_metrics)
    │       - generate_suggestions(analysis)
    │       - [optional] create_report()
    │
    ├──► PART 2: Prepare for Generation
    │       - load_image_and_mask(image, mask, size)
    │       - create_class_masks(label_mask)
    │       - create_inpainting_mask(editable_classes)
    │       - create_immutable_mask(immutable_classes)
    │       - extract_canny_edges(image) [from reference or unplanned]
    │       - create_seg_control_image(label_mask)
    │       - [optional] extract_depth_image(reference/unplanned)
    │
    ├──► PART 3: Generation
    │       - SmartCitySatelliteGenerator.generate_smart_city_image(...)
    │       - [optional] denoise_generated_regions()
    │       - [optional] sharpen_generated_regions()
    │       - composite_with_original(generated, original, mask, immutable_mask)
    │       - [optional] Hires Fix: second pass at 2x
    │       - [optional] simple Lanczos upscale
    │
    └──► Output: *_input.png, *_mask.png, *_output.png, [*_reference.png, *_reference_canny.png]
```

### Reference Image Logic

- **If `--reference_image` provided or auto-detected from `reference_image/`:** Canny and depth for ControlNet are extracted from the reference image (planned city layout).
- **Inpainting and segmentation:** Always use the unplanned image and its mask (what to edit, which pixels to preserve).
- **Result:** Layout/structure from reference; which regions are edited and composited from unplanned image.

### Compositing

- **Editable regions** (Residential, Unused Land, Agricultural): Replaced by generated pixels.
- **Immutable regions** (Road, River, Forest): Preserved exactly from the original image at the pixel level.

---

## 4. Configuration and Constants

### Default Paths (in `generate_satellite_image.py`)

| Constant | Value | Description |
|----------|-------|-------------|
| `DEFAULT_IMAGE_PATH` | `"../dataset/images/output_337.png"` | Default input satellite image |
| `DEFAULT_MASK_PATH` | `"../dataset/masks/output_337.png"` | Default segmentation mask |
| `DEFAULT_REFERENCE_IMAGE_PATH` | `"reference_image/reference_image_elche_alicante_spain.jpg"` | Default planned-city reference (relative to project root) |

### Segmentation Classes (BGR Label Mapping)

| Label ID | Class Name | BGR Color | Status |
|----------|------------|-----------|--------|
| 1 | Residential Area | (128, 0, 0) | Editable |
| 2 | Road | (0, 128, 0) | Immutable |
| 3 | River | (0, 0, 128) | Immutable |
| 4 | Forest | (0, 128, 128) | Immutable |
| 5 | Unused Land | (128, 128, 0) | Editable |
| 6 | Agricultural Area | (128, 0, 128) | Editable |

- **EDITABLE_CLASSES:** Residential Area, Unused Land, Agricultural Area  
- **IMMUTABLE_CLASSES:** Road, River, Forest

### ADE20K Palette (for Segmentation ControlNet)

Our 6 land-use classes map to ADE20K palette indices (for `lllyasviel/control_v11p_sd15_seg`):

| Our Label ID | Class | ADE20K Index | ADE20K Class |
|--------------|-------|--------------|--------------|
| 1 | Residential Area | 1 | building (gray) |
| 2 | Road | 6 | road (olive) |
| 3 | River | 21 | water (blue) |
| 4 | Forest | 5 | tree (green) |
| 5 | Unused Land | 10 | grass (bright green) |
| 6 | Agricultural Area | 12 | field (yellow-green) |

### Inpainting Mask

- **Editable regions:** White (255) = pixels to regenerate (Residential, Unused Land, Agricultural).
- **Preserved regions:** Black (0) = pixels kept from original.
- **Boundary erosion:** 2-pixel erosion at editable boundaries to reduce diffusion bleeding into immutable regions.

### ControlNet Models Used

| ControlNet | Model ID | Used For |
|------------|----------|----------|
| Canny | `lllyasviel/control_v11p_sd15_canny` | Layout/edges |
| Segmentation | `lllyasviel/control_v11p_sd15_seg` | Semantic regions |
| Depth | `lllyasviel/control_v11f1p_sd15_depth` | Lighting (optional) |

### Base Pipeline

- `StableDiffusionControlNetInpaintPipeline` from `runwayml/stable-diffusion-inpainting` (or `--model`).

---

## 5. Prompts

### DEFAULT_PROMPT (Positive)

```
aerial satellite photography of futuristic planned smart city, 
organized grid layout, Barcelona Eixample style urban planning, 
crisp white modern architecture, blue glass roofs, solar panels, 
lush green parks between blocks, wide paved boulevards, 
highly detailed buildings, sharp edges, clear windows, 
8k resolution, photorealistic, cinematic lighting, bright daylight, 
hyper-realistic, ultra-detailed, vibrant colors, surrounded by dense green forest, 
sharp focus, distinct modern buildings, clear architectural details, hard edges
```

**Rationale:** "Barcelona Eixample style" triggers planned blocks; "crisp white" and "blue glass" avoid muddy browns; "sharp edges", "8k", "clear windows" enforce detail.

### DEFAULT_NEGATIVE_PROMPT

```
blurry, noisy, grainy, brown mud, lowres, messy, chaotic, organic growth, 
shantytown, slums, ruins, smog, haze, fog, distorted buildings, 
low quality, worst quality, JPEG artifacts, text, watermark, signature, 
indistinct shapes, monochrome, dark, gloomy, 
abstract, painting, brush strokes, distorted, melted, surreal, texture hallucination, 
clouds, cloud cover, cloudy, overcast, mist, atmospheric effects, 
watercolor, painted, soft edges, smudged, amorphous shapes, fuzzy, out of focus, 
informal settlement, unplanned sprawl, cramped housing, dense slum, 
overgrown vacant lot, barren land
```

**Rationale:** "brown mud", "shantytown", "slums", "indistinct shapes" actively suppress muddy output.

### CLASS_PROMPTS (per Editable Class)

| Class | Prompt Fragment |
|-------|-----------------|
| Residential Area | planned grid blocks, crisp white modern buildings, blue glass roofs, clear windows, sharp edges, solar panels, green rooftops, tree-lined streets |
| Unused Land | lush green parks, recreational areas, playgrounds, sports fields, walking paths, community garden, vibrant green space |
| Agricultural Area | modern precision agriculture, organized crop fields, greenhouses, structured farm layout, irrigation patterns |

- `build_smart_city_prompt(coverage, base_prompt)` concatenates the base prompt with class prompts for classes with >0 coverage; truncates to ~280 chars for CLIP.

---

## 6. Features Implemented

### 6.1 Reference Image Support

- **`--reference_image`:** Path to planned-city image for ControlNet structure (Canny and depth).
- **Default:** If `reference_image/` folder exists, first image found is used automatically.
- **`--no_reference`:** Disable reference; use only unplanned image for ControlNet.
- **Output:** `*_reference.png` and `*_reference_canny.png` saved when reference is used.

### 6.2 ControlNet Configuration

- **Canny ControlNet:** Always on. Extracted from reference (if used) or unplanned image. Default scale: 0.7.
- **Segmentation ControlNet:** On by default. From unplanned mask. Default scale: 0.75. `--no_seg_control` to disable (logs warning).
- **Depth ControlNet:** Optional via `--use_depth_control`. Default scale: 0.6.
- **`control_guidance_start`:** 0.0 (default).
- **`control_guidance_end`:** 0.7 (default). ControlNet stops at 70% of steps so the model can refine in the final 30%.

### 6.3 Model and LoRA

- **`--model`:** Base model (HuggingFace ID or path to `.ckpt`/`.safetensors`). Default: `runwayml/stable-diffusion-inpainting`.
- **`--lora`:** Path to LoRA weights (e.g., satellite/aerial LoRA from Civitai).
- **`--lora_scale`:** LoRA strength (default 0.8). Passed as `cross_attention_kwargs={"scale": X}`.

### 6.4 Schedulers

- **`dpm++2m` (default):** DPMSolverMultistepScheduler with Karras sigmas, algorithm_type `dpmsolver++`. Recommended for sharp buildings.
- **`euler`:** EulerDiscreteScheduler (deterministic). Alternative for sharp output.
- **Euler a removed:** Ancestral variants add random noise and produce softer, painterly results; avoided for architecture.

### 6.5 Hires Fix (Two-Pass)

- **`--hires_fix`:** Enables two-pass generation.
- **Pass 1:** Generate at base size (e.g., 768×768).
- **Pass 2:** Lanczos upscale to 2×; run pipeline again at 2× with `strength=0.45` (or `--hires_fix_strength`).
- **`--hires_fix_scale`:** Upscale factor (default 2).
- **`--hires_fix_strength`:** Denoising strength for second pass (default 0.45; 0.4–0.5 recommended).
- **`--hires_fix_steps`:** Second-pass steps (default 25).
- **VRAM:** ~4× base size for 2× resolution.

### 6.6 Simple Upscale

- **`--upscale`:** Post-upscale factor (1 or 2). Lanczos only; no second diffusion pass. Ignored when `--hires_fix` is used.

### 6.7 Post-Processing

- **`--denoise`:** Light bilateral denoising on generated regions. `--denoise_strength` (default 0.3).
- **`--sharpen`:** Unsharp mask on generated regions. `--sharpen_amount` (default 0.3).

### 6.8 Low VRAM Mode

- **`--low_vram`:** Enables model CPU offload. `--size` is capped at 512 when used.

### 6.9 Batch Processing

- **`--batch`:** Process all image-mask pairs in `dataset/images/` and `dataset/masks/` (or `--dataset_dir`).

### 6.10 Analysis Only

- **`--analysis_only`:** Run spatial analysis and generate report; skip image generation.

### 6.11 IP-Adapter (Experimental)

- **`--use_ip_adapter`:** Enable IP-Adapter for texture/lighting consistency. May conflict with ControlNet+inpainting.
- **`--ip_adapter_scale`:** Default 0.6.

---

## 7. Command-Line Arguments

### Input/Output

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--image` | str | `DEFAULT_IMAGE_PATH` | Input satellite image path |
| `--mask` | str | `DEFAULT_MASK_PATH` | Segmentation mask path |
| `--output` | str | `output/smart_city_generated.png` | Output image path |

### Prompts

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--prompt` | str | `DEFAULT_PROMPT` | Positive prompt |
| `--negative_prompt` | str | `DEFAULT_NEGATIVE_PROMPT` | Negative prompt |

### Generation

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--steps` | int | 35 | Inference steps |
| `--guidance_scale` | float | 7.0 | CFG scale (5–8 recommended) |
| `--strength` | float | 0.60 | Inpainting strength (0.5–0.8) |
| `--seed` | int | None | Random seed |
| `--size` | int | 768 | Image size (512 with `--low_vram`) |

### ControlNet

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--controlnet_scale` | float | 0.7 | Canny ControlNet scale |
| `--seg_scale` | float | 0.75 | Segmentation ControlNet scale |
| `--depth_scale` | float | 0.6 | Depth ControlNet scale |
| `--no_seg_control` | flag | False | Disable segmentation ControlNet |
| `--use_depth_control` | flag | False | Enable depth ControlNet |
| `--control_guidance_start` | float | 0.0 | ControlNet start (0–1) |
| `--control_guidance_end` | float | 0.7 | ControlNet end (0–1) |

### Canny

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--canny_low` | int | 50 | Canny low threshold |
| `--canny_high` | int | 120 | Canny high threshold |

### Model/LoRA/Scheduler

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--model` | str | `runwayml/stable-diffusion-inpainting` | Base model |
| `--lora` | str | None | LoRA path |
| `--lora_scale` | float | 0.8 | LoRA strength |
| `--scheduler` | str | `dpm++2m` | `dpm++2m` or `euler` |

### Reference Image

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--reference_image` | str | None | Planned-city reference path |
| `--no_reference` | flag | False | Do not use reference |

### Resolution/Upscaling

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--upscale` | int | 1 | Post-upscale factor (1 or 2) |
| `--hires_fix` | flag | False | Two-pass Hires Fix |
| `--hires_fix_scale` | int | 2 | Hires Fix upscale factor |
| `--hires_fix_strength` | float | 0.45 | Hires Fix denoising strength |
| `--hires_fix_steps` | int | 25 | Hires Fix second-pass steps |

### Post-Processing

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--denoise` | flag | False | Bilateral denoising |
| `--denoise_strength` | float | 0.3 | Denoise blend strength |
| `--sharpen` | flag | False | Unsharp mask |
| `--sharpen_amount` | float | 0.3 | Sharpen strength |

### Performance

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--low_vram` | flag | False | CPU offload for low VRAM |
| `--use_ip_adapter` | flag | False | Enable IP-Adapter |
| `--ip_adapter_scale` | float | 0.6 | IP-Adapter scale |

### Batch/Analysis

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--batch` | flag | False | Batch process dataset |
| `--dataset_dir` | str | None | Dataset path (default: project_root/dataset) |
| `--analysis_only` | flag | False | Analysis and report only |

---

## 8. Run Examples

### Default

```bash
python src/generate_satellite_image.py
```

### With Explicit Paths

```bash
python src/generate_satellite_image.py \
  --image dataset/images/output_337.png \
  --mask dataset/masks/output_337.png \
  --output output/smart_city_generated.png
```

### Low VRAM (4GB GPUs)

```bash
python src/generate_satellite_image.py --low_vram
```

### Hires Fix (Crisp Buildings)

```bash
python src/generate_satellite_image.py --hires_fix
```

### Batch

```bash
python src/generate_satellite_image.py --batch
```

### No Reference (Original Behavior)

```bash
python src/generate_satellite_image.py --no_reference
```

### Full Example (Best for Crisp Buildings)

```bash
python src/generate_satellite_image.py \
  --image dataset/images/output_337.png \
  --mask dataset/masks/output_337.png \
  --output output/smart_city_generated.png \
  --reference_image reference_image/barcelona_city.jpg \
  --size 768 \
  --steps 35 \
  --guidance_scale 7.0 \
  --controlnet_scale 0.7 \
  --seg_scale 0.75 \
  --control_guidance_end 0.7 \
  --scheduler dpm++2m \
  --hires_fix \
  --hires_fix_strength 0.45 \
  --denoise \
  --sharpen \
  --seed 42
```

---

## 9. Output Files

| File | Description |
|------|-------------|
| `*_input.png` | Input satellite image (resized to `--size`) |
| `*_mask.png` | Segmentation mask |
| `*_output.png` | Generated smart city image |
| `*_reference.png` | Reference image used (when `--reference_image`) |
| `*_reference_canny.png` | Canny edges from reference (when used) |
| `*_report.md` | Analysis report (when `--analysis_only`) |

---

## 10. File Structure

```
Stable-diffusion-with-control-net/
├── src/
│   ├── generate_satellite_image.py   # Main pipeline
│   ├── compute_coverage.py           # Coverage from mask
│   └── test_dataset_size.py          # Dataset utilities
├── dataset/
│   ├── images/                       # Input satellite images
│   └── masks/                        # Segmentation masks
├── reference_image/                  # Planned-city reference(s)
├── output/                           # Generated images
├── requirements.txt
├── README.md
├── USAGE_EXAMPLES.md                 # Run examples
├── PROJECT_DOCUMENTATION.md          # This file
├── GETTING_STARTED.md
├── INDEX.md
├── PROJECT_OVERVIEW.md
├── SETUP.md
├── demo.sh
├── quickstart.sh
├── validate.sh
└── run.sh
```

---

## 11. Dependencies

From `requirements.txt`:

- torch >= 2.0.0
- torchvision >= 0.15.0
- diffusers >= 0.24.0
- transformers >= 4.33.0
- controlnet-aux >= 0.0.7
- Pillow >= 10.0.0
- numpy >= 1.24.0
- scipy >= 1.10.0
- opencv-python >= 4.8.0
- matplotlib >= 3.7.0
- tqdm >= 4.66.0
- accelerate >= 0.24.0
- safetensors >= 0.4.0
- xformers >= 0.0.20
- omegaconf >= 2.3.0

---

## Summary of Changes Made (Chronological)

1. **Reference image support:** `--reference_image`, default from `reference_image/`, `--no_reference`
2. **Updated prompts:** Barcelona Eixample, crisp white, blue glass, sharp edges; negative: brown mud, shantytown, indistinct shapes
3. **ControlNet tuning:** Lower scales (Canny 0.7, Seg 0.75, Depth 0.6), `control_guidance_end=0.7`
4. **Resolution:** Default `--size` 768; `--low_vram` caps at 512
5. **Scheduler:** DPM++ 2M Karras default; Euler (non-ancestral) option; Euler a removed
6. **Model/LoRA:** `--model`, `--lora`, `--lora_scale`
7. **Hires Fix:** `--hires_fix` with two-pass at 2x, denoising 0.45
8. **Simple upscale:** `--upscale` 2 (Lanczos)
9. **Post-processing:** `--denoise`, `--sharpen`
10. **Documentation:** USAGE_EXAMPLES.md, README, GETTING_STARTED, INDEX, demo.sh, quickstart.sh updated

---

*Document generated for project documentation. Last updated: January 2026.*
