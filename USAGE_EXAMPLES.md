# Run Examples - Smart City Satellite Image Generation

This document lists documented run examples for the smart city satellite image generation pipeline. The pipeline uses **ControlNet + Stable Diffusion Inpainting** with optional reference images, LoRA, and quality tuning.

---

## Prerequisites

- Activate virtual environment: `source venv/bin/activate`
- Place your satellite image and segmentation mask in `dataset/images/` and `dataset/masks/` (or use `--image` and `--mask` paths)
- Put a planned-city reference image in `reference_image/` to use it automatically

---

## Basic Examples

### Default run (recommended)

Uses 768x768 resolution, lower ControlNet scales, and `control_guidance_end=0.7` for cleaner results. Reference image is auto-used if present in `reference_image/`.

```bash
python src/generate_satellite_image.py
```

Or with explicit image and mask:

```bash
python src/generate_satellite_image.py \
  --image dataset/images/output_337.png \
  --mask dataset/masks/output_337.png \
  --output output/smart_city_generated.png
```

---

### Low VRAM (4GB GPUs)

Automatically caps resolution at 512x512 when `--low_vram` is used.

```bash
python src/generate_satellite_image.py --low_vram
```

---

### 2x post-upscale

Upscale the output 2x with Lanczos after generation for more pixel density.

```bash
python src/generate_satellite_image.py --upscale 2
```

---

### Custom model and LoRA

Use an aerial/satellite checkpoint and LoRA from Civitai for crisper buildings.

```bash
python src/generate_satellite_image.py \
  --model path/to/aerial_checkpoint.safetensors \
  --lora path/to/satellite_lora.safetensors \
  --lora_scale 0.8
```

---

### Euler a scheduler

Use Euler Ancestral scheduler instead of DPM++ 2M Karras.

```bash
python src/generate_satellite_image.py --scheduler euler_a
```

---

## Reference Image Examples

### Use planned-city reference (default)

If `reference_image/` contains an image, it is used automatically for ControlNet structure (Canny and depth). Inpainting still uses your unplanned image and mask.

```bash
# Reference is auto-detected from reference_image/
python src/generate_satellite_image.py \
  --image dataset/images/unplanned.png \
  --mask dataset/masks/unplanned.png
```

---

### Explicit reference image

```bash
python src/generate_satellite_image.py \
  --image dataset/images/unplanned.png \
  --mask dataset/masks/unplanned.png \
  --reference_image reference_image/barcelona_city.jpg
```

---

### No reference (original behavior)

Use only the unplanned image for ControlNet structure.

```bash
python src/generate_satellite_image.py --no_reference
```

---

## Batch Processing

Process all image-mask pairs in the dataset directory.

```bash
python src/generate_satellite_image.py --batch
```

With custom dataset directory:

```bash
python src/generate_satellite_image.py --batch --dataset_dir /path/to/dataset
```

---

## Quality Tuning Examples

### Lower ControlNet strength (more prompt freedom)

```bash
python src/generate_satellite_image.py \
  --controlnet_scale 0.6 \
  --seg_scale 0.7 \
  --control_guidance_end 0.65
```

---

### Higher CFG (stricter prompt)

```bash
python src/generate_satellite_image.py --guidance_scale 8.0
```

---

### More inference steps

```bash
python src/generate_satellite_image.py --steps 40
```

---

### Reproducible seed

```bash
python src/generate_satellite_image.py --seed 42
```

---

### Denoise and sharpen post-processing

```bash
python src/generate_satellite_image.py --denoise --sharpen
```

---

## Analysis Only (no generation)

Run spatial analysis and generate a report without generating images.

```bash
python src/generate_satellite_image.py --analysis_only
```

---

## Full Example (all options)

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
  --upscale 2 \
  --denoise \
  --sharpen \
  --seed 42
```

---

## Output Files

For each run, the pipeline saves to the output directory:

| File | Description |
|------|-------------|
| `*_input.png` | Input satellite image (resized) |
| `*_mask.png` | Segmentation mask |
| `*_output.png` | Generated smart city image |
| `*_reference.png` | Reference image used (if `--reference_image`) |
| `*_reference_canny.png` | Canny edges from reference (if used) |

---

## See Also

- `python src/generate_satellite_image.py --help` for all CLI options
- `README.md` for setup and overview
- `GETTING_STARTED.md` for first-time setup
