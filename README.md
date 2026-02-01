# ControlNet + Stable Diffusion for Smart City Satellite Image Generation

This project uses **ControlNet** + **Stable Diffusion** to generate realistic satellite images of modern smart cities. You can provide satellite images as control inputs and customize prompts to generate variations.

## 📁 Project Structure

```
Stable-diffusion-with-control-net/
├── src/                             # Main Python code
│   ├── generate_satellite_image.py  # Single + batch image generation
│   └── compute_coverage.py           # Coverage analysis
├── dataset/                         # Satellite images and masks
│   ├── images/                      # Input images
│   └── masks/                       # Segmentation masks
├── reference_image/                 # Planned-city reference (optional)
├── output/                          # Generated images
├── requirements.txt
├── USAGE_EXAMPLES.md                # Run examples (see this!)
└── README.md
```

## 🚀 Setup Instructions

### 1. Create Virtual Environment

```bash
cd /home/sugam/Desktop/controlnet+stable_diffusion
python3 -m venv venv
source venv/bin/activate  # On Linux/Mac
# OR
# venv\Scripts\activate  # On Windows
```

### 2. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Note:** This will download ~10-15GB of model files on first run (Stable Diffusion weights + ControlNet model). Make sure you have enough disk space.

### 3. Verify Installation

```bash
python3 -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"
python3 -c "from diffusers import StableDiffusionControlNetPipeline; print('Diffusers installed successfully')"
```

## 🎯 Usage

The pipeline requires an **image** and **segmentation mask**. See **[USAGE_EXAMPLES.md](USAGE_EXAMPLES.md)** for full run examples.

### Basic run (default settings)

```bash
python src/generate_satellite_image.py
```

Uses defaults: `dataset/images/output_337.png`, `dataset/masks/output_337.png`, and auto-detects reference from `reference_image/`.

### With explicit image and mask

```bash
python src/generate_satellite_image.py \
  --image dataset/images/your_image.png \
  --mask dataset/masks/your_image.png \
  --output output/generated_smart_city.png
```

### Low VRAM (4GB GPUs)

```bash
python src/generate_satellite_image.py --low_vram
```

### Batch processing

```bash
python src/generate_satellite_image.py --batch
```

### Quality options

```bash
# 2x upscale, Euler scheduler, custom model
python src/generate_satellite_image.py --upscale 2 --scheduler euler_a
python src/generate_satellite_image.py --model path/to/checkpoint.safetensors --lora path/to/lora.safetensors
```

**See [USAGE_EXAMPLES.md](USAGE_EXAMPLES.md) for all documented run examples.**

## 📋 Prompt Examples for Smart Cities

### Urban Planning

```
aerial view of a modern smart city with:
- organized grid layout with green rooftops
- solar panel arrays on buildings
- EV charging stations
- tree-lined streets with public spaces
- 5G tower infrastructure visible
- sustainable water management systems
```

### Technology Integration

```
satellite image of an advanced smart city featuring:
- AI surveillance cameras mounted on poles
- autonomous vehicle lanes clearly marked
- smart traffic lights at intersections
- IoT sensor networks
- fiber optic infrastructure visualization
- real-time traffic flow optimization visible from above
```

### Sustainability

```
top-down view of an eco-friendly smart city with:
- extensive solar panel arrays
- vertical gardens and green spaces
- wind turbines in strategic locations
- rainwater harvesting systems
- waste management facilities
- carbon-neutral buildings
- protected natural areas integrated into urban planning
```

## ⚙️ Command Line Arguments

### src/generate_satellite_image.py

| Argument | Description |
|----------|-------------|
| `--image` | Path to input satellite image |
| `--mask` | Path to segmentation mask |
| `--output` | Path to save generated image |
| `--reference_image` | Planned-city image for ControlNet structure (optional) |
| `--no_reference` | Use only unplanned image (no reference) |
| `--prompt` | Text prompt for generation |
| `--negative_prompt` | What to avoid |
| `--steps` | Inference steps (30-40 recommended, default 35) |
| `--guidance_scale` | CFG scale (5-8 recommended, default 7.0) |
| `--size` | Image size (768 default; 512 with --low_vram) |
| `--low_vram` | Use CPU offload for 4GB GPUs |
| `--upscale` | Post-upscale factor (1 or 2) |
| `--model` | Custom checkpoint (HuggingFace ID or .ckpt/.safetensors) |
| `--lora` | Path to LoRA weights |
| `--scheduler` | `dpm++2m` or `euler_a` |
| `--control_guidance_end` | Stop ControlNet at this step % (default 0.7) |
| `--batch` | Process all image-mask pairs in dataset |
| `--seed` | Random seed for reproducibility |

Run `python src/generate_satellite_image.py --help` for full list.

## 📊 Expected Results

The script generates 2 outputs for each input:

1. **Generated Image**: Your new smart city satellite image
2. **Control Image**: Edge map used to guide generation

## 🎨 Tips for Better Results

1. **Quality Prompts**: Be specific about features you want to see
2. **Inference Steps**:
   - 20-25 steps: Fast, lower quality
   - 30-40 steps: Good balance
   - 50+ steps: Best quality, slower
3. **Guidance Scale**:
   - 7.5: Recommended
   - 5-7: More creative
   - 10-15: Strictly follows prompt
4. **Input Images**: Use high-quality satellite images for better control
5. **Seed**: Use the same seed for reproducible results

## 💾 Dataset Folder

Place your satellite images in the `dataset/` folder:

- Supported formats: JPG, PNG, BMP, TIFF
- Any resolution (will be resized to 512x512)
- Can be satellite, drone, or aerial images

## 🔧 Troubleshooting

### CUDA Out of Memory

```bash
# Reduce image quality or use CPU
python3 src/generate_satellite_image.py --input dataset/image.png --prompt "smart city" --steps 15
```

### Slow on CPU

- Recommended: Use GPU for 5x speedup
- Set `--steps` to 20-25 for faster generation

### Model Download Fails

- Check internet connection
- Models cache in `~/.cache/huggingface/`
- Try clearing cache: `rm -rf ~/.cache/huggingface/`

## 📦 Requirements

- Python 3.8+
- PyTorch with CUDA support (GPU recommended)
- 16GB+ RAM for GPU (8GB minimum with CPU)
- ~15GB disk space for models

## 🔗 References

- [ControlNet Paper](https://arxiv.org/abs/2302.05543)
- [Stable Diffusion Documentation](https://huggingface.co/docs/diffusers)
- [Diffusers Library](https://github.com/huggingface/diffusers)

## 📝 License

This project uses open-source models from Hugging Face. Please refer to their license terms.
