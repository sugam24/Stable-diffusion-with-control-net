# 🎨 ControlNet + Stable Diffusion Project - Complete Setup Guide

**Status:** ✅ **READY TO USE**  
**Setup Date:** January 20, 2026  
**Python Version:** 3.13  
**PyTorch Version:** 2.9.1 (CUDA 12.8)

---

## 📋 Table of Contents

1. [Project Overview](#project-overview)
2. [What's Installed](#whats-installed)
3. [Quick Start](#quick-start)
4. [Project Structure](#project-structure)
5. [Usage Examples](#usage-examples)
6. [Smart City Prompts](#smart-city-prompts)
7. [Advanced Configuration](#advanced-configuration)
8. [Troubleshooting](#troubleshooting)

---

## 🎯 Project Overview

This project combines **ControlNet** and **Stable Diffusion** to generate realistic satellite images of modern smart cities.

**Key Features:**

- ✅ Uses edge detection to guide image generation
- ✅ Accepts satellite images as input for controlled generation
- ✅ Generates new variations of satellite images based on text prompts
- ✅ Supports batch processing of multiple images
- ✅ Full CUDA GPU acceleration
- ✅ Customizable prompts for various smart city scenarios

---

## 📦 What's Installed

### Core Libraries

```
torch (2.9.1)              - Deep learning framework with CUDA support
diffusers (0.36.0)         - Pre-trained diffusion models library
transformers (4.57.6)      - Hugging Face transformers
controlnet-aux (0.0.10)    - ControlNet utilities
torchvision (0.24.1)       - Computer vision utilities
accelerate (1.12.0)        - Distributed training utilities
```

### Image Processing

```
Pillow (12.1.0)            - Image manipulation
opencv-python (4.13.0)     - Computer vision
scikit-image (0.26.0)      - Image processing algorithms
```

### Optimization

```
xformers (0.0.33)          - Memory-efficient attention mechanisms
safetensors (0.7.0)        - Safe tensor serialization
```

### Utilities

```
numpy (2.4.1)              - Numerical computing
matplotlib (3.10.8)        - Visualization
tqdm (4.67.1)              - Progress bars
```

---

## 🚀 Quick Start

### 1. Activate Virtual Environment

```bash
cd ~/Desktop/controlnet+stable_diffusion
source venv/bin/activate
```

### 2. Add Your Satellite Image

```bash
cp your_satellite_image.png dataset/
```

### 3. Generate Smart City Image

```bash
python3 src/generate_satellite_image.py \
  --input dataset/your_satellite_image.png \
  --prompt "modern smart city with solar panels, green spaces, 5G infrastructure" \
  --output output/generated_smart_city.png
```

### 4. Check Results

```bash
# Generated images are in output/ folder
ls -lh output/
```

---

## 📁 Project Structure

```
controlnet+stable_diffusion/
│
├── 📥 dataset/                      # Place your satellite images here
│   └── (add your .png, .jpg, etc.)
│
├── 📤 output/                       # Generated images saved here
│   └── (results appear here after generation)
│
├── 🐍 venv/                         # Python virtual environment (ready)
│   ├── bin/
│   ├── lib/
│   └── pyvenv.cfg
│
├── 📂 src/                          # Main Python code
│   ├── generate_satellite_image.py  # Single image generation script
│   │                                # Use: python3 src/generate_satellite_image.py --help
│   │
│   └── batch_generate.py            # Batch processing script
│                                    # Use: python3 src/batch_generate.py --help
│
├── 📋 requirements.txt               # Python dependencies list
│
├── 📖 README.md                      # Comprehensive documentation
│
├── 📝 SETUP.md                       # Setup instructions and parameters
│
├── 🚀 quickstart.sh                  # Quick reference guide
│
├── 🎬 demo.sh                        # Demo and usage examples
│
└── .gitignore                        # Git configuration
```

---

## 💻 Usage Examples

### Example 1: Generate from Satellite Image

```bash
source venv/bin/activate

python3 src/generate_satellite_image.py \
  --input dataset/city_satellite.png \
  --prompt "modern smart city with solar panels, electric vehicles, green infrastructure" \
  --output output/smart_city_v1.png \
  --steps 30 \
  --guidance_scale 7.5
```

### Example 2: Batch Process Multiple Images

```bash
# Place multiple satellite images in dataset/ folder first
python3 src/batch_generate.py \
  --dataset_dir dataset \
  --prompt "sustainable smart city with IoT sensors and renewable energy" \
  --output_dir output \
  --steps 25 \
  --guidance_scale 7.5
```

### Example 3: Generate from Scratch

```bash
# No input image needed
python3 src/generate_satellite_image.py \
  --prompt "aerial view of a futuristic smart city with autonomous transportation" \
  --output output/generated_from_scratch.png \
  --steps 30
```

### Example 4: With Custom Seed for Reproducibility

```bash
python3 src/generate_satellite_image.py \
  --input dataset/city.png \
  --prompt "smart city" \
  --output output/result_v1.png \
  --seed 42  # Same seed = same result
```

### Example 5: High Quality Generation

```bash
python3 src/generate_satellite_image.py \
  --input dataset/city.png \
  --prompt "professional satellite image of smart city infrastructure" \
  --output output/high_quality.png \
  --steps 50 \
  --guidance_scale 10
```

---

## 🏙️ Smart City Prompts

### Prompt 1: Modern Urban Planning

```
"aerial satellite view of a modern smart city featuring:
- organized grid layout with parks and green spaces
- solar panel arrays covering 40% of rooftops
- electric vehicle charging stations at regular intervals
- smart traffic light infrastructure visible
- 5G communication towers strategically placed
- efficient water management systems
- bike lanes and pedestrian zones integrated
- high quality, detailed, professional satellite imagery"
```

### Prompt 2: Sustainability & Green Infrastructure

```
"top-down view of an eco-friendly smart city with:
- extensive solar panel installations on all buildings
- vertical gardens and rooftop vegetation
- wind turbine farms at city periphery
- rainwater harvesting systems integrated
- underground waste management infrastructure
- carbon-neutral buildings with green certification
- protected natural habitats within urban area
- clean energy generation visible from above
- high quality, detailed aerial photograph"
```

### Prompt 3: Technology Integration

```
"satellite image of an advanced smart city featuring:
- AI surveillance and monitoring systems network
- autonomous vehicle transportation corridors
- smart traffic management infrastructure
- IoT sensor deployment across streets
- fiber optic network infrastructure visible
- real-time data flow visualization
- digital information hubs and kiosks
- 5G antenna network optimization
- cybersecurity infrastructure indicators
- professional, high quality satellite view"
```

### Prompt 4: Futuristic Vision

```
"aerial view of a futuristic smart city with:
- innovative architectural designs
- drone delivery zones clearly marked
- autonomous transportation hubs
- mixed reality integration infrastructure
- sustainable energy generation systems
- intelligent building automation visible
- connected digital infrastructure
- space-age transportation networks
- advanced urban planning visible
- stunning, professional, high quality imagery"
```

### Prompt 5: Night-time Smart City

```
"satellite view of a smart city at night featuring:
- intelligent street lighting creating optimal patterns
- communication network lights
- autonomous vehicles with navigation lights
- building automation systems illuminated
- data center and server facility indicators
- energy management system efficiency visible
- cybersecurity monitoring indicators
- real-time city operations visualization
- professional thermal and night vision imagery"
```

---

## ⚙️ Advanced Configuration

### Command Line Parameters

| Parameter           | Type  | Range | Default                    | Purpose                                 |
| ------------------- | ----- | ----- | -------------------------- | --------------------------------------- |
| `--input`           | str   | N/A   | None                       | Path to input satellite image           |
| `--prompt`          | str   | N/A   | Required                   | Text prompt for generation              |
| `--output`          | str   | N/A   | output/generated_image.png | Output file path                        |
| `--negative_prompt` | str   | N/A   | "blurry, low quality"      | What to avoid generating                |
| `--steps`           | int   | 20-50 | 30                         | Inference steps (more = higher quality) |
| `--guidance_scale`  | float | 5-15  | 7.5                        | Prompt adherence strength               |
| `--seed`            | int   | Any   | None                       | Random seed for reproducibility         |

### Performance Recommendations

**Fast Mode (Draft Quality)**

```bash
--steps 20 --guidance_scale 5
# Time: ~30-40 seconds on GPU
```

**Balanced Mode (Recommended)**

```bash
--steps 30 --guidance_scale 7.5
# Time: ~45-60 seconds on GPU
```

**High Quality Mode (Best Results)**

```bash
--steps 50 --guidance_scale 10
# Time: ~2-3 minutes on GPU
```

---

## 🔧 Troubleshooting

### Issue: CUDA Out of Memory

**Solution:**

```bash
# Reduce inference steps
python3 src/generate_satellite_image.py \
  --input dataset/image.png \
  --prompt "smart city" \
  --steps 15  # Reduced from 30
```

### Issue: Very Slow Generation (CPU Usage)

**Solution:**

```bash
# Check CUDA availability
python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# If False, install CUDA drivers or use pre-installed CUDA
# Current: ✅ CUDA Available
```

### Issue: Model Download Fails

**Solution:**

```bash
# Clear cache
rm -rf ~/.cache/huggingface/

# Retry download
python3 src/generate_satellite_image.py --input dataset/image.png --prompt "test"
```

### Issue: Poor Quality Output

**Solution:**

```bash
# Try these adjustments:
# 1. Increase steps
--steps 40

# 2. Adjust guidance scale
--guidance_scale 10  # More strict, or 5-7 for creativity

# 3. Use better prompt with details
--prompt "high quality satellite image of smart city with..."

# 4. Use better input image if available
```

### Issue: Can't Find Input Image

**Solution:**

```bash
# Check file exists
ls -lh dataset/your_image.png

# Ensure full path
python3 src/generate_satellite_image.py \
  --input dataset/your_image.png  # Use relative or absolute path
```

---

## 📊 System Information

```
OS: Linux
Python: 3.13
PyTorch: 2.9.1
CUDA: Available (12.8)
GPU: Enabled for acceleration
Device Memory: Optimized with xformers
```

---

## 📈 Expected Performance

| Operation                   | Time (GPU) | Time (CPU) | Quality   |
| --------------------------- | ---------- | ---------- | --------- |
| Single image, 20 steps      | ~30s       | ~5-10m     | Draft     |
| Single image, 30 steps      | ~45s       | ~7-15m     | Good      |
| Single image, 50 steps      | ~75s       | ~10-20m    | Excellent |
| Batch (10 images, 30 steps) | ~8m        | ~1.5-2h    | Good      |

---

## ✨ Next Steps

1. **Add Data**: Place satellite images in `dataset/` folder
2. **Test**: Run a simple generation with default settings
3. **Iterate**: Try different prompts and parameters
4. **Optimize**: Find your sweet spot for quality vs. speed
5. **Scale**: Use batch processing for multiple images

---

## 📞 Support

For detailed information:

- See **README.md** for comprehensive docs
- See **SETUP.md** for setup details
- Run **./demo.sh** for demo and examples
- Run **./quickstart.sh** for quick reference

---

## 🎉 Project Status

**✅ Installation Complete**  
**✅ Virtual Environment Ready**  
**✅ All Dependencies Installed**  
**✅ CUDA/GPU Enabled**  
**✅ Dataset Folder Created**  
**✅ Output Folder Created**  
**✅ Scripts Ready to Use**

**You're all set to generate smart city satellite images!** 🚀

---

_Setup completed successfully on January 20, 2026_  
_Project location: /home/sugam/Desktop/controlnet+stable_diffusion_
