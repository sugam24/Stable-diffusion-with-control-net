# 📚 ControlNet + Stable Diffusion Project - Complete Documentation Index

## 🎯 Start Here

**New to this project?** Start with one of these:

1. **[GETTING_STARTED.md](GETTING_STARTED.md)** - Quick start (5 min read)
2. **[USAGE_EXAMPLES.md](USAGE_EXAMPLES.md)** - Run examples (documented commands)
3. **[README.md](README.md)** - User guide (15 min read)
4. **[PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md)** - Complete overview (20 min read)

---

## 📖 Documentation Map

### Quick Reference

| File                   | Time  | Purpose                             |
| ---------------------- | ----- | ----------------------------------- |
| **GETTING_STARTED.md** | 5 min | First-time setup and quick examples |
| **USAGE_EXAMPLES.md**  | 5 min | Documented run examples             |
| **quickstart.sh**      | 1 min | Print quick reference to terminal   |
| **demo.sh**            | 1 min | Show usage examples                 |

### Detailed Documentation

| File                    | Time   | Purpose                                    |
| ----------------------- | ------ | ------------------------------------------ |
| **README.md**           | 15 min | Full user guide with troubleshooting       |
| **SETUP.md**            | 15 min | Detailed setup instructions and parameters |
| **PROJECT_OVERVIEW.md** | 20 min | Complete project reference guide           |

### Verification

| File                 | Time  | Purpose                    |
| -------------------- | ----- | -------------------------- |
| **validate.sh**      | 1 min | Run project health check   |
| **requirements.txt** | -     | List of installed packages |

---

## 🚀 Quick Start (Copy-Paste)

```bash
# 1. Navigate to project
cd /path/to/Stable-diffusion-with-control-net
source venv/bin/activate

# 2. Add image and mask to dataset/images/ and dataset/masks/
# (e.g., cp your_image.png dataset/images/ && cp your_mask.png dataset/masks/)

# 3. Generate smart city image (default run)
python src/generate_satellite_image.py

# Or with explicit paths:
python src/generate_satellite_image.py \
  --image dataset/images/your_image.png \
  --mask dataset/masks/your_image.png \
  --output output/generated_smart_city.png

# 4. Check output/ folder for results!
```

See **[USAGE_EXAMPLES.md](USAGE_EXAMPLES.md)** for full documented run examples.

---

## 🎨 Python Scripts

### Main Generation Script

**`src/generate_satellite_image.py`** (280 lines)

- Single image generation
- Accepts satellite images as control input
- Uses ControlNet + Stable Diffusion
- Full command-line interface

**Usage:**

```bash
python src/generate_satellite_image.py --help
python src/generate_satellite_image.py --image dataset/images/img.png --mask dataset/masks/img.png
```

### Batch Processing (same script)

Use `--batch` to process all image-mask pairs in `dataset/`:

```bash
python src/generate_satellite_image.py --batch
```

---

## 📂 Project Structure

```
/home/sugam/Desktop/controlnet+stable_diffusion/
│
├── 📥 dataset/                    ← Your satellite images go here
│
├── 📤 output/                     ← Generated images appear here
│
├── 🐍 venv/                       ← Python virtual environment
│   └── bin/activate               ← Activate with: source venv/bin/activate
│
├── 🎨 GENERATION SCRIPTS
│   └── src/
│       ├── generate_satellite_image.py ← Single + batch generation
│       └── compute_coverage.py         ← Coverage analysis
│
├── 📖 DOCUMENTATION
│   ├── GETTING_STARTED.md         ← START HERE! (Quick start)
│   ├── USAGE_EXAMPLES.md          ← Run examples (documented commands)
│   ├── README.md                  ← Full user guide
│   ├── SETUP.md                   ← Setup details
│   ├── PROJECT_OVERVIEW.md        ← Complete reference
│   └── INDEX.md                   ← This file
│
├── 🛠️ UTILITIES & CONFIG
│   ├── requirements.txt            ← Python package list
│   ├── validate.sh                 ← Health check script
│   ├── quickstart.sh               ← Quick reference
│   ├── demo.sh                     ← Demo and examples
│   └── .gitignore                  ← Git configuration
│
└── 📋 File Tree Summary (above)
```

---

## 💾 Package Information

### Core Dependencies

- **PyTorch** 2.9.1 - Deep learning framework
- **Diffusers** 0.36.0 - Diffusion models
- **Transformers** 4.57.6 - Hugging Face models
- **ControlNet-aux** 0.0.10 - Control utilities
- **OpenCV** 4.13.0 - Image processing
- **xformers** 0.0.33 - Memory optimization

### Installation

```bash
# Recreate environment if needed:
pip install -r requirements.txt

# Check installation:
./validate.sh
```

---

## 🎯 Common Tasks

### Task 1: Default run

```bash
source venv/bin/activate
python src/generate_satellite_image.py
```

**Time:** ~60 seconds (GPU, 768x768)

### Task 2: Batch process

```bash
python src/generate_satellite_image.py --batch
```

**Time:** ~60s × number of pairs

### Task 3: Low VRAM (4GB GPUs)

```bash
python src/generate_satellite_image.py --low_vram
```

### Task 4: With quality options

```bash
python src/generate_satellite_image.py --upscale 2 --scheduler euler_a
```

See **[USAGE_EXAMPLES.md](USAGE_EXAMPLES.md)** for all run examples.

---

## ⚙️ Parameter Quick Reference

```
--image FILE          Input satellite image path
--mask FILE           Segmentation mask path
--output FILE         Output image path
--reference_image     Planned-city reference (optional)
--size N              Image size (768 default; 512 with --low_vram)
--low_vram            CPU offload for 4GB GPUs
--upscale 2           Post-upscale 2x Lanczos
--model               Custom checkpoint
--lora                LoRA path
--scheduler           dpm++2m or euler_a
--batch               Process all image-mask pairs
```

Run `python src/generate_satellite_image.py --help` for full list.

---

## 🎨 Prompt Templates

### Template 1: Urban Planning

```
"aerial satellite view of a modern smart city with:
- organized grid layout with parks
- solar panel rooftops
- green spaces and bike lanes
- efficient transportation
- 5G infrastructure
- high quality, detailed"
```

### Template 2: Sustainability

```
"top-down view of eco-friendly smart city with:
- solar arrays and wind turbines
- vertical gardens
- rainwater harvesting
- waste management
- carbon-neutral buildings
- professional photography"
```

### Template 3: Technology

```
"satellite image of advanced smart city featuring:
- AI surveillance networks
- autonomous vehicle lanes
- smart traffic systems
- IoT sensor deployment
- fiber optic infrastructure
- high quality, detailed"
```

---

## 🔧 System Information

**Your System:**

```
OS: Linux
Python: 3.13
PyTorch: 2.9.1 (CUDA 12.8)
GPU: NVIDIA GeForce RTX 3050 Laptop (3.7 GB)
Virtual Environment: Configured ✓
All Packages: Installed ✓
```

**Performance:**

- ✅ GPU Acceleration Enabled
- ✅ 5-10x faster than CPU
- ✅ 30-75 seconds per image

---

## 🆘 Troubleshooting Quick Links

**Issue:** Slow generation
→ [README.md - Troubleshooting](README.md#troubleshooting)

**Issue:** CUDA out of memory
→ [SETUP.md - Troubleshooting](SETUP.md#troubleshooting)

**Issue:** Cannot find modules
→ [Run validate.sh](validate.sh)

**Issue:** Lost?
→ [GETTING_STARTED.md](GETTING_STARTED.md)

---

## 📊 File Size Reference

| File                            | Size | Type     |
| ------------------------------- | ---- | -------- |
| src/generate_satellite_image.py | 7.2K | Python   |
| src/batch_generate.py           | 4.1K | Python   |
| validate.sh                     | 7.3K | Script   |
| README.md                       | 5.9K | Markdown |
| SETUP.md                        | 5.7K | Markdown |
| PROJECT_OVERVIEW.md             | 12K  | Markdown |
| requirements.txt                | 247B | Config   |

**Total Project:** ~7.9 GB (mostly venv/models cache)

---

## ✅ Verification Checklist

- [x] Virtual environment created
- [x] All packages installed
- [x] GPU/CUDA enabled
- [x] Project files organized
- [x] Documentation complete
- [x] Scripts tested
- [x] Ready to use!

---

## 🚀 Getting Started Path

1. **Read:** [GETTING_STARTED.md](GETTING_STARTED.md) (5 min)
2. **Verify:** Run `./validate.sh` (1 min)
3. **Add Data:** Copy images to `dataset/` folder
4. **Generate:** Run generation script
5. **Check:** View results in `output/` folder
6. **Learn More:** Read [README.md](README.md) for advanced usage

---

## 📞 Document Navigation

**Quick Help:**

```bash
# Print quick reference
./quickstart.sh

# Show examples
./demo.sh

# Run health check
./validate.sh

# View run examples
less USAGE_EXAMPLES.md

# View full documentation
less GETTING_STARTED.md
less README.md
less PROJECT_OVERVIEW.md
```

---

## 🎉 You're All Set!

Your ControlNet + Stable Diffusion project is fully configured and ready to generate amazing smart city satellite images.

**Next Step:** Read [GETTING_STARTED.md](GETTING_STARTED.md) and start generating! 🎨✨

---

**Setup Date:** January 20, 2026  
**Status:** ✅ Ready to Use  
**Location:** /home/sugam/Desktop/controlnet+stable_diffusion
