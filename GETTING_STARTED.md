# 🎉 Setup Complete - Getting Started Guide

## ✅ Validation Results

**Status:** ✅ **ALL SYSTEMS GO**

```
✓ Passed: 19/19 checks
✗ Failed: 0

System: Linux
Python: 3.13.11
PyTorch: 2.9.1 (CUDA 12.8)
GPU: NVIDIA GeForce RTX 3050 Laptop GPU (3.7 GB)
```

---

## 📦 What Was Set Up

### Virtual Environment

- ✅ Python 3.13 virtual environment created
- ✅ All 47 packages installed successfully
- ✅ GPU/CUDA acceleration enabled

### Project Structure

- ✅ `src/` folder created (main Python code)
- ✅ `dataset/` folder created (for your satellite images)
- ✅ `output/` folder created (for generated images)
- ✅ Main generation scripts ready
- ✅ Documentation complete

### Scripts Created

| Script                            | Purpose                               |
| --------------------------------- | ------------------------------------- |
| `src/generate_satellite_image.py` | Generate images from satellite photos |
| `src/batch_generate.py`           | Process multiple images at once       |
| `quickstart.sh`                   | Quick reference guide                 |
| `demo.sh`                         | Usage examples                        |
| `validate.sh`                     | System health check                   |

---

## 🚀 How to Use (3 Easy Steps)

### Step 1: Prepare Your Data

```bash
# Copy your satellite images to the dataset folder
cp your_satellite_image.png ~/Desktop/controlnet+stable_diffusion/dataset/
```

### Step 2: Activate Environment

```bash
cd ~/Desktop/controlnet+stable_diffusion
source venv/bin/activate
```

### Step 3: Generate Images

```bash
# Default run (uses dataset defaults + reference_image/ if present)
python src/generate_satellite_image.py
```

Or with explicit image and mask:

```bash
python src/generate_satellite_image.py \
  --image dataset/images/your_image.png \
  --mask dataset/masks/your_image.png \
  --output output/generated_smart_city.png
```

That's it! Check the `output/` folder for your generated images.

**See [USAGE_EXAMPLES.md](USAGE_EXAMPLES.md) for full documented run examples.**

---

## 💡 Example Commands

### Default Run (Recommended)

```bash
python src/generate_satellite_image.py
```

⏱️ **Time:** ~60 seconds (768x768)

### Low VRAM (4GB GPUs)

```bash
python src/generate_satellite_image.py --low_vram
```

### With 2x Upscale

```bash
python src/generate_satellite_image.py --upscale 2
```

### Batch Process (Multiple Image-Mask Pairs)

```bash
python src/generate_satellite_image.py --batch
```

⏱️ **Time:** ~60s × number of pairs

### Full documented examples

See **[USAGE_EXAMPLES.md](USAGE_EXAMPLES.md)** for all run examples (reference image, model, LoRA, scheduler, quality tuning, etc.).

---

## 🎨 Best Prompts for Smart Cities

### Modern Urban Planning

```
"aerial satellite view of a modern smart city with organized grid layout,
solar panel rooftops, green spaces between buildings, efficient transportation
corridors, autonomous vehicle lanes, 5G infrastructure, sustainable architecture,
high quality, detailed professional photography"
```

### Futuristic Technology

```
"satellite image of an advanced smart city featuring AI surveillance networks,
autonomous vehicle hubs, smart traffic systems, IoT sensor deployment,
fiber optic infrastructure, real-time optimization visible from above,
high quality, futuristic, detailed"
```

### Sustainability Focus

```
"top-down view of an eco-friendly smart city with solar panel arrays,
vertical gardens, wind turbines, rainwater harvesting, waste management,
carbon-neutral buildings, protected natural areas, clean energy visible,
professional high quality satellite imagery"
```

---

## ⚙️ Parameter Reference

```bash
python src/generate_satellite_image.py \
  --image dataset/images/img.png     # Input satellite image
  --mask dataset/masks/img.png       # Segmentation mask
  --output output/result.png         # Where to save
  --reference_image ref.png          # Planned-city reference (optional)
  --low_vram                         # For 4GB GPUs
  --upscale 2                        # Post-upscale 2x
  --steps 35                         # Inference steps (30-40 recommended)
  --guidance_scale 7.0               # CFG (5-8 recommended)
  --seed 42                          # Reproducibility
```

### Quick Parameter Guide

| Parameter          | Fast  | Balanced | Quality |
| ------------------ | ----- | -------- | ------- |
| `--size`           | 512   | 768      | 768     |
| `--steps`          | 30    | 35       | 40      |
| `--guidance_scale` | 6     | 7.0      | 8       |

See **[USAGE_EXAMPLES.md](USAGE_EXAMPLES.md)** for full parameter list.

---

## 📂 File Organization

```
Your workspace:
/home/sugam/Desktop/controlnet+stable_diffusion/

📁 dataset/                    ← PUT YOUR IMAGES HERE
   ├── city1.jpg
   ├── city2.png
   └── city3.jpg

📁 output/                     ← RESULTS APPEAR HERE
   ├── city1_generated.png
   ├── city1_control.png
   ├── city2_generated.png
   └── city2_control.png

📜 Main Scripts:
   ├── src/
│   ├── generate_satellite_image.py   ← Single image
   └── batch_generate.py             ← Multiple images
```

---

## 🔍 Verify Everything Works

```bash
# 1. Go to project folder
cd ~/Desktop/controlnet+stable_diffusion

# 2. Activate virtual environment
source venv/bin/activate

# 3. Run validation
./validate.sh

# 4. Should see: "✓ All checks passed! Project is ready to use."
```

---

## 📖 Documentation Files

| File                    | Purpose                       |
| ----------------------- | ----------------------------- |
| **USAGE_EXAMPLES.md**   | Documented run examples       |
| **README.md**           | Comprehensive user guide      |
| **SETUP.md**            | Detailed setup instructions   |
| **PROJECT_OVERVIEW.md** | Project overview and examples |
| **quickstart.sh**       | Quick reference guide         |
| **demo.sh**             | Demo script with examples     |

Read them with: `cat filename.md` or `less filename.md`

---

## 💻 System Specs

**Your System:**

- OS: Linux
- Python: 3.13.11
- PyTorch: 2.9.1+cu128
- CUDA: 12.8 (Enabled ✓)
- GPU: NVIDIA GeForce RTX 3050 Laptop (3.7 GB)
- Virtual Env: Ready ✓

**Performance:**

- ⚡ GPU-accelerated (5-10x faster than CPU)
- 🎨 High-quality image generation
- ⏱️ Fast inference (30-75 seconds per image)

---

## 🎯 Next Actions

1. **Add satellite images** to `dataset/` folder
2. **Run a test generation** with default settings
3. **Experiment with prompts** to see different results
4. **Iterate and refine** parameters for your needs
5. **Batch process** multiple images when satisfied

---

## ✨ Quick Troubleshooting

**Problem:** Not seeing GPU acceleration

```bash
# Check CUDA
python3 -c "import torch; print(torch.cuda.is_available())"
# Should print: True
```

**Problem:** Slow generation

- Using GPU? (should be by default)
- Reduce `--steps` to 20-25
- Try CPU with `--steps 15`

**Problem:** "Module not found" error

```bash
# Make sure virtual environment is active
source venv/bin/activate
```

**Problem:** Out of memory

```bash
# Reduce inference steps
--steps 15  # instead of 30
```

---

## 🎬 First Time Setup Checklist

- [ ] Read this file (you're here!)
- [ ] Add satellite images to `dataset/` folder
- [ ] Activate virtual environment: `source venv/bin/activate`
- [ ] Run a test: `python src/generate_satellite_image.py`
- [ ] Check `output/` folder for results
- [ ] Adjust parameters and try again
- [ ] Read detailed docs if needed

---

## 🚀 Ready to Generate!

Your ControlNet + Stable Diffusion project is fully set up and ready to generate amazing smart city satellite images.

**Let's get started:**

```bash
# 1. Go to project folder
cd /path/to/Stable-diffusion-with-control-net

# 2. Activate environment
source venv/bin/activate

# 3. Generate your first smart city image!
python src/generate_satellite_image.py
```

Or with explicit paths:

```bash
python src/generate_satellite_image.py \
  --image dataset/images/your_image.png \
  --mask dataset/masks/your_image.png \
  --output output/my_first_smart_city.png
```

Happy generating! 🎨✨

---

**Questions?** See [USAGE_EXAMPLES.md](USAGE_EXAMPLES.md) for run examples, or run `./demo.sh`.
