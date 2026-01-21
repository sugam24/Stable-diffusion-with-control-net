# ControlNet + Stable Diffusion for Smart City Satellite Image Generation

This project uses **ControlNet** + **Stable Diffusion** to generate realistic satellite images of modern smart cities. You can provide satellite images as control inputs and customize prompts to generate variations.

## 📁 Project Structure

```
controlnet+stable_diffusion/
├── src/                             # Main Python code
│   ├── generate_satellite_image.py  # Single image generation
│   └── batch_generate.py            # Batch processing script
├── dataset/                        # Put your satellite images here
├── output/                         # Generated images will be saved here
├── venv/                           # Virtual environment (will be created)
├── requirements.txt                # Python dependencies
└── README.md                       # This file
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

### Option 1: Generate from Satellite Image (Recommended)

```bash
python3 src/generate_satellite_image.py \
  --input dataset/your_satellite_image.png \
  --prompt "modern smart city with solar panels, green spaces, autonomous vehicles, 5G towers" \
  --output output/generated_smart_city.png \
  --steps 30 \
  --guidance_scale 7.5
```

### Option 2: Generate from Scratch

```bash
python3 src/generate_satellite_image.py \
  --prompt "futuristic smart city layout with AI monitoring systems, sustainable buildings, drone delivery zones" \
  --output output/generated_from_scratch.png \
  --steps 30
```

### Option 3: Batch Processing

Place multiple satellite images in the `dataset/` folder, then run:

```bash
python3 src/batch_generate.py \
  --dataset_dir dataset \
  --prompt "smart city with IoT sensors, sustainable architecture, green infrastructure" \
  --output_dir output \
  --steps 25
```

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

- `--input`: Path to input satellite image (optional)
- `--prompt`: Text prompt for generation (required)
- `--negative_prompt`: What to avoid in generation (default: "blurry, low quality, distorted")
- `--output`: Path to save generated image (default: "output/generated_image.png")
- `--steps`: Number of inference steps, 20-50 (default: 30)
- `--guidance_scale`: How much to follow the prompt, 7-15 (default: 7.5)
- `--seed`: Random seed for reproducibility (optional)

### src/batch_generate.py

Same arguments as above, with:

- `--dataset_dir`: Directory containing satellite images (default: "dataset")

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
