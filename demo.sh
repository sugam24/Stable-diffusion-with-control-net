#!/bin/bash
# Demo script showing how to use the ControlNet + Stable Diffusion project

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║         ControlNet + Stable Diffusion Demo Guide              ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

# Check if virtual environment is activated
if [[ "$VIRTUAL_ENV" == "" ]]; then
    echo "⚠️  Virtual environment not activated!"
    echo "Run: source venv/bin/activate"
    echo ""
    exit 1
fi

echo "✓ Virtual environment is active"
echo ""

# Create sample prompts file
cat > sample_prompts.txt << 'EOF'
# Sample Prompts for Smart City Generation

## Prompt 1: Modern Urban Planning
"aerial satellite view of a modern smart city with organized grid layout, solar panel rooftops, green spaces between buildings, efficient transportation corridors, autonomous vehicle lanes clearly marked, 5G tower infrastructure visible, sustainable water management systems, high quality, detailed, professional photography"

## Prompt 2: Futuristic Technology-Focused
"satellite image of an advanced smart city featuring AI surveillance camera networks, autonomous vehicle transportation hubs, smart traffic light systems at intersections, IoT sensor deployment across streets, fiber optic infrastructure, real-time traffic flow optimization visible from above, digital infrastructure visualization, high quality, futuristic"

## Prompt 3: Sustainability & Green Infrastructure
"top-down view of an eco-friendly smart city with extensive solar panel arrays on rooftops, vertical gardens and green spaces, wind turbine farms on periphery, rainwater harvesting systems integrated into design, waste management facilities hidden underground, carbon-neutral buildings with green certification, protected natural areas, clean energy transportation visible, professional high quality"

## Prompt 4: Mixed Reality Integration
"aerial view of a modern smart city with mixed reality overlay, digital information visualization on streets, smart city metrics displayed above buildings, real-time data flow visualization, sustainable architecture highlighted, underground infrastructure visible through digital transparency, autonomous systems coordinated, futuristic yet realistic, high quality"

## Prompt 5: Night-time Smart City
"satellite view of a smart city at night with intelligent street lighting systems creating efficient patterns, communication networks glowing, autonomous vehicles with lights moving through streets, building automation systems illuminated, data centers and servers visible from thermal imaging, energy management systems optimized, cybersecurity visualization, high quality night photography"
EOF

echo "📝 Sample prompts created in sample_prompts.txt"
echo ""
echo "════════════════════════════════════════════════════════════════"
echo ""
echo "USAGE EXAMPLES:"
echo ""
echo "1️⃣  Default run (uses dataset defaults + reference from reference_image/):"
echo "    python src/generate_satellite_image.py"
echo ""
echo "2️⃣  With explicit image and mask:"
echo "    python src/generate_satellite_image.py \\"
echo "      --image dataset/images/city.png \\"
echo "      --mask dataset/masks/city.png \\"
echo "      --output output/smart_city_generated.png"
echo ""
echo "3️⃣  Batch process all image-mask pairs:"
echo "    python src/generate_satellite_image.py --batch"
echo ""
echo "4️⃣  Low VRAM (4GB GPUs):"
echo "    python src/generate_satellite_image.py --low_vram"
echo ""
echo "5️⃣  With 2x upscale and Euler scheduler:"
echo "    python src/generate_satellite_image.py --upscale 2 --scheduler euler_a"
echo ""
echo "6️⃣  With custom seed for reproducibility:"
echo "    python src/generate_satellite_image.py --seed 42"
echo ""
echo "See USAGE_EXAMPLES.md for full documented examples."
echo ""
echo "════════════════════════════════════════════════════════════════"
echo ""
echo "📁 DATASET SETUP:"
echo ""
echo "1. Place images in dataset/images/ and masks in dataset/masks/:"
echo "   cp your_satellite_image.png dataset/images/"
echo "   cp your_segmentation_mask.png dataset/masks/"
echo ""
echo "2. Supported formats: JPG, PNG, BMP, TIFF"
echo ""
echo "3. Images resized to --size (768 default, 512 with --low_vram)"
echo ""
echo "════════════════════════════════════════════════════════════════"
echo ""
echo "⚙️  PARAMETER GUIDE:"
echo ""
echo "  --steps             : Inference steps (20-50)"
echo "                        20-25: Fast, lower quality"
echo "                        30: Balanced (RECOMMENDED)"
echo "                        40-50: Best quality, slower"
echo ""
echo "  --guidance_scale    : Prompt adherence (5-15)"
echo "                        7.5: Recommended"
echo "                        5-7: More creative"
echo "                        10-15: Strictly follows prompt"
echo ""
echo "  --seed              : Random seed for reproducibility"
echo "                        Use same seed for consistent results"
echo ""
echo "  --negative_prompt   : What to avoid generating"
echo "                        Default: blurry, low quality, distorted"
echo ""
echo "════════════════════════════════════════════════════════════════"
echo ""
echo "💡 TIPS FOR BEST RESULTS:"
echo ""
echo "✓ Be specific in your prompts"
echo "✓ Mention desired features (solar panels, green spaces, etc.)"
echo "✓ Include 'high quality, detailed' for better results"
echo "✓ Use CUDA GPU for 5x faster generation"
echo "✓ Experiment with guidance_scale values"
echo "✓ Use same seed to reproduce similar results"
echo "✓ 30 steps is usually the sweet spot (quality vs speed)"
echo ""
echo "════════════════════════════════════════════════════════════════"
echo ""
echo "📊 EXPECTED OUTPUTS:"
echo ""
echo "For each input, you get TWO files:"
echo "  1. generated_image.png  - Your new smart city image"
echo "  2. control_image.png    - Edge map used as control"
echo ""
echo "════════════════════════════════════════════════════════════════"
echo ""
echo "📖 For more information, see:"
echo "   - USAGE_EXAMPLES.md for full run examples"
echo "   - README.md for comprehensive docs"
echo "   - SETUP.md for setup details"
echo ""
echo "Ready to generate! 🎨✨"
echo ""
