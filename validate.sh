#!/bin/bash
# Project validation script - Ensures everything is set up correctly

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║          ControlNet Project Validation & Health Check          ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

# Color codes
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

passed=0
failed=0

# Function to check if a command exists
check_command() {
    if command -v $1 &> /dev/null; then
        echo -e "${GREEN}✓${NC} $1 is installed"
        ((passed++))
    else
        echo -e "${RED}✗${NC} $1 is NOT installed"
        ((failed++))
    fi
}

# Function to check if Python module exists
check_module() {
    python3 -c "import $1" 2>/dev/null
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓${NC} Python module '$1' is available"
        ((passed++))
    else
        echo -e "${RED}✗${NC} Python module '$1' is NOT available"
        ((failed++))
    fi
}

# Function to check if file/directory exists
check_file() {
    if [ -e "$1" ]; then
        echo -e "${GREEN}✓${NC} $1 exists"
        ((passed++))
    else
        echo -e "${RED}✗${NC} $1 does NOT exist"
        ((failed++))
    fi
}

# Activate virtual environment if not already
if [[ "$VIRTUAL_ENV" == "" ]]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

echo "═══════════════════════════════════════════════════════════════"
echo "🔍 System Information"
echo "═══════════════════════════════════════════════════════════════"

check_command python3
python3 --version

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "📦 Python Packages"
echo "═══════════════════════════════════════════════════════════════"

check_module torch
check_module torchvision
check_module diffusers
check_module transformers
check_module controlnet_aux
check_module cv2
check_module PIL
check_module numpy
check_module matplotlib

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "💾 Project Files"
echo "═══════════════════════════════════════════════════════════════"

check_file "src/generate_satellite_image.py"
check_file "src/batch_generate.py"
check_file "src"
check_file "requirements.txt"
check_file "README.md"
check_file "SETUP.md"
check_file "quickstart.sh"
check_file "dataset"
check_file "output"
check_file "venv"

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "⚙️  GPU & CUDA Support"
echo "═══════════════════════════════════════════════════════════════"

python3 << 'PYTHON_CODE'
import torch
import sys

print(f"PyTorch Version: {torch.__version__}")

if torch.cuda.is_available():
    print(f"✓ CUDA Available: YES")
    print(f"  - CUDA Version: {torch.version.cuda}")
    print(f"  - GPU Count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  - GPU {i}: {torch.cuda.get_device_name(i)}")
    
    # Check memory
    try:
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"  - GPU Memory: ~{gpu_mem:.1f} GB")
    except:
        pass
else:
    print(f"⚠  CUDA Available: NO")
    print(f"   CPU mode will be used (slower)")
PYTHON_CODE

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "🧪 Quick Functionality Test"
echo "═══════════════════════════════════════════════════════════════"

# Try to import main modules
python3 << 'PYTHON_CODE'
import sys
sys.path.insert(0, '.')

try:
    from src.generate_satellite_image import SatelliteImageGenerator
    print("✓ SatelliteImageGenerator class loads successfully")
except Exception as e:
    print(f"✗ Error loading SatelliteImageGenerator: {e}")
    sys.exit(1)

print("✓ All critical imports working")
PYTHON_CODE

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "📊 Validation Summary"
echo "═══════════════════════════════════════════════════════════════"
echo -e "✓ Passed: ${GREEN}${passed}${NC}"
echo -e "✗ Failed: ${RED}${failed}${NC}"
echo ""

if [ $failed -eq 0 ]; then
    echo -e "${GREEN}✓ All checks passed! Project is ready to use.${NC}"
    echo ""
    echo "Next steps:"
    echo "1. Add satellite images to 'dataset/' folder"
    echo "2. Run: python3 src/generate_satellite_image.py --input dataset/image.png --prompt 'smart city'"
    echo "3. Check output in 'output/' folder"
    echo ""
else
    echo -e "${RED}✗ Some checks failed. Please fix the issues above.${NC}"
    echo ""
    echo "Common fixes:"
    echo "- Make sure virtual environment is activated: source venv/bin/activate"
    echo "- Reinstall packages: pip install -r requirements.txt"
    echo "- Check disk space: df -h"
    echo "- Check file permissions: chmod -R 755 ."
    echo ""
fi

echo "═══════════════════════════════════════════════════════════════"
