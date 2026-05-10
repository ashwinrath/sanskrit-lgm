#!/bin/bash
# ============================================================
# Sanskrit LGM — Environment Setup
# ============================================================

set -e

echo "========================================="
echo "  Sanskrit LGM Setup"
echo "========================================="

# Detect OS
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    echo "Windows detected"
    python -m venv lgm-env
    echo "Run: lgm-env\\Scripts\\Activate"
else
    echo "Linux/macOS detected"
    python3 -m venv lgm-env
    source lgm-env/bin/activate
fi

# Detect GPU
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')" 2>/dev/null && HAS_GPU=1 || HAS_GPU=0

if [ "$HAS_GPU" = "1" ]; then
    echo "GPU detected — installing CUDA PyTorch"
    pip install torch
else
    echo "No GPU — installing CPU PyTorch"
    pip install torch --index-url https://download.pytorch.org/whl/cpu
fi

pip install numpy requests

echo ""
echo "✓ Setup complete!"
echo ""
echo "Next: python scripts/02_explore_corpus.py"
