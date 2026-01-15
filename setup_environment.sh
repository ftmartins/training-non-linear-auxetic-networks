#!/bin/bash
# Setup script for mech_diffrax environment
# Usage: bash setup_environment.sh

set -e  # Exit on error

echo "=========================================="
echo "Setting up mech_diffrax environment"
echo "=========================================="
echo ""

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "Error: conda not found. Please install Anaconda or Miniconda first."
    echo "Visit: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

# Check if environment already exists
if conda env list | grep -q "mech_diffrax"; then
    echo "Environment 'mech_diffrax' already exists."
    read -p "Do you want to remove and recreate it? [y/N] " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Removing existing environment..."
        conda env remove -n mech_diffrax -y
    else
        echo "Exiting without changes."
        exit 0
    fi
fi

# Create conda environment
echo ""
echo "Step 1: Creating conda environment from environment_mech_diffrax.yml..."
conda env create -f environment_mech_diffrax.yml

# Activate environment (for this script)
echo ""
echo "Step 2: Activating environment..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate mech_diffrax

# Verify key packages
echo ""
echo "Step 3: Verifying installation..."
python -c "import numpy, scipy, jax, matplotlib, networkx, pandas, joblib, tqdm; print('✓ All core packages installed successfully')"

# Check if cmocean is installed
if python -c "import cmocean" 2>/dev/null; then
    echo "✓ cmocean package available"
else
    echo "Warning: cmocean not found. Installing..."
    conda install -c conda-forge cmocean -y
fi

# Compile Cython extensions
echo ""
echo "Step 4: Compiling Cython FIRE minimizer..."
if [ -f "../instruments/setup_fire_minimizer_memview_cython.py" ]; then
    cd ../instruments/
    python setup_fire_minimizer_memview_cython.py build_ext --inplace
    cd ../ensemble_training/
    echo "✓ Cython extensions compiled successfully"
else
    echo "Warning: Cython setup file not found at ../instruments/setup_fire_minimizer_memview_cython.py"
    echo "You may need to compile it manually later."
fi

# Test configuration
echo ""
echo "Step 5: Testing configuration..."
if python config.py > /dev/null 2>&1; then
    echo "✓ Configuration validation passed"
else
    echo "Warning: Configuration validation failed. Check config.py for errors."
fi

# Print success message
echo ""
echo "=========================================="
echo "✓ Setup complete!"
echo "=========================================="
echo ""
echo "To activate the environment, run:"
echo "  conda activate mech_diffrax"
echo ""
echo "To test the installation, run:"
echo "  python task_generator.py"
echo "  python ensemble_runner.py --mode status"
echo ""
echo "To run a single training job (test):"
echo "  python ensemble_runner.py --mode single --task 0 --realization 0"
echo ""
echo "For more information, see:"
echo "  - README.md (usage guide)"
echo "  - SETUP.md (detailed setup instructions)"
echo ""
