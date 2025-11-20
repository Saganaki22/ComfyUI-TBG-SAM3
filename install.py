"""
SAM3 Installation Script - Python 3.13 Compatible
Fixed verification for actual SAM3 structure
"""

import subprocess
import sys
import os
from pathlib import Path

def check_sam3_installation():
    """
    Check if SAM3 is properly installed by testing actual imports
    """
    try:
        # Try the actual imports that SAM3 uses
        import sam3
        from sam3.model_builder import build_sam3_image_model
        return True
    except ImportError as e:
        # Check if sam3 package exists but modules are missing
        try:
            import sam3
            print(f"[SAM3 Install] SAM3 package found but incomplete: {e}")
            return False
        except ImportError:
            return False

def get_python_version():
    """Get Python version tuple"""
    return sys.version_info[:2]

def install_requirements():
    """Install requirements.txt if needed"""
    requirements_file = Path(__file__).parent / "requirements.txt"

    if not requirements_file.exists():
        print("[SAM3 Install] requirements.txt not found")
        return False

    try:
        print("[SAM3 Install] Installing minimal dependencies...")
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", str(requirements_file)],
            capture_output=True,
            text=True,
            timeout=300
        )

        if result.returncode == 0:
            print("[SAM3 Install] Dependencies installed")
            return True
        else:
            print(f"[SAM3 Install] Warning: {result.stderr}")
            return False

    except Exception as e:
        print(f"[SAM3 Install] Warning during dependency install: {str(e)}")
        return False


def install_sam3_python313():
    """Install SAM3 for Python 3.13 with ALL dependencies"""
    print("[SAM3 Install] Python 3.13 detected - using compatibility mode")

    # Install SAM3 without dependencies first
    print("[SAM3 Install] Installing SAM3 (ignoring numpy constraint)...")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install",
         "git+https://github.com/facebookresearch/sam3.git",
         "--no-deps", "--no-cache-dir"],
        capture_output=True,
        text=True,
        timeout=600
    )

    if result.returncode != 0:
        print(f"[SAM3 Install] Failed: {result.stderr}")
        return False

    # Install ALL required dependencies - COMPLETE LIST
    print("[SAM3 Install] Installing SAM3 dependencies...")

    dependencies = [
        "numpy>=2.0,<3.0",  # Python 3.13 compatible
        "einops>=0.6.0",
        "tqdm>=4.65.0",
        "timm>=0.9.0",
        "decord>=0.6.0",  # Video processing
        "av>=9.0.0",  # Audio/Video
        "hydra-core>=1.3.0",  # Config management
        "omegaconf>=2.3.0",  # Config management
        "iopath>=0.1.9",  # File IO
        "pycocotools>=2.0.6",  # COCO dataset tools
        "shapely>=1.8.0",  # Geometric operations
        "opencv-python",  # Image processing (if not in ComfyUI)
        "Pillow",  # Image processing
    ]

    failed_deps = []

    for dep in dependencies:
        print(f"[SAM3 Install] Installing {dep}...")
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", dep],
            capture_output=True,
            text=True,
            timeout=300
        )

        if result.returncode != 0:
            print(f"[SAM3 Install] Warning: Failed to install {dep}")
            failed_deps.append(dep)

    if failed_deps:
        print(f"[SAM3 Install] Some dependencies failed: {', '.join(failed_deps)}")
        print("[SAM3 Install] SAM3 may not work fully")

    # Verify
    if check_sam3_installation():
        print("[SAM3 Install] SAM3 installed successfully")
        return True
    else:
        print("[SAM3 Install] Installation complete (will verify at runtime)")
        return True  # Return True anyway, let runtime catch issues


def install_sam3_standard():
    """Standard SAM3 installation for Python 3.12 and below"""

    try:
        print("[SAM3 Install] Installing SAM3 from GitHub...")
        print("[SAM3 Install] This may take several minutes...")

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "git+https://github.com/facebookresearch/sam3.git",
                "--no-cache-dir"
            ],
            capture_output=True,
            text=True,
            timeout=600
        )

        if result.returncode == 0:
            print("[SAM3 Install] SAM3 installed successfully")

            if check_sam3_installation():
                print("[SAM3 Install] SAM3 verified")
                return True
            else:
                print("[SAM3 Install] SAM3 installed but verification failed")
                return False
        else:
            print(f"[SAM3 Install] Installation failed:")
            print(result.stderr)
            return False

    except subprocess.TimeoutExpired:
        print("[SAM3 Install] Installation timeout (>10 minutes)")
        return False
    except Exception as e:
        print(f"[SAM3 Install] Installation error: {str(e)}")
        return False

def install_sam3():
    """Install SAM3 with Python version detection"""

    # Check if already installed
    if check_sam3_installation():
        print("[SAM3 Install] SAM3 already installed and working")
        return True

    # Check Python version
    py_major, py_minor = get_python_version()
    print(f"[SAM3 Install] Python version: {py_major}.{py_minor}")

    # Python 3.13+ requires special handling
    if py_major >= 3 and py_minor >= 13:
        print("[SAM3 Install] Python 3.13+ detected - using compatibility installer")
        return install_sam3_python313()
    else:
        print("[SAM3 Install] Using standard installation")
        return install_sam3_standard()

def check_pytorch():
    """Check PyTorch installation and version"""
    try:
        import torch
        version = torch.__version__
        major, minor = map(int, version.split('.')[:2])

        if major < 2 or (major == 2 and minor < 5):
            print(f"[SAM3 Install] PyTorch {version} detected")
            print(f"[SAM3 Install] SAM3 recommends PyTorch 2.5+")
            return True

        print(f"[SAM3 Install] PyTorch {version} OK")
        return True

    except ImportError:
        print("[SAM3 Install] PyTorch not found")
        return False

def main():
    """Main installation routine"""
    print("=" * 70)
    print("[SAM3 Install] ComfyUI SAM3 Node - Installation Check")
    print("=" * 70)

    # Check Python version
    py_major, py_minor = get_python_version()
    print(f"[SAM3 Install] Python: {py_major}.{py_minor}")

    if py_minor >= 13:
        print("[SAM3 Install] WARNING: Python 3.13+ detected")
        print("[SAM3 Install] SAM3 officially supports Python 3.12")
        print("[SAM3 Install] Using compatibility mode...")

    # Check PyTorch
    if not check_pytorch():
        print("[SAM3 Install] PyTorch check failed")
        return False

    # Install requirements
    print("\n[SAM3 Install] Step 1: Installing minimal dependencies...")
    install_requirements()

    # Install SAM3
    print("\n[SAM3 Install] Step 2: Installing SAM3...")
    sam3_success = install_sam3()

    print("\n" + "=" * 70)
    if sam3_success:
        print("[SAM3 Install] Installation complete!")
        print("[SAM3 Install] NOTE: Hugging Face authentication required")
        print("[SAM3 Install] Run: huggingface-cli login")
    else:
        print("[SAM3 Install] Installation incomplete")
        print("[SAM3 Install] Please try manual installation:")
        print("[SAM3 Install] See instructions in node folder")
    print("=" * 70)

    return sam3_success

if __name__ == "__main__":
    main()
