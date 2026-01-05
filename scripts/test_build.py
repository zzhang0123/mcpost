#!/usr/bin/env python3
"""
Script to test package building and installation for PyPI distribution.
"""

import subprocess
import sys
import tempfile
import shutil
import os
from pathlib import Path


def run_command(cmd, cwd=None, check=True):
    """Run a command and return the result."""
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    
    if result.stdout:
        print("STDOUT:", result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    
    if check and result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode, cmd)
    
    return result


def test_package_build():
    """Test building the package."""
    print("=" * 60)
    print("Testing package build...")
    print("=" * 60)
    
    # Clean previous builds
    for dir_name in ["build", "dist", "*.egg-info"]:
        if os.path.exists(dir_name):
            shutil.rmtree(dir_name, ignore_errors=True)
    
    # Build the package
    try:
        run_command([sys.executable, "-m", "build"])
        print("✓ Package build successful")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Package build failed: {e}")
        return False


def test_package_installation():
    """Test installing the built package."""
    print("=" * 60)
    print("Testing package installation...")
    print("=" * 60)
    
    # Find the built wheel
    dist_dir = Path("dist")
    if not dist_dir.exists():
        print("✗ No dist directory found. Build the package first.")
        return False
    
    wheel_files = list(dist_dir.glob("*.whl"))
    if not wheel_files:
        print("✗ No wheel file found in dist directory.")
        return False
    
    wheel_file = wheel_files[0]
    print(f"Found wheel: {wheel_file}")
    
    # Create a temporary virtual environment
    with tempfile.TemporaryDirectory() as temp_dir:
        venv_dir = Path(temp_dir) / "test_venv"
        
        try:
            # Create virtual environment
            run_command([sys.executable, "-m", "venv", str(venv_dir)])
            
            # Get python executable in venv
            if sys.platform == "win32":
                python_exe = venv_dir / "Scripts" / "python.exe"
                pip_exe = venv_dir / "Scripts" / "pip.exe"
            else:
                python_exe = venv_dir / "bin" / "python"
                pip_exe = venv_dir / "bin" / "pip"
            
            # Upgrade pip
            run_command([str(pip_exe), "install", "--upgrade", "pip"])
            
            # Install the package
            run_command([str(pip_exe), "install", str(wheel_file.absolute())])
            
            # Test importing the package
            test_script = """
import mcpost
print(f"MCPost version: {mcpost.__version__}")

# Test basic imports
from mcpost import gsa_pipeline, monte_carlo_integral
from mcpost.utils import GSAConfig, IntegrationConfig

print("✓ All imports successful")

# Test version info
from mcpost._version import get_version, get_version_info
print(f"Version: {get_version()}")
print(f"Version info: {get_version_info()}")
"""
            
            result = run_command([str(python_exe), "-c", test_script])
            print("✓ Package installation and import test successful")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"✗ Package installation test failed: {e}")
            return False


def test_version_consistency():
    """Test that version is consistent across files."""
    print("=" * 60)
    print("Testing version consistency...")
    print("=" * 60)
    
    try:
        # Import version from package
        sys.path.insert(0, ".")
        from mcpost._version import __version__, get_version, get_version_info
        
        # Check version format (semantic versioning)
        version_parts = __version__.split(".")
        if len(version_parts) != 3:
            print(f"✗ Version format invalid: {__version__} (should be X.Y.Z)")
            return False
        
        # Check that all parts are integers
        try:
            [int(part) for part in version_parts]
        except ValueError:
            print(f"✗ Version parts should be integers: {__version__}")
            return False
        
        # Check version functions
        if get_version() != __version__:
            print(f"✗ get_version() mismatch: {get_version()} != {__version__}")
            return False
        
        expected_info = tuple(map(int, version_parts))
        if get_version_info() != expected_info:
            print(f"✗ get_version_info() mismatch: {get_version_info()} != {expected_info}")
            return False
        
        print(f"✓ Version consistency check passed: {__version__}")
        return True
        
    except Exception as e:
        print(f"✗ Version consistency check failed: {e}")
        return False


def main():
    """Run all build and installation tests."""
    print("MCPost Package Build and Installation Test")
    print("=" * 60)
    
    tests = [
        ("Version Consistency", test_version_consistency),
        ("Package Build", test_package_build),
        ("Package Installation", test_package_installation),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"✗ {test_name} failed with exception: {e}")
            results.append((test_name, False))
        print()
    
    # Summary
    print("=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    all_passed = True
    for test_name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{test_name}: {status}")
        if not success:
            all_passed = False
    
    print("=" * 60)
    if all_passed:
        print("🎉 All tests passed! Package is ready for PyPI distribution.")
        return 0
    else:
        print("❌ Some tests failed. Please fix issues before distribution.")
        return 1


if __name__ == "__main__":
    sys.exit(main())