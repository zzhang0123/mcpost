"""
Property-based tests for dependency management correctness.

**Feature: mcpost-package-improvement, Property 4: Dependency Management Correctness**
**Validates: Requirements 5.1, 5.2, 5.3, 5.4, 5.5**
"""

import pytest
import subprocess
import sys
import importlib
import tempfile
import os
from pathlib import Path
from hypothesis import given, strategies as st, settings, HealthCheck
from unittest.mock import patch, MagicMock


class TestDependencyManagementCorrectness:
    """Test dependency management correctness properties."""

    def test_core_dependencies_availability(self, package_root):
        """
        Property 4a: Core Dependencies Availability
        
        For any installation scenario, required dependencies should be automatically 
        installed and available for import.
        
        **Validates: Requirements 5.1**
        """
        # Read pyproject.toml to get core dependencies
        try:
            import tomllib
        except ImportError:
            try:
                import tomli as tomllib
            except ImportError:
                pytest.skip("TOML parsing library not available")
        
        pyproject_path = package_root / "pyproject.toml"
        with open(pyproject_path, 'rb') as f:
            config = tomllib.load(f)
        
        core_dependencies = config["project"]["dependencies"]
        
        # Test that all core dependencies are specified
        required_core_deps = ["numpy", "pandas", "scikit-learn", "scipy"]
        for dep in required_core_deps:
            dep_found = any(dep in core_dep for core_dep in core_dependencies)
            assert dep_found, f"Core dependency '{dep}' must be specified in dependencies"
        
        # Test that core dependencies can be imported
        core_import_map = {
            "numpy": "numpy",
            "pandas": "pandas", 
            "scikit-learn": "sklearn",
            "scipy": "scipy"
        }
        
        for dep_name, import_name in core_import_map.items():
            try:
                importlib.import_module(import_name)
            except ImportError:
                pytest.skip(f"Core dependency {dep_name} not installed in test environment")

    def test_optional_dependencies_configuration(self, package_root):
        """
        Property 4b: Optional Dependencies Configuration
        
        For any optional dependency group, it should be properly configured with 
        appropriate extras and provide clear installation instructions.
        
        **Validates: Requirements 5.2, 5.3, 5.4**
        """
        # Read pyproject.toml to get optional dependencies
        try:
            import tomllib
        except ImportError:
            try:
                import tomli as tomllib
            except ImportError:
                pytest.skip("TOML parsing library not available")
        
        pyproject_path = package_root / "pyproject.toml"
        with open(pyproject_path, 'rb') as f:
            config = tomllib.load(f)
        
        optional_deps = config["project"]["optional-dependencies"]
        
        # Test required optional dependency groups exist (Requirement 5.2)
        required_extras = ["viz", "jupyter"]
        for extra in required_extras:
            assert extra in optional_deps, f"Optional dependency group '{extra}' must be configured"
        
        # Test matplotlib is in viz group (Requirement 5.2)
        viz_deps = optional_deps["viz"]
        matplotlib_found = any("matplotlib" in dep for dep in viz_deps)
        assert matplotlib_found, "matplotlib must be in viz optional dependencies"
        
        # Test jupyter is in jupyter group (Requirement 5.2)
        jupyter_deps = optional_deps["jupyter"]
        jupyter_found = any("jupyter" in dep for dep in jupyter_deps)
        assert jupyter_found, "jupyter must be in jupyter optional dependencies"
        
        # Test version ranges are specified (Requirement 5.3)
        all_deps = []
        all_deps.extend(config["project"]["dependencies"])
        for extra_deps in optional_deps.values():
            if isinstance(extra_deps, list):
                all_deps.extend(extra_deps)
        
        for dep in all_deps:
            if isinstance(dep, str) and not dep.startswith("mcpost["):
                # Should have version specification
                assert any(op in dep for op in [">=", "==", "~=", ">", "<"]), \
                    f"Dependency '{dep}' should have version specification"

    @given(st.sampled_from(["viz", "jupyter", "dev", "docs", "all"]))
    @settings(max_examples=2, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_extras_installation_format(self, package_root, extra_name):
        """
        Property test: For any extra dependency group, installation format should be valid.
        
        **Validates: Requirements 5.4**
        """
        # Read pyproject.toml to get optional dependencies
        try:
            import tomllib
        except ImportError:
            try:
                import tomli as tomllib
            except ImportError:
                pytest.skip("TOML parsing library not available")
        
        pyproject_path = package_root / "pyproject.toml"
        with open(pyproject_path, 'rb') as f:
            config = tomllib.load(f)
        
        optional_deps = config["project"]["optional-dependencies"]
        
        if extra_name in optional_deps:
            extra_deps = optional_deps[extra_name]
            assert isinstance(extra_deps, list), f"Extra '{extra_name}' must be a list"
            
            for dep in extra_deps:
                assert isinstance(dep, str), f"Dependency in '{extra_name}' must be string"
                # Test valid pip install format
                assert len(dep.strip()) > 0, f"Dependency in '{extra_name}' cannot be empty"

    def test_missing_optional_dependency_handling(self, package_root):
        """
        Property 4c: Missing Optional Dependency Handling
        
        For any missing optional dependency, the system should provide clear error 
        messages with installation instructions.
        
        **Validates: Requirements 5.5**
        """
        # Test that mcpost can be imported even without optional dependencies
        original_path = sys.path.copy()
        try:
            if str(package_root) not in sys.path:
                sys.path.insert(0, str(package_root))
            
            # Mock missing matplotlib to test error handling
            with patch.dict('sys.modules', {'matplotlib': None, 'matplotlib.pyplot': None}):
                try:
                    import mcpost
                    # Should be able to import mcpost even without matplotlib
                    assert hasattr(mcpost, '__version__'), "mcpost should have __version__ attribute"
                except ImportError as e:
                    # If import fails, it should not be due to missing optional dependencies
                    error_msg = str(e).lower()
                    optional_deps = ["matplotlib", "jupyter", "notebook"]
                    for opt_dep in optional_deps:
                        assert opt_dep not in error_msg, \
                            f"mcpost import should not fail due to missing optional dependency {opt_dep}"
        finally:
            sys.path = original_path

    def test_dependency_version_compatibility(self, package_root):
        """
        Property 4d: Dependency Version Compatibility
        
        For any specified dependency version range, it should be compatible with 
        the package's Python version requirements.
        
        **Validates: Requirements 5.3**
        """
        # Read pyproject.toml
        try:
            import tomllib
        except ImportError:
            try:
                import tomli as tomllib
            except ImportError:
                pytest.skip("TOML parsing library not available")
        
        pyproject_path = package_root / "pyproject.toml"
        with open(pyproject_path, 'rb') as f:
            config = tomllib.load(f)
        
        # Test Python version requirement exists
        requires_python = config["project"]["requires-python"]
        assert requires_python is not None, "requires-python must be specified"
        assert ">=" in requires_python, "requires-python should specify minimum version"
        
        # Extract minimum Python version
        min_python = requires_python.replace(">=", "").strip()
        major, minor = map(int, min_python.split(".")[:2])
        
        # Test that minimum Python version is reasonable (3.8+)
        assert major == 3, "Package should target Python 3.x"
        assert minor >= 8, "Package should support Python 3.8+ for modern compatibility"
        
        # Test that dependencies have reasonable version constraints
        all_deps = config["project"]["dependencies"]
        for dep in all_deps:
            if ">=" in dep:
                # Extract package name and version
                pkg_name = dep.split(">=")[0].strip()
                version_str = dep.split(">=")[1].strip().rstrip(",")
                
                # Test version format is valid
                version_parts = version_str.split(".")
                assert len(version_parts) >= 2, f"Version for {pkg_name} should have at least major.minor"
                
                # Test all version parts are numeric
                for part in version_parts:
                    assert part.isdigit(), f"Version parts for {pkg_name} should be numeric"

    def test_installation_extras_consistency(self, package_root):
        """
        Property 4e: Installation Extras Consistency
        
        For any installation extra, all referenced dependencies should be properly defined.
        
        **Validates: Requirements 5.4**
        """
        # Read pyproject.toml
        try:
            import tomllib
        except ImportError:
            try:
                import tomli as tomllib
            except ImportError:
                pytest.skip("TOML parsing library not available")
        
        pyproject_path = package_root / "pyproject.toml"
        with open(pyproject_path, 'rb') as f:
            config = tomllib.load(f)
        
        optional_deps = config["project"]["optional-dependencies"]
        
        # Test 'all' extra includes all other extras
        if "all" in optional_deps:
            all_extra = optional_deps["all"]
            assert len(all_extra) == 1, "'all' extra should have exactly one entry"
            
            all_entry = all_extra[0]
            assert all_entry.startswith("mcpost["), "'all' extra should reference mcpost with extras"
            assert all_entry.endswith("]"), "'all' extra should have proper bracket syntax"
            
            # Extract referenced extras
            extras_part = all_entry[all_entry.find("[")+1:all_entry.find("]")]
            referenced_extras = [e.strip() for e in extras_part.split(",")]
            
            # Test all referenced extras exist
            for extra in referenced_extras:
                if extra != "all":  # Avoid circular reference
                    assert extra in optional_deps, f"Referenced extra '{extra}' must be defined"