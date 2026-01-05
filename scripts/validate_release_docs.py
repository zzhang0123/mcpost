#!/usr/bin/env python3
"""
Script to validate release documentation completeness and consistency.
"""

import os
import re
import sys
from pathlib import Path


def check_changelog_format():
    """Check CHANGELOG.md format and completeness."""
    print("Checking CHANGELOG.md format...")
    
    changelog_path = Path("CHANGELOG.md")
    if not changelog_path.exists():
        print("  ✗ CHANGELOG.md not found")
        return False
    
    with open(changelog_path, 'r') as f:
        content = f.read()
    
    # Check for required sections
    required_patterns = [
        r"# Changelog",
        r"## \[Unreleased\]",
        r"## \[\d+\.\d+\.\d+\] - \d{4}-\d{2}-\d{2}",
        r"### Added",
        r"### Changed",
        r"### Fixed"
    ]
    
    for pattern in required_patterns:
        if not re.search(pattern, content):
            print(f"  ✗ Missing pattern: {pattern}")
            return False
    
    # Check version format consistency
    version_matches = re.findall(r"\[(\d+\.\d+\.\d+)\]", content)
    if not version_matches:
        print("  ✗ No version numbers found")
        return False
    
    print(f"  ✓ Found versions: {', '.join(version_matches)}")
    print("  ✓ CHANGELOG.md format is valid")
    return True


def check_version_consistency():
    """Check version consistency across files."""
    print("Checking version consistency...")
    
    # Read version from _version.py
    version_file = Path("mcpost/_version.py")
    if not version_file.exists():
        print("  ✗ mcpost/_version.py not found")
        return False
    
    with open(version_file, 'r') as f:
        version_content = f.read()
    
    version_match = re.search(r'__version__ = ["\']([^"\']+)["\']', version_content)
    if not version_match:
        print("  ✗ Version not found in _version.py")
        return False
    
    package_version = version_match.group(1)
    print(f"  Package version: {package_version}")
    
    # Check CHANGELOG.md has this version
    changelog_path = Path("CHANGELOG.md")
    with open(changelog_path, 'r') as f:
        changelog_content = f.read()
    
    if f"[{package_version}]" not in changelog_content:
        print(f"  ✗ Version {package_version} not found in CHANGELOG.md")
        return False
    
    print("  ✓ Version consistency check passed")
    return True


def check_documentation_files():
    """Check that all required documentation files exist."""
    print("Checking documentation files...")
    
    required_docs = [
        "README.md",
        "CHANGELOG.md",
        "LICENSE"
    ]
    
    missing_files = []
    for doc_file in required_docs:
        if not Path(doc_file).exists():
            missing_files.append(doc_file)
    
    if missing_files:
        print(f"  ✗ Missing documentation files: {', '.join(missing_files)}")
        return False
    
    print("  ✓ All required documentation files present")
    return True


def validate_links():
    """Validate internal links in documentation."""
    print("Validating documentation links...")
    
    # This is a simplified check - in practice you'd want more sophisticated link checking
    doc_files = [
        "README.md"
    ]
    
    broken_links = []
    for doc_file in doc_files:
        if not Path(doc_file).exists():
            continue
            
        with open(doc_file, 'r') as f:
            content = f.read()
        
        # Check for relative links to files that should exist
        relative_links = re.findall(r'\[.*?\]\(([^http][^)]+)\)', content)
        for link in relative_links:
            # Remove anchors
            file_path = link.split('#')[0]
            if file_path and not Path(file_path).exists():
                broken_links.append(f"{doc_file}: {link}")
    
    if broken_links:
        print(f"  ⚠️  Potential broken links found:")
        for link in broken_links:
            print(f"    - {link}")
        # Don't fail for this - just warn
    
    print("  ✓ Link validation completed")
    return True


def main():
    """Run all release documentation validation checks."""
    print("MCPost Release Documentation Validator")
    print("=" * 60)
    
    checks = [
        ("CHANGELOG Format", check_changelog_format),
        ("Version Consistency", check_version_consistency),
        ("Documentation Files", check_documentation_files),
        ("Link Validation", validate_links),
    ]
    
    all_passed = True
    for check_name, check_func in checks:
        try:
            success = check_func()
            if not success:
                all_passed = False
        except Exception as e:
            print(f"✗ {check_name} check failed: {e}")
            all_passed = False
        print()
    
    print("=" * 60)
    if all_passed:
        print("🎉 All release documentation checks passed!")
        print("\nRelease documentation is ready. Next steps:")
        print("1. Review all documentation for accuracy")
        print("2. Update version numbers when ready to release")
        return 0
    else:
        print("❌ Some release documentation checks failed.")
        print("Please fix the issues above before proceeding with release.")
        return 1


if __name__ == "__main__":
    sys.exit(main())