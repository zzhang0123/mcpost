#!/usr/bin/env python3
"""
Script to validate CI/CD workflow configuration.
"""

import os
import yaml
import sys
from pathlib import Path


def validate_workflow_file(workflow_path):
    """Validate a single workflow file."""
    print(f"Validating {workflow_path}...")
    
    try:
        with open(workflow_path, 'r') as f:
            workflow = yaml.safe_load(f)
        
        # Check required fields
        required_fields = ['name', 'jobs']
        for field in required_fields:
            if field not in workflow:
                print(f"  ✗ Missing required field: {field}")
                return False
        
        # Check for 'on' field (might be parsed as True in YAML)
        if 'on' not in workflow and True not in workflow:
            print(f"  ✗ Missing required field: on")
            return False
        
        # Check that jobs exist
        if not workflow['jobs']:
            print(f"  ✗ No jobs defined")
            return False
        
        # Validate each job
        for job_name, job_config in workflow['jobs'].items():
            if 'runs-on' not in job_config:
                print(f"  ✗ Job '{job_name}' missing 'runs-on'")
                return False
            
            if 'steps' not in job_config:
                print(f"  ✗ Job '{job_name}' missing 'steps'")
                return False
        
        print(f"  ✓ Valid workflow file")
        return True
        
    except yaml.YAMLError as e:
        print(f"  ✗ YAML parsing error: {e}")
        return False
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def validate_ci_configuration():
    """Validate all CI/CD configuration."""
    print("Validating CI/CD Configuration")
    print("=" * 50)
    
    workflows_dir = Path(".github/workflows")
    
    if not workflows_dir.exists():
        print("✗ .github/workflows directory not found")
        return False
    
    workflow_files = list(workflows_dir.glob("*.yml")) + list(workflows_dir.glob("*.yaml"))
    
    if not workflow_files:
        print("✗ No workflow files found")
        return False
    
    all_valid = True
    for workflow_file in workflow_files:
        if not validate_workflow_file(workflow_file):
            all_valid = False
    
    print("\n" + "=" * 50)
    if all_valid:
        print("🎉 All CI/CD workflows are valid!")
        return True
    else:
        print("❌ Some CI/CD workflows have issues.")
        return False


def check_required_secrets():
    """Check that required secrets are documented."""
    print("\nRequired GitHub Secrets:")
    print("=" * 30)
    
    secrets = [
        ("PYPI_API_TOKEN", "PyPI API token for package publishing"),
        ("CODECOV_TOKEN", "Codecov token for coverage reporting (optional)"),
    ]
    
    print("The following secrets need to be configured in GitHub repository settings:")
    for secret_name, description in secrets:
        print(f"  - {secret_name}: {description}")
    
    return True


def check_branch_protection():
    """Check branch protection recommendations."""
    print("\nBranch Protection Recommendations:")
    print("=" * 40)
    
    recommendations = [
        "Enable 'Require status checks to pass before merging'",
        "Enable 'Require branches to be up to date before merging'",
        "Select required status checks: CI / test, CI / build, Code Quality / quality",
        "Enable 'Require pull request reviews before merging'",
        "Enable 'Dismiss stale PR approvals when new commits are pushed'",
        "Enable 'Restrict pushes that create files larger than 100 MB'",
    ]
    
    print("Configure these settings for the 'main' branch:")
    for i, rec in enumerate(recommendations, 1):
        print(f"  {i}. {rec}")
    
    return True


def main():
    """Run all CI/CD validation checks."""
    print("MCPost CI/CD Configuration Validator")
    print("=" * 60)
    
    checks = [
        ("Workflow Files", validate_ci_configuration),
        ("Required Secrets", check_required_secrets),
        ("Branch Protection", check_branch_protection),
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
        print("🎉 CI/CD configuration validation completed successfully!")
        print("\nNext steps:")
        print("1. Commit and push the workflow files")
        print("2. Configure required secrets in GitHub repository settings")
        print("3. Set up branch protection rules")
        print("4. Create a test pull request to verify workflows")
        return 0
    else:
        print("❌ CI/CD configuration validation failed.")
        print("Please fix the issues above before proceeding.")
        return 1


if __name__ == "__main__":
    sys.exit(main())