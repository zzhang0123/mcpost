# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial package structure and modern Python packaging setup
- Global Sensitivity Analysis (GSA) module with comprehensive metrics
- Monte Carlo Integration module with QMC support
- Utilities module for data preprocessing and validation
- Type hints and comprehensive documentation
- Property-based testing framework
- Performance benchmarks and profiling utilities

### Changed
- Refactored from standalone scripts to modular package structure
- Improved API design with backward compatibility
- Enhanced error handling and input validation

### Fixed
- Resolved StandardScaler mismatch in Sobol sampling
- Fixed constant column detection and handling
- Improved numerical stability in GP fitting

## [0.1.0] - 2024-01-05

### Added
- Initial release of MCPost package
- Core GSA functionality with multiple sensitivity metrics
- Monte Carlo and Quasi-Monte Carlo integration methods
- Modern Python packaging with pyproject.toml
- Comprehensive test suite with property-based tests
- Documentation and examples
- CI/CD workflows for automated testing and release
- Distribution and release infrastructure

### Features
- **Global Sensitivity Analysis**:
  - Mutual Information and Distance Correlation metrics
  - Permutation Importance with Random Forest
  - Gaussian Process surrogates with ARD kernels
  - Sobol' indices for variance-based analysis
  - Partial Dependence Plots for interpretability
  - Chunked processing for large datasets
  
- **Monte Carlo Integration**:
  - Standard Monte Carlo with importance sampling
  - Quasi-Monte Carlo methods (Sobol, Halton)
  - Automatic integration with adaptive strategies
  - Flexible PDF specification
  - Chunked integration for memory efficiency
  
- **Package Infrastructure**:
  - Modern packaging with optional dependencies
  - Type hints and comprehensive documentation
  - Extensive testing with property-based tests
  - Performance optimizations for large datasets
  - Backward compatibility with existing code
  - Extension interfaces for custom methods
  - Automated CI/CD with GitHub Actions
  - PyPI distribution ready