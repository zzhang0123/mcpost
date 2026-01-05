# Contributing to MCPost

Thank you for your interest in contributing to MCPost! This document provides guidelines for contributing to the project.

## Development Setup

1. Clone the repository:
```bash
git clone https://github.com/zzhang0123/mcpost.git
cd mcpost
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install in development mode:
```bash
pip install -e ".[dev,viz]"
```

4. Run tests to ensure everything works:
```bash
pytest tests/
```

## Development Workflow

1. Fork the repository on GitHub
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes
4. Add tests for new functionality
5. Run the test suite: `pytest tests/`
6. Run code quality checks: `python scripts/validate_ci.py`
7. Commit your changes: `git commit -m "Description of changes"`
8. Push to your fork: `git push origin feature-name`
9. Create a Pull Request on GitHub

## Code Standards

- Follow PEP 8 style guidelines
- Add docstrings to all public functions (NumPy style)
- Include type hints where appropriate
- Write tests for new functionality
- Maintain backward compatibility

## Testing

- Unit tests: Test individual functions and classes
- Property-based tests: Test universal properties with Hypothesis
- Integration tests: Test complete workflows
- Backward compatibility tests: Ensure API compatibility

Run specific test categories:
```bash
# Unit tests only
pytest tests/test_gsa/ tests/test_integration/ tests/test_utils/

# Property-based tests
pytest tests/ -k "property"

# Integration tests
pytest tests/integration/
```

## Documentation

- Update docstrings for any API changes
- Add examples for new features
- Update README.md if needed
- Add entries to CHANGELOG.md for significant changes

## Release Process

See [RELEASE_GUIDE.md](docs/RELEASE_GUIDE.md) for detailed release procedures.

## Questions?

Feel free to open an issue for questions or discussions about contributing.