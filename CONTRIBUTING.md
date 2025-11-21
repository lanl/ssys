# Contributing to ssys

Thank you for considering contributing to ssys! This document provides guidelines for contributing to the project.

## Code of Conduct

This project adheres to a Code of Conduct (see CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.

## How to Contribute

### Reporting Bugs

If you find a bug, please open an issue on GitHub with:
- A clear, descriptive title
- Steps to reproduce the issue
- Expected vs actual behavior
- Your environment (OS, Python version, package versions)
- Minimal example code if applicable

### Suggesting Enhancements

Enhancement suggestions are welcome! Please open an issue with:
- A clear description of the enhancement
- Motivation and use cases
- Potential implementation approach (if applicable)

### Pull Requests

1. **Fork the repository** and create a new branch from `main`
2. **Make your changes** following the coding standards below
3. **Add tests** for new functionality
4. **Update documentation** as needed (README, docstrings, etc.)
5. **Run tests** to ensure everything passes
6. **Submit a pull request** with a clear description of changes

## Development Setup

### Installation

```bash
# Clone your fork
git clone https://lisdi-git.lanl.gov/hlavacek/ssys.git
cd ssys

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in editable mode with dev dependencies
pip install -e ".[dev]"
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=ssys --cov-report=term-missing

# Run specific test file
pytest tests/test_recaster.py
```

### Code Quality

```bash
# Run linter
ruff check src/ tests/

# Run type checker (optional)
mypy src/

# Auto-format code
ruff format src/ tests/
```

## Coding Standards

- Follow PEP 8 style guidelines
- Use type hints where appropriate
- Write docstrings for all public functions/classes (Google style)
- Keep line length to 100 characters (configured in pyproject.toml)
- Write clear, descriptive variable and function names
- Add comments for complex logic

### Example Docstring

```python
def recast_to_ssystem(sym: SymSystem) -> RecastResult:
    """
    Transform symbolic ODEs into canonical S-system form.

    Args:
        sym: Symbolic ODE system with variables, parameters, and equations

    Returns:
        RecastResult containing S-system equations, auxiliaries, and factor map

    Raises:
        ValueError: If ODE contains unsupported mathematical operations
    """
    pass
```

## Testing Guidelines

- Write tests for all new functionality
- Aim for high code coverage (>80%)
- Use descriptive test names: `test_<function>_<scenario>_<expected_result>`
- Include both positive and negative test cases
- Test edge cases (empty input, zero values, negative exponents, etc.)

## Documentation

- Update README.md for user-facing changes
- Add docstrings to new functions/classes
- Update CHANGELOG.md following [Keep a Changelog](https://keepachangelog.com) format
- Include examples for new features

## Git Commit Messages

- Use present tense ("Add feature" not "Added feature")
- Use imperative mood ("Move cursor to..." not "Moves cursor to...")
- First line should be concise (<72 characters)
- Reference issues/PRs when applicable

Example:
```
Add support for piecewise functions in parser

- Implement piecewise expression detection
- Add tests for piecewise ODE models
- Update documentation

Fixes #42
```

## Release Process

(For maintainers)

1. Update version in `pyproject.toml` and `src/ssys/__init__.py`
2. Update CHANGELOG.md with release notes
3. Update CITATION.cff with new version and date
4. Create git tag: `git tag -a v0.1.0 -m "Release v0.1.0"`
5. Push tag: `git push origin v0.1.0`
6. Create GitHub release
7. Build and upload to PyPI (if applicable)

## Questions?

Feel free to open an issue for any questions about contributing!
