# Development Tools

This directory contains various tool scripts required for project development and maintenance.

## Available Tools

### `dev_helper.py` ⭐ **Recommended**
Main development helper script providing common development tasks:

```bash
# Run tests
python tools/dev_helper.py test

# Format code
python tools/dev_helper.py format

# Code quality checks
python tools/dev_helper.py lint

# Build package
python tools/dev_helper.py build

# Clean build artifacts
python tools/dev_helper.py clean

# Development mode installation
python tools/dev_helper.py install

# Run all quality checks
python tools/dev_helper.py check
```

### `test_runner.py`
Advanced test runner providing various test configurations and coverage reports:

```bash
# Basic test execution
python tools/test_runner.py

# Run tests and generate coverage report
python tools/test_runner.py --coverage

# Run specific test module
python tools/test_runner.py --module test_algorithms

# Verbose mode
python tools/test_runner.py --verbose

# Fast testing (skip slow tests)
python tools/test_runner.py --fast
```

### `cleanup_test_cache.py`
Test cache cleanup tool for maintaining a clean development environment:

```bash
# Clean all test cache files
python tools/cleanup_test_cache.py

# This tool cleans:
# - All files in test_cache/ directory
# - __pycache__ directories
# - .pytest_cache directory
```

### `tests/` Directory
Contains backup/fallback tests for the development tools ecosystem:

- **Purpose**: Provides basic smoke tests when main test directory is unavailable
- **Usage**: Automatically used by `test_runner.py` as a fallback mechanism
- **Content**: Simplified versions of core functionality tests
- **Note**: This is NOT the main test suite (see `/tests` in project root)

## Usage Recommendations

1. **Daily Development**: Use `dev_helper.py` for common tasks
2. **Test Debugging**: Use `test_runner.py` for advanced testing
3. **Environment Maintenance**: Regularly use `cleanup_test_cache.py` to clean cache
4. **CI/CD**: Use these tools in automation workflows

## Tool Features

- ✅ Cross-platform compatibility
- ✅ Automatic project root detection
- ✅ Clear output messages
- ✅ Support for various parameter options
- ✅ Robust error handling

## Related Documentation

For more detailed information about development tools, see: `reports/DEVELOPMENT_TOOLS.md`
