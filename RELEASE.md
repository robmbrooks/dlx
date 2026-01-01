# Release Process for PyPI

This document outlines the steps to release a new version of DLX-Cython to PyPI.

## Prerequisites

1. **PyPI Account**: Create an account at https://pypi.org/account/register/
2. **API Token**: Generate an API token at https://pypi.org/manage/account/token/
3. **Install Tools**:
   ```bash
   pip install build twine
   ```

## Pre-Release Checklist

- [ ] Update version number in `setup.py`
- [ ] Update `CHANGELOG.md` with new version and changes
- [ ] Update `README.md` if needed
- [ ] Run tests: `pytest tests/`
- [ ] Verify examples work: `python example_usage.py`
- [ ] Check that all documentation is up to date
- [ ] Ensure LICENSE file is correct

## Release Steps

### 1. Update Version

Edit `setup.py`:
```python
version='0.1.0',  # Update this
```

Update `CHANGELOG.md`:
```markdown
## [0.1.0] - 2024-XX-XX
```

### 2. Clean Previous Builds

```bash
make clean
# or manually:
rm -rf build/ dist/ *.egg-info
```

### 3. Build Distribution Packages

```bash
python -m build
```

This creates:
- `dist/dlx_cython-0.1.0.tar.gz` (source distribution)
- `dist/dlx_cython-0.1.0-py3-none-any.whl` (wheel)

### 4. Check Package

```bash
twine check dist/*
```

### 5. Test Upload to TestPyPI (Recommended)

First, configure TestPyPI credentials in `~/.pypirc` or use environment variables:

```bash
export TWINE_USERNAME=__token__
export TWINE_PASSWORD=pypi-your-test-token-here
```

Upload to TestPyPI:
```bash
twine upload --repository testpypi dist/*
```

Test installation:
```bash
pip install --index-url https://test.pypi.org/simple/ dlx-cython
```

### 6. Upload to PyPI

Once tested, upload to production PyPI:

```bash
export TWINE_USERNAME=__token__
export TWINE_PASSWORD=pypi-your-production-token-here

twine upload dist/*
```

Or use the Makefile:
```bash
make upload
```

### 7. Verify Installation

```bash
pip install dlx-cython
python -c "from dlxsolver import DLXSolver; print('Success!')"
```

### 8. Create Git Tag

```bash
git tag -a v0.1.0 -m "Release version 0.1.0"
git push origin v0.1.0
```

## Using Makefile (Simplified)

```bash
# Clean, build, test, and upload
make release

# Or step by step:
make clean
make build
make test
make check
make upload
```

## Troubleshooting

### "File already exists" error

The version already exists on PyPI. Update the version number.

### "Invalid distribution" error

Run `twine check dist/*` to see specific issues.

### Authentication errors

Make sure your API token is correct and has the right permissions.

### Build errors

Ensure all dependencies are installed:
```bash
pip install build twine cython numpy setuptools wheel
```

## Automated Release (GitHub Actions)

See `.github/workflows/publish.yml` for automated release on git tags.

## Version Numbering

Follow [Semantic Versioning](https://semver.org/):
- **MAJOR**: Breaking changes
- **MINOR**: New features, backwards compatible
- **PATCH**: Bug fixes, backwards compatible

Example: `0.1.0` → `0.1.1` (patch) → `0.2.0` (minor) → `1.0.0` (major)
