# Installation Guide

This guide explains how to install and use the DLX implementation.

## Option 1: Editable Install (Recommended) ⭐

This is the best option for development - changes to the DLX code will be immediately available in your project.

### Steps:

1. **Navigate to the DLX directory:**
   ```bash
   cd /path/to/dlx
   ```

2. **Install in editable mode:**
   ```bash
   pip install -e .
   ```
   
   Or if you need to specify a virtual environment:
   ```bash
   python -m pip install -e .
   ```

3. **Use in your project:**
   ```python
   from dlxsolver import DLXSolver
   
   matrix = [[1, 0, 1], [0, 1, 0]]
   solver = DLXSolver(matrix)
   solution = solver.solve_one()
   ```

**Pros:**
- Changes to DLX code are immediately available
- Proper package installation
- Works with virtual environments

**Cons:**
- Requires the DLX directory to remain accessible

## Option 2: Add to PYTHONPATH

Add the DLX directory to your Python path.

### Steps:

1. **In your project, add to PYTHONPATH:**
   ```bash
   export PYTHONPATH="/path/to/dlx:$PYTHONPATH"
   ```

2. **Or in your Python code:**
   ```python
   import sys
   sys.path.insert(0, '/path/to/dlx')
   
   from dlxsolver import DLXSolver
   ```

**Pros:**
- Quick and simple
- No installation needed

**Cons:**
- Must set PYTHONPATH each time
- Less clean than proper installation

## Option 3: Copy Files to Your Project

Copy the necessary files directly into your project.

### Steps:

1. **Copy these files to your project:**
   - `dlx.pyx` (source)
   - `dlxsolver.py` (wrapper)
   - `setup.py` (build script)

2. **Build in your project:**
   ```bash
   python setup.py build_ext --inplace
   ```

3. **Use:**
   ```python
   from dlxsolver import DLXSolver
   ```

**Pros:**
- Self-contained
- No external dependencies

**Cons:**
- Duplicates code
- Harder to update

## Option 4: Install as Regular Package

Build and install as a regular package (not editable).

### Steps:

1. **Build the package:**
   ```bash
   cd /path/to/dlx
   python setup.py build
   python setup.py install
   ```

2. **Use in your project:**
   ```python
   from dlxsolver import DLXSolver
   ```

**Pros:**
- Proper installation
- Works system-wide or in venv

**Cons:**
- Must reinstall after changes
- Less convenient for development

## Recommended Setup

### For Local Development

```bash
# Clone or navigate to the DLX directory
cd /path/to/dlx

# Install in editable mode
pip install -e .
```

### For Use in Another Project

```bash
# From your project directory
pip install -e /path/to/dlx

# Or if published to PyPI
pip install dlx-cython
```

### Using in Your Code

```python
from dlxsolver import DLXSolver

matrix = [[1, 0, 1], [0, 1, 0], [1, 1, 0]]
solver = DLXSolver(matrix)
solution = solver.solve_one()
print(f"Solution: {solution}")
```

## Troubleshooting

### Issue: "No module named 'dlx'"

**Solution:** Make sure you've built the Cython extension:
```bash
cd /path/to/dlx
python setup.py build_ext --inplace
```

### Issue: "dlx.cpython-*.so not found"

**Solution:** The extension needs to be rebuilt. Run:
```bash
python setup.py build_ext --inplace
```

### Issue: Import works but slow

**Solution:** Make sure you're using the compiled Cython extension, not the .pyx file directly. Check that `dlx.cpython-*.so` exists in the dlx directory.

## Verifying Installation

Run this to verify everything works:

```python
from dlxsolver import DLXSolver

matrix = [[1, 0, 1], [0, 1, 0]]
solver = DLXSolver(matrix)
solution = solver.solve_one()
print(f"Installation successful! Solution: {solution}")
```

## Dependencies

Make sure these are installed:
- Python 3.7+
- NumPy
- Cython
- Setuptools

Install with:
```bash
pip install numpy cython setuptools
```
