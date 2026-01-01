.PHONY: clean build install test upload-test upload docs

# Clean build artifacts
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	find . -type d -name __pycache__ -exec rm -r {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.so" -delete
	find . -type f -name "*.c" -delete
	find . -type f -name "*.html" -delete

# Build the package
build: clean
	python setup.py sdist bdist_wheel

# Install in development mode
install:
	pip install -e .

# Run tests
test:
	pytest tests/ -v

# Build documentation (if using sphinx)
docs:
	@echo "Documentation building not configured yet"

# Upload to TestPyPI
upload-test: build
	twine upload --repository testpypi dist/*

# Upload to PyPI
upload: build
	twine upload dist/*

# Check package before uploading
check: build
	twine check dist/*

# Full release process
release: clean test build check upload
