# Makefile

.PHONY: install lint test coverage clean

# Install dependencies
install:
	@echo "Installing dependencies..."
	pip install -r requirements.txt

# Lint the code with flake8
lint:
	@echo "Running flake8 for linting..."
	flake8 src tests

# Run tests with coverage
test:
	@echo "Running tests with coverage..."
	pytest --cov=src --cov-report=term-missing

# Generate a coverage report
coverage:
	@echo "Generating HTML coverage report..."
	pytest --cov=src --cov-report=html

# Clean up generated files
clean:
	@echo "Cleaning up..."
	rm -rf .pytest_cache __pycache__ .coverage htmlcov
