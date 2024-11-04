
# Multiple Kernel SVM

An advanced Support Vector Machine (SVM) package implementing multiple kernel types and solvers. This package includes Nyström approximation, LaSVM, NySVM, and CoreSVM solvers to support a variety of SVM applications, including online learning and efficient kernel approximations. It supports various kernel functions and allows flexible kernel chaining through a tree structure for enhanced performance.

## Table of Contents
- [Multiple Kernel SVM](#multiple-kernel-svm)
  - [Table of Contents](#table-of-contents)
  - [Features](#features)
  - [Installation](#installation)
  - [Usage](#usage)
  - [Supported Kernels](#supported-kernels)
  - [Solvers](#solvers)
  - [Development](#development)
    - [Testing and Coverage](#testing-and-coverage)
    - [Linting](#linting)
  - [Contributing](#contributing)

---

## Features

- **Multiple Kernel Options**: Support for Linear, Polynomial, RBF, Chi-squared, and other specialized kernels.
- **Kernel Chaining**: Flexible tree-based structure allowing combinations of multiple kernels.
- **Nyström Approximation**: Efficiently handles large datasets by approximating kernel matrices.
- **Multiple Solvers**: CoreSVM, NySVM, and LaSVM solvers with options for online learning.
- **Customization**: Allows custom kernel functions and multiple sampling strategies.

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Koaha/multiple_kernel.git
   cd multiple_kernel
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up the environment** (optional, recommended for development):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scriptsctivate`
   ```

## Usage

Here’s how to use the package for training and predicting with different SVM solvers.

1. **Initialize `SVMWrapper`**:
   ```python
   from svm_wrapper import SVMWrapper

   # Load toy data
   X_train, Y_train = ...  # Training data and labels
   X_test = ...            # Test data

   # Initialize SVM with CoreSVM solver
   svm = SVMWrapper(solver_type='coresvm', C=1.0, kernel='linear')
   svm.fit(X_train, Y_train)
   predictions = svm.predict(X_test)
   ```

2. **Use Nyström Approximation**:
   ```python
   # Define custom kernel function
   def rbf_kernel(x, y, gamma=0.5):
       return np.exp(-gamma * np.linalg.norm(x - y) ** 2)

   # Configure Nyström parameters
   nystrom_params = {
       'X': X_train,
       'kernel': rbf_kernel,
       'sample_size': 100,
       'sampling_method': 'kmeans'
   }

   # Initialize SVM with NySVM solver and Nyström approximation
   svm = SVMWrapper(solver_type='nysvm', use_nystrom=True, nystrom_params=nystrom_params, C=1.0)
   svm.fit(X_train, Y_train)
   predictions = svm.predict(X_test)
   ```

3. **Online Learning with LaSVM**:
   ```python
   # Initialize SVM with LaSVM solver
   lasvm = SVMWrapper(solver_type='lasvm', C=1.0, tau=0.1, kernel='rbf')
   lasvm.fit(X_train, Y_train)
   predictions = lasvm.predict(X_test)
   ```

## Supported Kernels

- **Linear Kernel**
- **Polynomial Kernel**
- **RBF (Gaussian) Kernel**
- **Chi-squared Kernel**
- **Laplacian Kernel**
- **Histogram Intersection Kernel**
- **Generalized Min Kernel**
- **Custom Kernel Support**: Define your own kernel function to use in any solver.

## Solvers

1. **CoreSVM**: Standard SVM using SMO with multiple kernel options.
2. **NySVM**: SVM with Nyström approximation for handling large datasets.
3. **LaSVM**: Online learning SVM that can dynamically update support vectors.

## Development

### Testing and Coverage

To run tests and check code coverage:

1. **Run tests**:
   ```bash
   make test
   ```

2. **Generate a coverage report**:
   ```bash
   make coverage
   ```

### Linting

Ensure code quality and style consistency by running:

```bash
make lint
```

## Contributing

Contributions are welcome! To contribute:

1. Fork the repository.
2. Create a new branch for your feature (`git checkout -b feature-name`).
3. Commit your changes (`git commit -m 'Add new feature'`).
4. Push the branch (`git push origin feature-name`).
5. Open a pull request.

---

Feel free to reach out if you encounter issues or have suggestions for improvement.
