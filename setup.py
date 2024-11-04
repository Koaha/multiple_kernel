from setuptools import setup, find_packages

setup(
    name="multiple_kernel_svm",
    version="1.0.0",
    description="An advanced SVM package with multiple kernel options and solvers, supporting Nyström approximation and online learning.",
    author="vankoha",
    author_email="ldvankhoa@gmail.com",
    url="https://github.com/Koaha/multiple_kernel",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy>=1.18.0",
        "scipy>=1.4.0",
        "cvxpy>=1.1.7",
        "scikit-learn>=0.24.0",
        "tqdm>=4.41.0",
        "pandas>=1.1.0"
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "flake8>=3.8.0",
            "coverage>=5.3",
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    entry_points={
        "console_scripts": [
            "multiple_kernel_svm=src.coresvm.svm_wrapper:main",
        ]
    },
)
