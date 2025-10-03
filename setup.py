from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="claims-reserving-toolkit",
    version="0.1.0",
    author="SF",
    description="LSTM toolkit for insurance claims reserving and ultimate loss prediction",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(include=['claims_reserving_toolkit', 'claims_reserving_toolkit.*']),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Financial and Insurance Industry",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0",
            "jupyter>=1.0",
            "matplotlib>=3.5",
            "pandas>=2.0",
        ]
    },
)

