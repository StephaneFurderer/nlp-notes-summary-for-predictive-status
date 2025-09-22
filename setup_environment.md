# Environment Setup Guide

This guide provides instructions for setting up a virtual environment for the NLP Notes Summary application.

## Option 1: Using Conda (Recommended)

```bash
# Create new conda environment
conda create -n nlp-notes python=3.9 -y

# Activate environment
conda activate nlp-notes

# Install dependencies
pip install -r requirements.txt

# Download spaCy English model
python -m spacy download en_core_web_sm
```

## Option 2: Using Python venv

```bash
# Create virtual environment
python -m venv nlp-notes-env

# Activate environment (Linux/Mac)
source nlp-notes-env/bin/activate

# Activate environment (Windows)
nlp-notes-env\Scripts\activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt

# Download spaCy English model
python -m spacy download en_core_web_sm
```

## Option 3: Using virtualenv

```bash
# Install virtualenv if not already installed
pip install virtualenv

# Create virtual environment
virtualenv nlp-notes-env

# Activate environment (Linux/Mac)
source nlp-notes-env/bin/activate

# Activate environment (Windows)
nlp-notes-env\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download spaCy English model
python -m spacy download en_core_web_sm
```

## Running the Application

After setting up the environment:

```bash
# Make sure environment is activated
# conda activate nlp-notes  # for conda
# source nlp-notes-env/bin/activate  # for venv/virtualenv

# Run the Streamlit app
streamlit run app.py
```

## Troubleshooting

### NumPy Version Conflicts
If you encounter numpy version conflicts:
```bash
pip install "numpy>=1.24.0,<2.0.0"
```

### spaCy Model Issues
If spaCy model download fails:
```bash
python -m spacy download en_core_web_sm --user
```

### Package Conflicts
If you have existing conflicting packages:
```bash
# Remove problematic packages
pip uninstall mlflow evidently tensorflow-intel skimpy numba -y

# Reinstall requirements
pip install -r requirements.txt
```

## Deactivating Environment

To deactivate the virtual environment:
```bash
# For conda
conda deactivate

# For venv/virtualenv
deactivate
```

## Environment Verification

To verify your setup is working:
```python
import pandas as pd
import numpy as np
import streamlit as st
import spacy

# Check spaCy model
nlp = spacy.load("en_core_web_sm")
print("✅ Environment setup successful!")
```