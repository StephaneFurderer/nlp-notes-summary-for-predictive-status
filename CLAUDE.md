# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Streamlit application for insurance claims analysis using Natural Language Processing to predict claim outcomes from notes data. The system provides machine learning models to analyze claim status (PAID, DENIED, CLOSED) with NLP feature extraction, keyword analysis, and sentiment analysis capabilities.

## Key Commands

### Environment Setup
```bash
# Create and activate conda environment (recommended)
conda create -n nlp-notes python=3.9 -y
conda activate nlp-notes

# Install dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### Running Applications
```bash
# Main comprehensive NLP analysis application
streamlit run app.py

# Standardized claims visualization (micro-level reserving)
streamlit run app_2.py

# Demo version
streamlit run app_2_demo.py
```

### Data Processing Commands
```bash
# Debug data structure issues
python debug_data_structure.py

# Test structured data management examples
python example_structured_data_management.py
python example_simplified_structure.py
python example_multi_file_caching.py
```

## Architecture Overview

### Core Application Structure

**Main Applications:**
- `app.py`: Primary Streamlit interface with 7 analysis tabs (Feature Summary, Claim Analysis, Insights, ML Baseline, Pattern Analysis, Enhanced Model, NER Model)
- `app_2.py`: Standardized claims visualization following micro-level reserving framework
- `app_2_demo.py`: Demo version with simplified interface

**Helper Module Organization (`helpers/functions/`):**

**Data Processing Layer:**
- `claims_utils.py`: Core data loading, filtering, and claim feature calculation with vectorized operations and intelligent caching
- `load_cache_data.py`: Structured data management with extraction date organization
- `standardized_claims_transformer.py`: Transforms raw transaction data to standardized 30-day periods
- `standardized_claims_schema.py`: Configuration for claim data standardization

**NLP Analysis Layer:**
- `nlp_utils.py`: `ClaimNotesNLPAnalyzer` class with 8 keyword categories, sentiment analysis, and report-date-aware caching
- `notes_utils.py`: `NotesReviewerAgent` for notes processing, CSV to parquet conversion, timeline generation
- `pattern_analysis.py`: `ClaimNotesPatternAnalyzer` for discovering discriminative terms and n-grams

**Machine Learning Layer:**
- `ml_models.py`: Three model classes - `ClaimStatusBaselineModel` (Random Forest + NLP features), `ClaimStatusEnhancedModel` (+ TF-IDF), `ClaimStatusNERModel` (+ Named Entity Recognition)

**Visualization Layer:**
- `plot_utils.py`: Claim lifetime visualization and plotting utilities
- `app_2_template.py`: Reusable Streamlit interface template for claims analysis

### Data Management Architecture

**Data Directory Structure:**
```
_data/
├── [extraction_date]/           # Organized by extraction date
│   ├── clm_with_amt.csv        # Raw claims transaction data
│   ├── clm_with_amt.parquet    # Processed claims data
│   ├── notes.csv               # Raw notes data
│   └── notes.parquet           # Processed notes data
└── cache/                      # Performance optimization caches
    ├── claim_features_*.parquet     # Vectorized claim features
    └── nlp_features_*.parquet       # NLP analysis results
```

**Caching Strategy:**
- **Feature Calculation Caching**: `calculate_claim_features()` uses data hash-based caching in `_data/cache/`
- **NLP Analysis Caching**: `analyze_claims_with_caching()` combines notes data hash with report date for unique cache keys
- **Data Loading Caching**: Streamlit `@st.cache_data` decorators for UI performance

### Machine Learning Pipeline

**Three-Tier Model Architecture:**
1. **Baseline Model**: Random Forest with NLP features (keyword counts, sentiment, communication metrics)
2. **Enhanced Model**: Adds TF-IDF text vectorization for improved feature representation
3. **NER Model**: Incorporates Named Entity Recognition using spaCy for maximum accuracy

**Feature Categories (26+ features per claim):**
- Financial terms, process/investigation terms, sentiment indicators
- Urgency markers, outcome terms, legal/dispute terms
- Documentation terms, stakeholder terms
- Communication frequency, sentiment scores, temporal patterns

### Key Business Logic

**Claim Status Logic:**
- Claims grouped by `booknum`, `cidpol`, `clmNum` for unique identification
- Transaction vs Final views: All transactions over time vs latest state only
- Report date filtering: Shows claim status as of specific date with temporal logic
- Financial calculations: Benefit, Expense, Recovery, Reserve categorization with cumulative tracking

**NLP Processing Pipeline:**
1. Text preprocessing and cleaning
2. Domain-specific keyword extraction (8 categories)
3. Sentiment analysis with insurance-specific weights
4. Named entity recognition for people, organizations, locations
5. Temporal pattern analysis and communication frequency metrics

## Development Guidelines

### Performance Optimization
- Use vectorized pandas operations instead of claim-by-claim loops
- Leverage intelligent caching for expensive calculations (features, NLP analysis)
- All caching uses data hashes to detect changes and auto-invalidate
- Parquet files for optimized data storage and loading

### Data Processing Patterns
- Always check cache existence before expensive calculations
- Use `force_recalculate=True` parameter to bypass caches when needed
- Filter data by report date before analysis when specified
- Maintain transaction order with `sort_values(['clmNum', 'datetxn'])`

### Model Training Workflow
1. Load and preprocess data with temporal filtering
2. Extract NLP features with caching
3. Train models incrementally (Baseline → Enhanced → NER)
4. Compare performance across all three models
5. Generate feature importance and prediction confidence analysis

### spaCy Model Requirements
- Requires `en_core_web_sm` model for NER functionality
- Handle missing model gracefully with clear error messages
- Use spaCy for entity extraction: PERSON, ORG, GPE, MONEY, DATE

### Data Validation
- Ensure required columns exist before processing
- Handle missing notes data by generating dummy data when needed
- Validate claim number uniqueness and temporal consistency
- Check for null values in critical fields before feature calculation