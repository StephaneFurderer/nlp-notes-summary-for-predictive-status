# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Streamlit application for analyzing insurance claims data with notes review functionality. The application provides predictive status analysis by processing claims transactions and associated notes data to help identify patterns and insights for claims management.

## Key Commands

### Running the Application
```bash
streamlit run app.py
```

### Data Processing
The application automatically loads and processes data from the `_data/` directory:
- Claims transaction data: `_data/clm_with_amt.csv`
- Notes data: `_data/notes.csv`
- Processed data is cached as parquet files for performance

## Architecture

### Core Components

**Main Application (`app.py`)**
- Streamlit interface for claims analysis
- Date-based filtering for report generation
- Interactive claim selection and filtering by cause
- Notes timeline visualization for individual claims

**Data Processing (`helpers/functions/data_utils.py`)**
- `load_data()`: Main data loading function that returns 8 dataframes for different views
- `import_data()`: Processes raw CSV data and creates optimized parquet files
- `aggregate_by_booking_policy_claim()`: Core business logic for claim amount calculations
- `calculate_claim_amounts()`: Handles payment categorization (Benefit, Expense, Recovery, Reserve)

**Notes Analysis (`helpers/functions/notes_utils.py`)**
- `NotesReviewerAgent`: Main class for notes processing and analysis
- Automatic CSV to parquet conversion for performance
- Timeline generation and note frequency analysis
- Search functionality across notes content

### Data Model

**Claims Data Structure:**
- Claims are grouped by `booknum`, `cidpol`, `clmNum`
- Status tracking: OPEN, CLOSED, PAID, DENIED, etc.
- Financial calculations: paid, expense, recovery, reserve, incurred amounts
- Temporal tracking: `dateCompleted`, `dateReopened`, `datetxn`

**Transaction vs Final Views:**
- Transaction view (`transaction_view=True`): Shows all transactions over time with cumulative calculations
- Final view: Shows only the latest state for each claim

**Report Date Logic:**
- When a report date is specified, the system filters data to show claim status as of that date
- Claims not closed by report date are treated as open
- Claims closed and not reopened by report date remain closed

### Data Directory Structure
```
_data/
├── clm_with_amt.csv        # Raw claims transaction data
├── clm_with_amt.parquet    # Processed claims data (auto-generated)
├── notes.csv               # Raw notes data
├── notes.parquet           # Processed notes data (auto-generated)
└── cache/                  # Feature calculation cache directory
    └── claim_features_*.parquet  # Cached claim feature calculations
```

## Performance Optimizations

### Feature Calculation Caching
The `calculate_claim_features` function includes intelligent caching:
- Generates data hash to detect changes in input data
- Caches results as parquet files in `_data/cache/`
- Automatically loads from cache when data hasn't changed
- Use `force_recalculate=True` to bypass cache when needed

### NLP Analysis Caching
The `analyze_claims_with_caching` function provides report-date-aware caching:
- Combines notes data hash with report date for unique cache keys
- Creates separate cache files for different report dates
- Filters notes by report date before analysis when specified
- Stores results as `nlp_features_[hash].parquet` files
- Instant loading for previously analyzed date combinations

### Vectorized Operations
- Replaced claim-by-claim loops with pandas groupby operations
- Uses vectorized calculations for all metrics (26+ features per claim)
- Significant performance improvement for large datasets
- Maintains identical results to original implementation

## NLP Research: Predicting Claim Status from Notes

### Project Structure for NLP Analysis

The goal is to extract key information from claim notes to predict claim status (PAID/CLOSED/DENIED) using classification models.

### Text Feature Extraction Approaches

#### 1. Keyword/Phrase Analysis
- **Domain-specific terms**: "investigation", "fraud", "medical records", "liability", "settlement"
- **Sentiment indicators**: "disputed", "agreed", "denied", "approved", "pending"
- **Urgency markers**: "urgent", "immediate", "follow-up required", "deadline"
- **Financial terms**: "estimate", "quote", "damage assessment", "coverage limit"

#### 2. Named Entity Recognition (NER)
- **People**: Adjuster names, claimant names, witness names
- **Organizations**: Repair shops, medical facilities, legal firms
- **Locations**: Accident locations, addresses
- **Dates**: Important deadlines, incident dates, follow-up dates
- **Monetary amounts**: Claim values, deductibles, estimates

#### 3. Temporal Pattern Analysis
- **Note frequency**: Claims with many notes vs. few notes
- **Time gaps**: Long periods without updates might indicate stalled claims
- **Note timing**: Notes added close to deadlines
- **Communication patterns**: Regular vs. sporadic updates

### Advanced NLP Techniques

#### 4. Sentiment & Tone Analysis
- **Conflict indicators**: "disagreement", "dispute", "challenge"
- **Cooperation signals**: "agreed", "collaborative", "responsive"
- **Frustration markers**: "delays", "unresponsive", "complications"

#### 5. Topic Modeling
- **Claim categories**: Medical, property damage, liability, theft
- **Process stages**: Investigation, negotiation, documentation, closure
- **Issue types**: Coverage disputes, fraud concerns, documentation problems

#### 6. Sequential Patterns
- **Note progression**: How language changes over time
- **Status transitions**: Notes preceding status changes
- **Decision indicators**: Words that appear before denials/approvals

### Feature Engineering Strategy

#### 7. Aggregated Metrics
- **Note complexity**: Average words per note, vocabulary diversity
- **Communication frequency**: Notes per day/week
- **Stakeholder involvement**: Number of different people mentioned
- **Process indicators**: Mentions of specific procedures or requirements

#### 8. Risk Indicators
- **Red flags**: "legal", "attorney", "lawsuit", "subpoena"
- **Positive signals**: "straightforward", "clear", "documented"
- **Uncertainty markers**: "unclear", "investigating", "pending"

### Model Architecture Considerations

#### 9. Feature Combinations
- **Text + Financial**: Combine note features with transaction amounts
- **Text + Temporal**: Note sentiment + time since first transaction
- **Text + Claim attributes**: Note content + claim cause/type

#### 10. Prediction Targets
- **Binary**: Will claim be denied (Y/N)?
- **Multi-class**: Predict final status (PAID/CLOSED/DENIED)
- **Probability**: Likelihood of each outcome
- **Time-to-event**: Predict when status will change

### Implementation Plan
1. Start with keyword extraction and sentiment analysis (interpretable baseline)
2. Build domain-specific keyword dictionaries for insurance claims
3. Implement sentiment scoring for notes
4. Combine text features with existing claim financial features
5. Train classification models to predict claim outcomes

## Development Notes

- The application uses pandas extensively for data processing
- Parquet files are automatically generated for performance optimization
- The `NotesReviewerAgent` can generate dummy data if notes.csv doesn't exist
- Claims status logic includes complex business rules for determining open/closed/paid states
- Feature calculations are cached and vectorized for optimal performance
- The UI provides both portfolio-level and individual claim-level analysis