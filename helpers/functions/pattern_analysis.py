import pandas as pd
import numpy as np
from collections import Counter, defaultdict
import re
from typing import Dict, List, Tuple, Set
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.feature_selection import chi2
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data (run once)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

class ClaimNotesPatternAnalyzer:
    """
    Phase 1: Analyze note patterns and word frequencies by claim status
    to discover data-driven insights for enhanced keyword categories.
    """

    def __init__(self):
        """Initialize the pattern analyzer"""
        # Insurance-specific stop words to add to standard ones
        self.insurance_stopwords = {
            'claim', 'claims', 'claimant', 'policy', 'policyholder', 'insured',
            'insurance', 'company', 'adjuster', 'agent', 'representative',
            'will', 'would', 'could', 'should', 'may', 'might', 'can',
            'please', 'thank', 'thanks', 'regards', 'sincerely',
            'contact', 'call', 'email', 'phone', 'number'
        }

        # Get standard English stopwords and add insurance-specific ones
        try:
            self.stop_words = set(stopwords.words('english'))
        except:
            self.stop_words = set()
        self.stop_words.update(self.insurance_stopwords)

    def preprocess_text(self, text: str) -> str:
        """
        Clean and preprocess text for analysis

        Args:
            text: Raw note text

        Returns:
            Cleaned text
        """
        if pd.isna(text) or not isinstance(text, str):
            return ""

        # Convert to lowercase
        text = text.lower()

        # Remove special characters but keep spaces and basic punctuation
        text = re.sub(r'[^\w\s\.\,\!\?]', ' ', text)

        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        # Remove very short words (less than 2 characters)
        words = text.split()
        words = [word for word in words if len(word) >= 2]

        return ' '.join(words)

    def extract_word_frequencies(self, notes_df: pd.DataFrame, claims_df: pd.DataFrame) -> Dict:
        """
        Extract word frequencies by claim status

        Args:
            notes_df: DataFrame with notes
            claims_df: DataFrame with claim status information

        Returns:
            Dictionary with frequency analysis by status
        """
        # Merge notes with claim status
        merged_df = notes_df.merge(
            claims_df[['clmNum', 'clmStatus', 'dateReopened']],
            on='clmNum',
            how='inner'
        )

        # Create granular status (same logic as model)
        def create_granular_status(row):
            base_status = row['clmStatus']
            if base_status in ['PAID', 'DENIED', 'CLOSED']:
                is_reopened = pd.notna(row['dateReopened'])
                if is_reopened:
                    return f"{base_status}_REOPENED"
                else:
                    return base_status
            else:
                return base_status

        merged_df['granular_status'] = merged_df.apply(create_granular_status, axis=1)

        # Preprocess all notes
        print("Preprocessing notes...")
        merged_df['cleaned_note'] = merged_df['note'].apply(self.preprocess_text)

        # Aggregate notes by claim and status
        claim_notes = merged_df.groupby(['clmNum', 'granular_status'])['cleaned_note'].apply(' '.join).reset_index()

        # Group by status
        status_groups = {}
        for status in claim_notes['granular_status'].unique():
            status_notes = claim_notes[claim_notes['granular_status'] == status]['cleaned_note'].tolist()
            all_text = ' '.join(status_notes)

            # Tokenize and remove stopwords
            words = word_tokenize(all_text)
            words = [word for word in words if word.lower() not in self.stop_words and len(word) >= 3]

            status_groups[status] = {
                'text': all_text,
                'words': words,
                'word_freq': Counter(words),
                'num_claims': len(status_notes)
            }

        print(f"Analyzed {len(claim_notes)} claim-status combinations")
        print(f"Status breakdown: {claim_notes['granular_status'].value_counts().to_dict()}")

        return status_groups

    def extract_ngram_patterns(self, status_groups: Dict, n_range: Tuple[int, int] = (2, 3)) -> Dict:
        """
        Extract n-gram patterns for each status

        Args:
            status_groups: Output from extract_word_frequencies
            n_range: Range of n-grams to extract (min, max)

        Returns:
            Dictionary with n-gram patterns by status
        """
        ngram_patterns = {}

        for status, data in status_groups.items():
            ngram_patterns[status] = {}

            for n in range(n_range[0], n_range[1] + 1):
                # Extract n-grams
                tokens = data['words']
                if len(tokens) >= n:
                    ngrams_list = list(ngrams(tokens, n))
                    ngram_freq = Counter(ngrams_list)

                    # Convert tuples to strings and filter by frequency
                    ngram_strings = {' '.join(gram): freq for gram, freq in ngram_freq.items() if freq >= 2}
                    ngram_patterns[status][f'{n}gram'] = ngram_strings

        return ngram_patterns

    def calculate_tfidf_features(self, status_groups: Dict, max_features: int = 100) -> Dict:
        """
        Calculate TF-IDF features to find discriminative terms

        Args:
            status_groups: Output from extract_word_frequencies
            max_features: Maximum number of features to extract

        Returns:
            Dictionary with TF-IDF analysis
        """
        # Prepare documents (one per status)
        documents = []
        status_labels = []

        for status, data in status_groups.items():
            documents.append(data['text'])
            status_labels.append(status)

        if len(documents) < 2:
            return {"error": "Need at least 2 status groups for TF-IDF analysis"}

        # TF-IDF Vectorization
        vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words=list(self.stop_words),
            ngram_range=(1, 2),  # Include unigrams and bigrams
            min_df=2,  # Must appear in at least 2 documents
            max_df=0.8  # Must not appear in more than 80% of documents
        )

        try:
            tfidf_matrix = vectorizer.fit_transform(documents)
            feature_names = vectorizer.get_feature_names_out()

            # Get TF-IDF scores for each status
            tfidf_scores = {}
            for i, status in enumerate(status_labels):
                scores = tfidf_matrix[i].toarray()[0]
                status_scores = dict(zip(feature_names, scores))
                # Sort by score and keep top terms
                sorted_scores = sorted(status_scores.items(), key=lambda x: x[1], reverse=True)
                tfidf_scores[status] = sorted_scores[:50]  # Top 50 terms per status

            return {
                'tfidf_scores': tfidf_scores,
                'feature_names': feature_names,
                'vectorizer': vectorizer
            }

        except Exception as e:
            return {"error": f"TF-IDF calculation failed: {str(e)}"}

    def find_discriminative_terms(self, status_groups: Dict) -> Dict:
        """
        Find terms that are particularly discriminative for each status

        Args:
            status_groups: Output from extract_word_frequencies

        Returns:
            Dictionary with discriminative terms analysis
        """
        discriminative_terms = {}

        # Calculate relative frequencies
        for target_status, target_data in status_groups.items():
            target_freq = target_data['word_freq']
            target_total = sum(target_freq.values())

            # Compare against all other statuses combined
            other_freq = Counter()
            other_total = 0

            for other_status, other_data in status_groups.items():
                if other_status != target_status:
                    other_freq.update(other_data['word_freq'])
                    other_total += sum(other_data['word_freq'].values())

            # Calculate relative enrichment (how much more frequent in target vs others)
            enriched_terms = {}
            for word, count in target_freq.items():
                if count >= 3:  # Minimum frequency threshold
                    target_rate = count / target_total
                    other_rate = other_freq.get(word, 0) / max(other_total, 1)

                    if other_rate > 0:
                        enrichment = target_rate / other_rate
                        if enrichment > 1.5:  # At least 1.5x more frequent
                            enriched_terms[word] = {
                                'enrichment': enrichment,
                                'target_count': count,
                                'target_rate': target_rate,
                                'other_count': other_freq.get(word, 0),
                                'other_rate': other_rate
                            }

            # Sort by enrichment
            sorted_terms = sorted(enriched_terms.items(), key=lambda x: x[1]['enrichment'], reverse=True)
            discriminative_terms[target_status] = sorted_terms[:30]  # Top 30 discriminative terms

        return discriminative_terms

    def analyze_temporal_patterns(self, notes_df: pd.DataFrame, claims_df: pd.DataFrame) -> Dict:
        """
        Analyze how language patterns change over time within claims

        Args:
            notes_df: DataFrame with notes
            claims_df: DataFrame with claim information

        Returns:
            Dictionary with temporal pattern analysis
        """
        # Merge and sort by time
        merged_df = notes_df.merge(
            claims_df[['clmNum', 'clmStatus', 'dateReopened']],
            on='clmNum',
            how='inner'
        )

        merged_df = merged_df.sort_values(['clmNum', 'whenadded'])

        # Create granular status
        def create_granular_status(row):
            base_status = row['clmStatus']
            if base_status in ['PAID', 'DENIED', 'CLOSED']:
                is_reopened = pd.notna(row['dateReopened'])
                if is_reopened:
                    return f"{base_status}_REOPENED"
                else:
                    return base_status
            else:
                return base_status

        merged_df['granular_status'] = merged_df.apply(create_granular_status, axis=1)

        temporal_patterns = {}

        # Analyze by final status
        for status in merged_df['granular_status'].unique():
            status_claims = merged_df[merged_df['granular_status'] == status]

            # Group by claim and analyze note sequence
            claim_patterns = []
            for claim_num in status_claims['clmNum'].unique():
                claim_notes = status_claims[status_claims['clmNum'] == claim_num].sort_values('whenadded')

                if len(claim_notes) >= 2:  # Need at least 2 notes for temporal analysis
                    # Analyze first vs last note
                    first_note = self.preprocess_text(claim_notes.iloc[0]['note'])
                    last_note = self.preprocess_text(claim_notes.iloc[-1]['note'])

                    claim_patterns.append({
                        'claim_num': claim_num,
                        'first_note': first_note,
                        'last_note': last_note,
                        'note_count': len(claim_notes),
                        'time_span_days': (claim_notes['whenadded'].max() - claim_notes['whenadded'].min()).days
                    })

            temporal_patterns[status] = claim_patterns

        return temporal_patterns

    def generate_comprehensive_report(self, notes_df: pd.DataFrame, claims_df: pd.DataFrame,
                                     progress_callback=None) -> Dict:
        """
        Generate comprehensive pattern analysis report

        Args:
            notes_df: DataFrame with notes
            claims_df: DataFrame with claim information
            progress_callback: Optional callback function for progress updates (percent, message)

        Returns:
            Complete analysis report
        """
        def update_progress(percent, message):
            if progress_callback:
                progress_callback(percent, message)
            else:
                print(f"{message} ({percent:.0%})")

        update_progress(0.0, "🔍 Starting comprehensive pattern analysis...")

        # Extract word frequencies
        update_progress(0.1, "1. Analyzing word frequencies by status...")
        status_groups = self.extract_word_frequencies(notes_df, claims_df)

        # Extract n-gram patterns
        update_progress(0.3, "2. Extracting n-gram patterns...")
        ngram_patterns = self.extract_ngram_patterns(status_groups)

        # Calculate TF-IDF features
        update_progress(0.5, "3. Calculating TF-IDF discriminative features...")
        tfidf_analysis = self.calculate_tfidf_features(status_groups)

        # Find discriminative terms
        update_progress(0.7, "4. Finding discriminative terms...")
        discriminative_terms = self.find_discriminative_terms(status_groups)

        # Analyze temporal patterns
        update_progress(0.9, "5. Analyzing temporal patterns...")
        temporal_patterns = self.analyze_temporal_patterns(notes_df, claims_df)

        # Compile comprehensive report
        update_progress(0.95, "6. Compiling comprehensive report...")
        report = {
            'summary': {
                'total_notes': len(notes_df),
                'unique_claims': notes_df['clmNum'].nunique(),
                'status_groups': {status: data['num_claims'] for status, data in status_groups.items()},
                'analysis_timestamp': pd.Timestamp.now()
            },
            'word_frequencies': status_groups,
            'ngram_patterns': ngram_patterns,
            'tfidf_analysis': tfidf_analysis,
            'discriminative_terms': discriminative_terms,
            'temporal_patterns': temporal_patterns
        }

        update_progress(1.0, "✅ Comprehensive pattern analysis completed!")
        return report

def create_pattern_visualizations(report: Dict):
    """
    Create visualizations for pattern analysis

    Args:
        report: Output from generate_comprehensive_report
    """
    # Top discriminative terms visualization
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    axes = axes.flatten()

    discriminative_terms = report['discriminative_terms']

    for i, (status, terms) in enumerate(discriminative_terms.items()):
        if i >= 6:  # Limit to 6 subplots
            break

        if terms:  # Check if there are terms
            words = [term[0] for term in terms[:10]]  # Top 10 terms
            enrichments = [term[1]['enrichment'] for term in terms[:10]]

            axes[i].barh(words, enrichments)
            axes[i].set_title(f'Top Discriminative Terms - {status}')
            axes[i].set_xlabel('Enrichment Factor')
        else:
            axes[i].text(0.5, 0.5, f'No discriminative terms\nfound for {status}',
                        ha='center', va='center', transform=axes[i].transAxes)
            axes[i].set_title(f'Discriminative Terms - {status}')

    # Hide empty subplots
    for j in range(len(discriminative_terms), 6):
        axes[j].set_visible(False)

    plt.tight_layout()
    return fig

if __name__ == "__main__":
    # Example usage
    print("Phase 1: Pattern Analysis for Enhanced Keywords")
    print("This module analyzes note patterns to discover data-driven insights.")