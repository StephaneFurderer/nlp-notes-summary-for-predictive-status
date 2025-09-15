import pandas as pd
import numpy as np
import re
from typing import Dict, List, Tuple, Optional
from collections import Counter
import os

class ClaimNotesNLPAnalyzer:
    """
    NLP analyzer for extracting features from insurance claim notes
    to predict claim status outcomes.
    """

    def __init__(self):
        """Initialize the NLP analyzer with insurance domain-specific keywords"""
        self.keyword_categories = self._build_keyword_dictionaries()
        self.sentiment_weights = self._build_sentiment_weights()

    def _build_keyword_dictionaries(self) -> Dict[str, List[str]]:
        """Build domain-specific keyword dictionaries for insurance claims"""
        return {
            # Financial and monetary terms
            'financial_terms': [
                'estimate', 'quote', 'deductible', 'coverage', 'limit', 'premium',
                'settlement', 'payment', 'reimbursement', 'cost', 'expense', 'damage assessment',
                'total loss', 'salvage', 'depreciation', 'replacement cost', 'actual cash value'
            ],

            # Process and investigation terms
            'process_terms': [
                'investigation', 'review', 'inspect', 'evaluate', 'assess', 'verify',
                'documentation', 'evidence', 'proof', 'statement', 'report', 'analysis',
                'follow-up', 'pending', 'processing', 'adjuster', 'examination'
            ],

            # Legal and dispute terms
            'legal_terms': [
                'attorney', 'lawyer', 'legal', 'lawsuit', 'litigation', 'subpoena',
                'court', 'dispute', 'disagreement', 'challenge', 'appeal', 'hearing',
                'representation', 'counsel', 'defendant', 'plaintiff', 'liability'
            ],

            # Medical and injury terms
            'medical_terms': [
                'medical', 'injury', 'treatment', 'hospital', 'doctor', 'physician',
                'therapy', 'rehabilitation', 'diagnosis', 'surgery', 'medication',
                'disability', 'recovery', 'health', 'clinical', 'medical records'
            ],

            # Fraud and investigation terms
            'fraud_terms': [
                'fraud', 'fraudulent', 'suspicious', 'questionable', 'inconsistent',
                'investigation', 'suspicious activity', 'false claim', 'misrepresentation',
                'verify', 'authenticate', 'confirm', 'validate'
            ],

            # Positive outcome indicators
            'positive_terms': [
                'approved', 'agreed', 'accepted', 'straightforward', 'clear', 'documented',
                'cooperative', 'responsive', 'complete', 'satisfied', 'resolved',
                'settled', 'finalized', 'concluded', 'successful'
            ],

            # Negative outcome indicators
            'negative_terms': [
                'denied', 'rejected', 'disputed', 'disagreed', 'contested', 'declined',
                'uncooperative', 'unresponsive', 'incomplete', 'insufficient',
                'complications', 'delays', 'problems', 'issues', 'concerns'
            ],

            # Urgency and time-sensitive terms
            'urgency_terms': [
                'urgent', 'immediate', 'deadline', 'time-sensitive', 'expedite',
                'rush', 'priority', 'asap', 'critical', 'emergency'
            ],

            # Communication and contact terms
            'communication_terms': [
                'contact', 'call', 'email', 'message', 'communicate', 'discuss',
                'meeting', 'appointment', 'conference', 'conversation', 'correspondence'
            ]
        }

    def _build_sentiment_weights(self) -> Dict[str, float]:
        """Build sentiment weight mappings for keywords"""
        return {
            # Very positive (likely to result in payment/closure)
            'approved': 2.0, 'agreed': 1.8, 'accepted': 1.6, 'straightforward': 1.4,
            'clear': 1.2, 'documented': 1.0, 'cooperative': 1.2, 'responsive': 1.0,
            'complete': 1.0, 'satisfied': 1.6, 'resolved': 2.0, 'settled': 2.0,

            # Neutral terms
            'investigation': 0.0, 'review': 0.0, 'pending': 0.0, 'processing': 0.0,

            # Negative (likely to result in denial/complications)
            'denied': -2.0, 'rejected': -1.8, 'disputed': -1.6, 'disagreed': -1.4,
            'contested': -1.2, 'declined': -1.8, 'uncooperative': -1.4, 'unresponsive': -1.2,
            'incomplete': -1.0, 'insufficient': -1.2, 'complications': -1.4, 'delays': -1.0,
            'fraud': -2.0, 'fraudulent': -2.0, 'suspicious': -1.6, 'questionable': -1.4,
            'legal': -0.8, 'attorney': -0.6, 'lawsuit': -1.8, 'litigation': -1.6
        }

    def extract_keywords(self, text: str) -> Dict[str, int]:
        """Extract keyword counts from text by category"""
        if pd.isna(text) or not isinstance(text, str):
            return {category: 0 for category in self.keyword_categories.keys()}

        text_lower = text.lower()
        keyword_counts = {}

        for category, keywords in self.keyword_categories.items():
            count = 0
            for keyword in keywords:
                # Use word boundaries to avoid partial matches
                pattern = r'\b' + re.escape(keyword.lower()) + r'\b'
                count += len(re.findall(pattern, text_lower))
            keyword_counts[category] = count

        return keyword_counts

    def calculate_sentiment_score(self, text: str) -> float:
        """Calculate sentiment score based on keyword weights"""
        if pd.isna(text) or not isinstance(text, str):
            return 0.0

        text_lower = text.lower()
        total_score = 0.0
        word_count = 0

        for word, weight in self.sentiment_weights.items():
            pattern = r'\b' + re.escape(word.lower()) + r'\b'
            matches = len(re.findall(pattern, text_lower))
            if matches > 0:
                total_score += weight * matches
                word_count += matches

        # Normalize by word count to avoid bias toward longer texts
        if word_count > 0:
            return total_score / word_count
        else:
            return 0.0

    def extract_text_features(self, text: str) -> Dict[str, float]:
        """Extract comprehensive text features from a note"""
        if pd.isna(text) or not isinstance(text, str):
            return {
                'char_count': 0, 'word_count': 0, 'sentence_count': 0,
                'avg_word_length': 0, 'unique_word_ratio': 0,
                'question_count': 0, 'exclamation_count': 0,
                'number_count': 0, 'capital_ratio': 0
            }

        # Basic text statistics
        char_count = len(text)
        words = text.split()
        word_count = len(words)
        sentences = re.split(r'[.!?]+', text)
        sentence_count = len([s for s in sentences if s.strip()])

        # Advanced features
        avg_word_length = np.mean([len(word) for word in words]) if words else 0
        unique_words = set(word.lower() for word in words)
        unique_word_ratio = len(unique_words) / word_count if word_count > 0 else 0

        # Punctuation analysis
        question_count = text.count('?')
        exclamation_count = text.count('!')

        # Number mentions (potential monetary amounts or dates)
        number_count = len(re.findall(r'\d+', text))

        # Capital letter ratio (indicates urgency/emphasis)
        capital_count = sum(1 for c in text if c.isupper())
        capital_ratio = capital_count / char_count if char_count > 0 else 0

        return {
            'char_count': char_count,
            'word_count': word_count,
            'sentence_count': sentence_count,
            'avg_word_length': avg_word_length,
            'unique_word_ratio': unique_word_ratio,
            'question_count': question_count,
            'exclamation_count': exclamation_count,
            'number_count': number_count,
            'capital_ratio': capital_ratio
        }

    def analyze_notes_for_claim(self, notes_df: pd.DataFrame, claim_num: str) -> Dict[str, float]:
        """Analyze all notes for a specific claim and return aggregated features"""
        claim_notes = notes_df[notes_df['clmNum'] == claim_num].copy()

        if len(claim_notes) == 0:
            # Return zero features for claims with no notes
            base_features = {f'{cat}_count': 0 for cat in self.keyword_categories.keys()}
            base_features.update({
                'sentiment_score': 0.0, 'sentiment_variance': 0.0,
                'total_notes': 0, 'avg_note_length': 0, 'total_text_length': 0,
                'communication_frequency': 0.0, 'last_note_sentiment': 0.0
            })
            return base_features

        # Sort notes by date
        claim_notes = claim_notes.sort_values('whenadded')

        # Extract features for each note
        all_keyword_counts = []
        all_sentiment_scores = []
        all_text_features = []

        for _, note_row in claim_notes.iterrows():
            note_text = note_row['note']

            # Keyword extraction
            keyword_counts = self.extract_keywords(note_text)
            all_keyword_counts.append(keyword_counts)

            # Sentiment analysis
            sentiment = self.calculate_sentiment_score(note_text)
            all_sentiment_scores.append(sentiment)

            # Text features
            text_features = self.extract_text_features(note_text)
            all_text_features.append(text_features)

        # Aggregate keyword counts across all notes
        aggregated_keywords = {}
        for category in self.keyword_categories.keys():
            category_total = sum(counts[category] for counts in all_keyword_counts)
            aggregated_keywords[f'{category}_count'] = category_total

        # Calculate sentiment statistics
        sentiment_scores = np.array(all_sentiment_scores)
        sentiment_mean = np.mean(sentiment_scores)
        sentiment_variance = np.var(sentiment_scores) if len(sentiment_scores) > 1 else 0.0
        last_note_sentiment = sentiment_scores[-1] if len(sentiment_scores) > 0 else 0.0

        # Calculate temporal features
        total_notes = len(claim_notes)
        if total_notes > 1:
            time_span = (claim_notes['whenadded'].max() - claim_notes['whenadded'].min()).days
            communication_frequency = total_notes / max(time_span, 1)  # notes per day
        else:
            communication_frequency = 0.0

        # Aggregate text features
        total_text_length = sum(tf['char_count'] for tf in all_text_features)
        avg_note_length = total_text_length / total_notes if total_notes > 0 else 0

        # Combine all features
        features = aggregated_keywords.copy()
        features.update({
            'sentiment_score': sentiment_mean,
            'sentiment_variance': sentiment_variance,
            'total_notes': total_notes,
            'avg_note_length': avg_note_length,
            'total_text_length': total_text_length,
            'communication_frequency': communication_frequency,
            'last_note_sentiment': last_note_sentiment
        })

        return features

    def analyze_all_claims(self, notes_df: pd.DataFrame) -> pd.DataFrame:
        """Analyze notes for all claims and return a feature dataframe"""
        unique_claims = notes_df['clmNum'].unique()

        all_features = []
        for claim_num in unique_claims:
            features = self.analyze_notes_for_claim(notes_df, claim_num)
            features['clmNum'] = claim_num
            all_features.append(features)

        # Convert to DataFrame
        features_df = pd.DataFrame(all_features)

        # Reorder columns to put clmNum first
        cols = ['clmNum'] + [col for col in features_df.columns if col != 'clmNum']
        features_df = features_df[cols]

        return features_df

def create_nlp_feature_summary(features_df: pd.DataFrame) -> pd.DataFrame:
    """Create a summary of NLP features for analysis"""
    numeric_cols = features_df.select_dtypes(include=[np.number]).columns
    summary = features_df[numeric_cols].describe()
    return summary

if __name__ == "__main__":
    # Example usage
    from notes_utils import NotesReviewerAgent

    # Load notes data
    agent = NotesReviewerAgent()
    notes_df = agent.import_notes()

    # Initialize NLP analyzer
    nlp_analyzer = ClaimNotesNLPAnalyzer()

    # Analyze all claims
    nlp_features = nlp_analyzer.analyze_all_claims(notes_df)

    print("NLP Features extracted for", len(nlp_features), "claims")
    print("\nFeature columns:")
    print(nlp_features.columns.tolist())

    print("\nSample features for first claim:")
    print(nlp_features.head(1).T)

    # Create summary
    summary = create_nlp_feature_summary(nlp_features)
    print("\nNLP Features Summary:")
    print(summary)