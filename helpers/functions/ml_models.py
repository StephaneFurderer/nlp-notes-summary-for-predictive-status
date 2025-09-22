import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import spacy
from collections import Counter, defaultdict
import os
from typing import Dict, Tuple, List, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

class ClaimStatusBaselineModel:
    """
    Baseline Random Forest model for predicting claim status from NLP features.
    """

    def __init__(self, random_state=42):
        """Initialize the baseline model"""
        self.random_state = random_state
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=random_state,
            class_weight='balanced'  # Handle class imbalance
        )
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.feature_columns = None
        self.model_trained = False
        self.training_history = {}

    def prepare_features(self, nlp_features_df: pd.DataFrame, claims_df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare features for training by merging NLP features with claim status.
        Creates granular target classes: PAID/CLOSED/DENIED with _REOPENED suffix for reopened claims.

        Args:
            nlp_features_df: DataFrame with NLP features
            claims_df: DataFrame with claim information including status

        Returns:
            DataFrame ready for training
        """
        # Get available financial columns
        financial_cols = ['clmNum', 'clmStatus', 'clmCause', 'dateReopened']
        available_financial = []
        for col in ['current_incurred', 'current_paid', 'current_expense', 'incurred_cumsum', 'paid_cumsum', 'expense_cumsum']:
            if col in claims_df.columns:
                available_financial.append(col)

        merge_cols = financial_cols + available_financial

        # Merge NLP features with claim status
        merged_df = nlp_features_df.merge(
            claims_df[merge_cols],
            on='clmNum',
            how='inner'
        )

        # Remove rows with missing target
        merged_df = merged_df.dropna(subset=['clmStatus'])

        # Filter to training eligible statuses only (exclude INITIAL_REVIEW)
        training_statuses = ['PAID', 'DENIED', 'CLOSED']
        merged_df = merged_df[merged_df['clmStatus'].isin(training_statuses)]

        # Create granular target classes based on reopened status
        def create_granular_status(row):
            base_status = row['clmStatus']
            is_reopened = pd.notna(row['dateReopened'])

            if is_reopened:
                return f"{base_status}_REOPENED"
            else:
                return base_status

        merged_df['granular_status'] = merged_df.apply(create_granular_status, axis=1)

        print(f"Prepared dataset with {len(merged_df)} samples")
        print(f"Original status distribution:\n{merged_df['clmStatus'].value_counts()}")
        print(f"Granular status distribution:\n{merged_df['granular_status'].value_counts()}")

        return merged_df

    def select_features(self, df: pd.DataFrame) -> List[str]:
        """
        Select relevant features for training.

        Args:
            df: Prepared dataframe

        Returns:
            List of feature column names
        """
        # NLP features (exclude clmNum and target)
        nlp_features = [col for col in df.columns if col.endswith('_count')]
        nlp_features += ['sentiment_score', 'sentiment_variance', 'total_notes',
                        'avg_note_length', 'total_text_length', 'communication_frequency',
                        'last_note_sentiment']

        # Financial features if available (check multiple possible column names)
        possible_financial = ['current_incurred', 'current_paid', 'current_expense',
                             'incurred_cumsum', 'paid_cumsum', 'expense_cumsum']
        financial_features = [col for col in possible_financial if col in df.columns]

        # Combine all features
        all_features = nlp_features + financial_features

        # Only keep features that exist in the dataframe
        selected_features = [col for col in all_features if col in df.columns]

        print(f"Selected {len(selected_features)} features for training")
        return selected_features

    def train(self, nlp_features_df: pd.DataFrame, claims_df: pd.DataFrame,
              test_size: float = 0.2, validation_split: float = 0.2) -> Dict:
        """
        Train the baseline model.

        Args:
            nlp_features_df: NLP features
            claims_df: Claims data with status
            test_size: Proportion for test set
            validation_split: Proportion for validation set

        Returns:
            Training results dictionary
        """
        # Prepare data
        df = self.prepare_features(nlp_features_df, claims_df)

        if len(df) < 10:
            raise ValueError("Insufficient data for training (need at least 10 samples)")

        # Select features
        self.feature_columns = self.select_features(df)

        # Prepare X and y
        X = df[self.feature_columns].fillna(0)  # Fill NaN with 0
        y = df['granular_status']  # Use granular status as target

        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_encoded, test_size=test_size,
            random_state=self.random_state, stratify=y_encoded
        )

        # Train model
        print("Training Random Forest model...")
        self.model.fit(X_train, y_train)
        self.model_trained = True

        # Evaluate on training set
        train_pred = self.model.predict(X_train)
        train_proba = self.model.predict_proba(X_train)
        train_accuracy = accuracy_score(y_train, train_pred)

        # Evaluate on test set
        test_pred = self.model.predict(X_test)
        test_proba = self.model.predict_proba(X_test)
        test_accuracy = accuracy_score(y_test, test_pred)

        # Cross-validation
        cv_scores = cross_val_score(self.model, X_train, y_train, cv=5, scoring='accuracy')

        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)

        # Store training results
        self.training_history = {
            'timestamp': datetime.now(),
            'n_samples': len(df),
            'n_features': len(self.feature_columns),
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'feature_importance': feature_importance,
            'classification_report': classification_report(
                y_test, test_pred,
                target_names=self.label_encoder.classes_,
                output_dict=True
            ),
            'confusion_matrix': confusion_matrix(y_test, test_pred),
            'class_names': self.label_encoder.classes_,
            'test_actual': self.label_encoder.inverse_transform(y_test),
            'test_predicted': self.label_encoder.inverse_transform(test_pred)
        }

        print(f"Training completed!")
        print(f"Train Accuracy: {train_accuracy:.3f}")
        print(f"Test Accuracy: {test_accuracy:.3f}")
        print(f"CV Accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")

        return self.training_history

    def predict(self, nlp_features_df: pd.DataFrame, claims_df: pd.DataFrame = None) -> pd.DataFrame:
        """
        Make predictions on new data.

        Args:
            nlp_features_df: NLP features for prediction
            claims_df: Claims DataFrame with financial features (optional)

        Returns:
            DataFrame with predictions and probabilities
        """
        if not self.model_trained:
            raise ValueError("Model must be trained before making predictions")

        # If claims_df is provided, merge to get financial features
        if claims_df is not None:
            # Get available financial columns
            financial_cols = ['clmNum']
            available_financial = []
            for col in ['current_incurred', 'current_paid', 'current_expense', 'incurred_cumsum', 'paid_cumsum', 'expense_cumsum']:
                if col in claims_df.columns:
                    available_financial.append(col)

            merge_cols = financial_cols + available_financial

            # Merge NLP features with financial features
            prediction_df = nlp_features_df.merge(
                claims_df[merge_cols],
                on='clmNum',
                how='left'
            )
        else:
            prediction_df = nlp_features_df.copy()

        # Only use features that were used during training and are available
        available_features = [col for col in self.feature_columns if col in prediction_df.columns]
        missing_features = [col for col in self.feature_columns if col not in prediction_df.columns]

        if missing_features:
            print(f"Warning: Missing features for prediction: {missing_features}")
            # Create a DataFrame with only available features
            X = prediction_df[available_features].fillna(0)

            # Add missing features as zeros
            for feature in missing_features:
                X[feature] = 0

            # Reorder to match training feature order
            X = X[self.feature_columns]
        else:
            X = prediction_df[self.feature_columns].fillna(0)

        X_scaled = self.scaler.transform(X)

        # Make predictions
        predictions = self.model.predict(X_scaled)
        probabilities = self.model.predict_proba(X_scaled)

        # Create results dataframe
        results = nlp_features_df[['clmNum']].copy()
        results['predicted_status'] = self.label_encoder.inverse_transform(predictions)
        results['prediction_confidence'] = np.max(probabilities, axis=1)

        # Add probability for each class
        for i, class_name in enumerate(self.label_encoder.classes_):
            results[f'prob_{class_name}'] = probabilities[:, i]

        return results

    def get_feature_importance(self, top_n: int = 10) -> pd.DataFrame:
        """Get top N most important features"""
        if not self.model_trained:
            raise ValueError("Model must be trained first")

        return self.training_history['feature_importance'].head(top_n)

    def save_model(self, filepath: str):
        """Save the trained model"""
        if not self.model_trained:
            raise ValueError("Model must be trained before saving")

        model_data = {
            'model': self.model,
            'label_encoder': self.label_encoder,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'training_history': self.training_history
        }

        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath: str):
        """Load a previously trained model"""
        model_data = joblib.load(filepath)

        self.model = model_data['model']
        self.label_encoder = model_data['label_encoder']
        self.scaler = model_data['scaler']
        self.feature_columns = model_data['feature_columns']
        self.training_history = model_data['training_history']
        self.model_trained = True

        print(f"Model loaded from {filepath}")

class ClaimStatusEnhancedModel(ClaimStatusBaselineModel):
    """
    Enhanced Random Forest model that incorporates TF-IDF features
    from pattern analysis for improved prediction accuracy.
    """

    def __init__(self, random_state=42, max_tfidf_features=50):
        """Initialize the enhanced model with TF-IDF capabilities"""
        super().__init__(random_state)
        self.max_tfidf_features = max_tfidf_features
        self.tfidf_vectorizer = None
        self.tfidf_feature_names = []
        self.enhanced_feature_columns = None

    def prepare_tfidf_features(self, notes_df: pd.DataFrame, claims_df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare TF-IDF features from claim notes based on pattern analysis insights.

        Args:
            notes_df: DataFrame with claim notes
            claims_df: DataFrame with claim information

        Returns:
            DataFrame with TF-IDF features by claim
        """
        # Merge notes with claim status for filtering
        merged_df = notes_df.merge(
            claims_df[['clmNum', 'clmStatus', 'dateReopened']],
            on='clmNum',
            how='inner'
        )

        # Filter to training eligible statuses only
        training_statuses = ['PAID', 'DENIED', 'CLOSED']
        train_notes = merged_df[merged_df['clmStatus'].isin(training_statuses)]

        # Aggregate notes by claim
        claim_texts = train_notes.groupby('clmNum')['note'].apply(
            lambda x: ' '.join(x.fillna('').astype(str))
        ).reset_index()

        # Initialize TF-IDF vectorizer with liability-specific settings
        liability_stopwords = [
            # Generic insurance terms
            'claim', 'claims', 'claimant', 'policy', 'policyholder', 'insured',
            'insurance', 'company', 'adjuster', 'agent', 'representative',
            'will', 'would', 'could', 'should', 'may', 'might', 'can',
            'please', 'thank', 'thanks', 'regards', 'sincerely',
            'contact', 'call', 'email', 'phone', 'number', 'date', 'time',
            'file', 'received', 'sent', 'message', 'report', 'information',
            'note', 'notes', 'updated', 'status', 'review', 'reviewed',
            'pm', 'am', 'today', 'yesterday', 'tomorrow', 'monday', 'tuesday',
            'wednesday', 'thursday', 'friday', 'saturday', 'sunday',

            # Function words and generic terms that appear as false positives
            'has', 'our', 'null', 'there', 'subject', 'role', 'this', 'that',
            'with', 'from', 'they', 'them', 'their', 'these', 'those',
            'been', 'have', 'had', 'was', 'were', 'are', 'the', 'and',
            'for', 'but', 'not', 'you', 'all', 'any', 'can', 'had',

            # pecific terms to remove (not relevant for property liability)
            'salvage', 'salvaged', 'total', 'totaled', 'totalling',
            'auto', 'vehicle', 'car', 'truck', 'motorcycle', 'collision',
            'comprehensive', 'deductible', 'mechanic',
            'garage', 'towing', 'towed', 'estimate', 'estimates',
            'parts', 'labor', 'bodyshop', 'body', 'shop',
            'windshield', 'glass', 'bumper', 'fender', 'door',
            'engine', 'transmission', 'brake', 'tire', 'wheel',

            # Generic administrative terms
            'received', 'sent', 'forwarded', 'attached', 'copy',
            'original', 'signed', 'signature', 'document', 'documents',
            'form', 'forms', 'application', 'submitted', 'processing'
        ]

        def clean_text_for_tfidf(text):
            """Enhanced text cleaning for TF-IDF"""
            import re

            # Convert to lowercase
            text = text.lower()

            # Remove common patterns that aren't meaningful
            # Remove dates (various formats)
            text = re.sub(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b', '', text)
            text = re.sub(r'\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b', '', text)

            # Remove times
            text = re.sub(r'\b\d{1,2}:\d{2}(?::\d{2})?\s*(?:am|pm)?\b', '', text)

            # Remove standalone numbers (but keep words with numbers)
            text = re.sub(r'\b\d+\.?\d*\b', '', text)

            # Remove dollar amounts
            text = re.sub(r'\$\d+(?:,\d{3})*(?:\.\d{2})?', '', text)

            # Remove common abbreviations and codes
            text = re.sub(r'\b[A-Z]{2,}\b', '', text)  # Remove all-caps abbreviations

            # Remove email-like patterns
            text = re.sub(r'\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b', '', text)

            # Remove phone-like patterns
            text = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '', text)

            # Remove single characters and very short words
            text = re.sub(r'\b[a-z]{1,2}\b', '', text)

            # Clean up whitespace
            text = re.sub(r'\s+', ' ', text).strip()

            return text

        # Pre-process texts with enhanced cleaning
        cleaned_texts = claim_texts['note'].apply(clean_text_for_tfidf)

        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=self.max_tfidf_features,
            stop_words=liability_stopwords,
            ngram_range=(1, 2),  # Include unigrams and bigrams
            min_df=3,  # Must appear in at least 3 claims (increased from 2)
            max_df=0.7,  # Must not appear in more than 70% of claims (decreased from 80%)
            lowercase=True,
            token_pattern=r'\b[a-zA-Z][a-zA-Z]+\b',  # Only alphabetic words, 2+ chars
            strip_accents='ascii'  # Remove accents
        )

        # Fit and transform the cleaned texts
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(cleaned_texts)
        self.tfidf_feature_names = [f"tfidf_{name}" for name in self.tfidf_vectorizer.get_feature_names_out()]

        # Create DataFrame with TF-IDF features
        tfidf_df = pd.DataFrame(
            tfidf_matrix.toarray(),
            columns=self.tfidf_feature_names
        )
        tfidf_df['clmNum'] = claim_texts['clmNum'].values

        print(f"Created {len(self.tfidf_feature_names)} TF-IDF features from {len(claim_texts)} claims")

        return tfidf_df

    def prepare_enhanced_features(self, nlp_features_df: pd.DataFrame,
                                  claims_df: pd.DataFrame, notes_df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare enhanced features combining NLP features with TF-IDF features.

        Args:
            nlp_features_df: DataFrame with existing NLP features
            claims_df: DataFrame with claim information
            notes_df: DataFrame with claim notes

        Returns:
            DataFrame ready for enhanced training
        """
        # First prepare base features
        base_df = self.prepare_features(nlp_features_df, claims_df)

        # Prepare TF-IDF features
        tfidf_df = self.prepare_tfidf_features(notes_df, claims_df)

        # Merge TF-IDF features with base features
        enhanced_df = base_df.merge(tfidf_df, on='clmNum', how='left')

        # Fill NaN TF-IDF values with 0 (for claims not in TF-IDF training set)
        tfidf_cols = [col for col in enhanced_df.columns if col.startswith('tfidf_')]
        enhanced_df[tfidf_cols] = enhanced_df[tfidf_cols].fillna(0)

        print(f"Enhanced dataset with {len(enhanced_df)} samples and {len(tfidf_cols)} TF-IDF features")

        return enhanced_df

    def select_enhanced_features(self, df: pd.DataFrame) -> List[str]:
        """
        Select features for enhanced model including TF-IDF features.

        Args:
            df: Enhanced prepared dataframe

        Returns:
            List of enhanced feature column names
        """
        # Get base features
        base_features = super().select_features(df)

        # Add TF-IDF features
        tfidf_features = [col for col in df.columns if col.startswith('tfidf_')]

        # Combine all features
        enhanced_features = base_features + tfidf_features

        # Only keep features that exist in the dataframe
        selected_features = [col for col in enhanced_features if col in df.columns]

        print(f"Selected {len(selected_features)} enhanced features for training")
        print(f"  - Base NLP/Financial features: {len(base_features)}")
        print(f"  - TF-IDF features: {len(tfidf_features)}")

        return selected_features

    def train_enhanced(self, nlp_features_df: pd.DataFrame, claims_df: pd.DataFrame,
                      notes_df: pd.DataFrame, test_size: float = 0.2,
                      validation_split: float = 0.2) -> Dict:
        """
        Train the enhanced model with TF-IDF features.

        Args:
            nlp_features_df: NLP features
            claims_df: Claims data with status
            notes_df: Raw notes data for TF-IDF
            test_size: Proportion for test set
            validation_split: Proportion for validation set

        Returns:
            Enhanced training results dictionary
        """
        # Prepare enhanced data
        df = self.prepare_enhanced_features(nlp_features_df, claims_df, notes_df)

        if len(df) < 10:
            raise ValueError("Insufficient data for training (need at least 10 samples)")

        # Select enhanced features
        self.enhanced_feature_columns = self.select_enhanced_features(df)
        self.feature_columns = self.enhanced_feature_columns  # Update base class attribute

        # Prepare X and y
        X = df[self.enhanced_feature_columns].fillna(0)
        y = df['granular_status']

        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_encoded, test_size=test_size,
            random_state=self.random_state, stratify=y_encoded
        )

        # Train enhanced model with adjusted hyperparameters for more features
        print("Training Enhanced Random Forest model with TF-IDF features...")
        self.model = RandomForestClassifier(
            n_estimators=150,  # More trees for more features
            max_depth=15,  # Deeper trees
            min_samples_split=3,
            min_samples_leaf=2,
            random_state=self.random_state,
            class_weight='balanced',
            max_features='sqrt'  # Use sqrt for feature selection
        )

        self.model.fit(X_train, y_train)
        self.model_trained = True

        # Evaluate model
        train_pred = self.model.predict(X_train)
        train_accuracy = accuracy_score(y_train, train_pred)

        test_pred = self.model.predict(X_test)
        test_accuracy = accuracy_score(y_test, test_pred)

        # Cross-validation
        cv_scores = cross_val_score(self.model, X_train, y_train, cv=5, scoring='accuracy')

        # Enhanced feature importance analysis
        feature_importance = pd.DataFrame({
            'feature': self.enhanced_feature_columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)

        # Separate TF-IDF feature importance
        tfidf_importance = feature_importance[
            feature_importance['feature'].str.startswith('tfidf_')
        ].head(10)

        base_importance = feature_importance[
            ~feature_importance['feature'].str.startswith('tfidf_')
        ].head(10)

        # Store enhanced training results
        self.training_history = {
            'model_type': 'enhanced_tfidf',
            'timestamp': datetime.now(),
            'n_samples': len(df),
            'n_features': len(self.enhanced_feature_columns),
            'n_tfidf_features': len([f for f in self.enhanced_feature_columns if f.startswith('tfidf_')]),
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'feature_importance': feature_importance,
            'tfidf_importance': tfidf_importance,
            'base_importance': base_importance,
            'classification_report': classification_report(
                y_test, test_pred,
                target_names=self.label_encoder.classes_,
                output_dict=True
            ),
            'confusion_matrix': confusion_matrix(y_test, test_pred),
            'class_names': self.label_encoder.classes_,
            'test_actual': self.label_encoder.inverse_transform(y_test),
            'test_predicted': self.label_encoder.inverse_transform(test_pred)
        }

        print(f"Enhanced training completed!")
        print(f"Train Accuracy: {train_accuracy:.3f}")
        print(f"Test Accuracy: {test_accuracy:.3f}")
        print(f"CV Accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")

        return self.training_history

    def predict_enhanced(self, nlp_features_df: pd.DataFrame, notes_df: pd.DataFrame,
                        claims_df: pd.DataFrame = None) -> pd.DataFrame:
        """
        Make predictions using enhanced model with TF-IDF features.

        Args:
            nlp_features_df: NLP features for prediction
            notes_df: Raw notes for TF-IDF feature extraction
            claims_df: Claims DataFrame with financial features (optional)

        Returns:
            DataFrame with enhanced predictions and probabilities
        """
        if not self.model_trained or self.tfidf_vectorizer is None:
            raise ValueError("Enhanced model must be trained before making predictions")

        # Aggregate notes by claim for TF-IDF
        claim_texts = notes_df.groupby('clmNum')['note'].apply(
            lambda x: ' '.join(x.fillna('').astype(str))
        ).reset_index()

        # Apply same text cleaning as during training
        def clean_text_for_tfidf(text):
            """Enhanced text cleaning for TF-IDF"""
            import re

            # Convert to lowercase
            text = text.lower()

            # Remove common patterns that aren't meaningful
            # Remove dates (various formats)
            text = re.sub(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b', '', text)
            text = re.sub(r'\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b', '', text)

            # Remove times
            text = re.sub(r'\b\d{1,2}:\d{2}(?::\d{2})?\s*(?:am|pm)?\b', '', text)

            # Remove standalone numbers (but keep words with numbers)
            text = re.sub(r'\b\d+\.?\d*\b', '', text)

            # Remove dollar amounts
            text = re.sub(r'\$\d+(?:,\d{3})*(?:\.\d{2})?', '', text)

            # Remove common abbreviations and codes
            text = re.sub(r'\b[A-Z]{2,}\b', '', text)  # Remove all-caps abbreviations

            # Remove email-like patterns
            text = re.sub(r'\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b', '', text)

            # Remove phone-like patterns
            text = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '', text)

            # Remove single characters and very short words
            text = re.sub(r'\b[a-z]{1,2}\b', '', text)

            # Clean up whitespace
            text = re.sub(r'\s+', ' ', text).strip()

            return text

        # Clean the texts for prediction
        cleaned_texts = claim_texts['note'].apply(clean_text_for_tfidf)

        # Transform notes to TF-IDF features
        tfidf_matrix = self.tfidf_vectorizer.transform(cleaned_texts)
        tfidf_df = pd.DataFrame(
            tfidf_matrix.toarray(),
            columns=self.tfidf_feature_names
        )
        tfidf_df['clmNum'] = claim_texts['clmNum'].values

        # Merge with NLP features
        prediction_df = nlp_features_df.merge(tfidf_df, on='clmNum', how='left')

        # If claims_df is provided, merge financial features
        if claims_df is not None:
            financial_cols = ['clmNum']
            available_financial = []
            for col in ['current_incurred', 'current_paid', 'current_expense',
                       'incurred_cumsum', 'paid_cumsum', 'expense_cumsum']:
                if col in claims_df.columns:
                    available_financial.append(col)

            merge_cols = financial_cols + available_financial
            prediction_df = prediction_df.merge(
                claims_df[merge_cols], on='clmNum', how='left'
            )

        # Handle missing features
        available_features = [col for col in self.enhanced_feature_columns
                             if col in prediction_df.columns]
        missing_features = [col for col in self.enhanced_feature_columns
                           if col not in prediction_df.columns]

        if missing_features:
            print(f"Warning: Missing features for prediction: {len(missing_features)} features")

        # Create feature matrix with proper ordering
        X = prediction_df[available_features].fillna(0)

        # Add missing features as zeros
        for feature in missing_features:
            X[feature] = 0

        # Reorder to match training feature order
        X = X[self.enhanced_feature_columns]

        # Scale and predict
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        probabilities = self.model.predict_proba(X_scaled)

        # Create results dataframe
        results = nlp_features_df[['clmNum']].copy()
        results['predicted_status'] = self.label_encoder.inverse_transform(predictions)
        results['prediction_confidence'] = np.max(probabilities, axis=1)

        # Add probability for each class
        for i, class_name in enumerate(self.label_encoder.classes_):
            results[f'prob_{class_name}'] = probabilities[:, i]

        return results

    def save_model(self, filepath: str):
        """Save the enhanced model including TF-IDF vectorizer"""
        if not self.model_trained:
            raise ValueError("Enhanced model must be trained before saving")

        model_data = {
            'model': self.model,
            'label_encoder': self.label_encoder,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'enhanced_feature_columns': getattr(self, 'enhanced_feature_columns', None),
            'training_history': self.training_history,
            'tfidf_vectorizer': self.tfidf_vectorizer,
            'tfidf_feature_names': self.tfidf_feature_names,
            'max_tfidf_features': self.max_tfidf_features
        }

        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(model_data, filepath)
        print(f"Enhanced model saved to {filepath}")

    def load_model(self, filepath: str):
        """Load the enhanced model including TF-IDF vectorizer"""
        model_data = joblib.load(filepath)

        self.model = model_data['model']
        self.label_encoder = model_data['label_encoder']
        self.scaler = model_data['scaler']
        self.feature_columns = model_data['feature_columns']
        self.enhanced_feature_columns = model_data.get('enhanced_feature_columns', self.feature_columns)
        self.training_history = model_data['training_history']
        self.tfidf_vectorizer = model_data['tfidf_vectorizer']
        self.tfidf_feature_names = model_data['tfidf_feature_names']
        self.max_tfidf_features = model_data.get('max_tfidf_features', 50)
        self.model_trained = True

        print(f"Enhanced model loaded from {filepath}")

class ClaimStatusNERModel(ClaimStatusEnhancedModel):
    """
    NER-enhanced Random Forest model that incorporates Named Entity Recognition features
    on top of the enhanced TF-IDF model for maximum prediction accuracy.
    """

    def __init__(self, random_state=42, max_tfidf_features=50):
        """Initialize the NER-enhanced model"""
        super().__init__(random_state, max_tfidf_features)
        self.nlp_model = None
        self.ner_feature_names = []
        self.ner_enhanced_feature_columns = None

    def _load_spacy_model(self):
        """Load spaCy model for NER"""
        try:
            # Try to load the large English model first
            self.nlp_model = spacy.load("en_core_web_lg")
        except OSError:
            try:
                # Fall back to medium model
                self.nlp_model = spacy.load("en_core_web_md")
            except OSError:
                try:
                    # Fall back to small model
                    self.nlp_model = spacy.load("en_core_web_sm")
                except OSError:
                    raise ValueError("No spaCy English model found. Please install with: python -m spacy download en_core_web_sm")

        print(f"Loaded spaCy model: {self.nlp_model.meta['name']}")

    def extract_ner_features(self, notes_df: pd.DataFrame, claims_df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract Named Entity Recognition features from claim notes.

        Args:
            notes_df: DataFrame with claim notes
            claims_df: DataFrame with claim information

        Returns:
            DataFrame with NER features by claim
        """
        if self.nlp_model is None:
            self._load_spacy_model()

        # Merge notes with claim status for filtering
        merged_df = notes_df.merge(
            claims_df[['clmNum', 'clmStatus', 'dateReopened']],
            on='clmNum',
            how='inner'
        )

        # Filter to training eligible statuses only
        training_statuses = ['PAID', 'DENIED', 'CLOSED']
        train_notes = merged_df[merged_df['clmStatus'].isin(training_statuses)]

        # Aggregate notes by claim
        claim_texts = train_notes.groupby('clmNum')['note'].apply(
            lambda x: ' '.join(x.fillna('').astype(str))
        ).reset_index()

        print(f"Extracting NER features from {len(claim_texts)} claims...")

        # Define entity types relevant to insurance claims
        relevant_entities = {
            'PERSON': 'person_count',
            'ORG': 'organization_count',
            'GPE': 'location_count',  # Geopolitical entities (cities, countries)
            'MONEY': 'money_count',
            'DATE': 'date_count',
            'TIME': 'time_count',
            'PERCENT': 'percent_count',
            'FACILITY': 'facility_count',  # Buildings, airports, highways, bridges
            'EVENT': 'event_count',  # Named hurricanes, battles, wars, sports events
            'LAW': 'law_count',  # Named documents made into laws
            'PRODUCT': 'product_count'  # Objects, vehicles, foods, etc.
        }

        # Initialize feature dictionary
        ner_features = defaultdict(lambda: defaultdict(int))

        # Process each claim's text
        for idx, row in claim_texts.iterrows():
            claim_num = row['clmNum']
            text = row['note']

            if pd.isna(text) or len(text.strip()) == 0:
                continue

            # Process text with spaCy (limit text length for performance)
            text_limited = text[:10000]  # Limit to first 10k characters
            doc = self.nlp_model(text_limited)

            # Count entities by type
            entity_counts = Counter([ent.label_ for ent in doc.ents])

            # Extract relevant entity counts
            for ent_type, feature_name in relevant_entities.items():
                ner_features[claim_num][feature_name] = entity_counts.get(ent_type, 0)

            # Additional derived features
            ner_features[claim_num]['total_entities'] = len(doc.ents)
            ner_features[claim_num]['unique_entity_types'] = len(set([ent.label_ for ent in doc.ents]))

            # Entity density (entities per 100 words)
            if len(doc) > 0:
                ner_features[claim_num]['entity_density'] = (len(doc.ents) / len(doc)) * 100
            else:
                ner_features[claim_num]['entity_density'] = 0

            # Named entity diversity (unique entities / total entities)
            if len(doc.ents) > 0:
                unique_entities = len(set([ent.text.lower() for ent in doc.ents]))
                ner_features[claim_num]['entity_diversity'] = unique_entities / len(doc.ents)
            else:
                ner_features[claim_num]['entity_diversity'] = 0

        # Convert to DataFrame
        ner_df = pd.DataFrame.from_dict(ner_features, orient='index').fillna(0)
        ner_df.reset_index(inplace=True)
        ner_df.rename(columns={'index': 'clmNum'}, inplace=True)

        # Store feature names for later use
        self.ner_feature_names = [col for col in ner_df.columns if col != 'clmNum']

        print(f"Created {len(self.ner_feature_names)} NER features")
        return ner_df

    def prepare_ner_enhanced_features(self, nlp_features_df: pd.DataFrame,
                                     claims_df: pd.DataFrame, notes_df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare NER-enhanced features combining base + TF-IDF + NER features.

        Args:
            nlp_features_df: DataFrame with existing NLP features
            claims_df: DataFrame with claim information
            notes_df: DataFrame with claim notes

        Returns:
            DataFrame ready for NER-enhanced training
        """
        # First prepare enhanced features (base + TF-IDF)
        enhanced_df = self.prepare_enhanced_features(nlp_features_df, claims_df, notes_df)

        # Extract NER features
        ner_df = self.extract_ner_features(notes_df, claims_df)

        # Merge NER features with enhanced features
        ner_enhanced_df = enhanced_df.merge(ner_df, on='clmNum', how='left')

        # Fill NaN NER values with 0 (for claims not in NER training set)
        ner_cols = [col for col in ner_enhanced_df.columns if col in self.ner_feature_names]
        ner_enhanced_df[ner_cols] = ner_enhanced_df[ner_cols].fillna(0)

        print(f"NER-enhanced dataset with {len(ner_enhanced_df)} samples and {len(ner_cols)} NER features")

        return ner_enhanced_df

    def select_ner_enhanced_features(self, df: pd.DataFrame) -> List[str]:
        """
        Select features for NER-enhanced model including base + TF-IDF + NER features.

        Args:
            df: NER-enhanced prepared dataframe

        Returns:
            List of NER-enhanced feature column names
        """
        # Get enhanced features (base + TF-IDF)
        enhanced_features = super().select_enhanced_features(df)

        # Add NER features
        ner_features = [col for col in df.columns if col in self.ner_feature_names]

        # Combine all features
        ner_enhanced_features = enhanced_features + ner_features

        # Only keep features that exist in the dataframe
        selected_features = [col for col in ner_enhanced_features if col in df.columns]

        print(f"Selected {len(selected_features)} NER-enhanced features for training")
        print(f"  - Base NLP/Financial features: {len([f for f in selected_features if not f.startswith('tfidf_') and f not in self.ner_feature_names])}")
        print(f"  - TF-IDF features: {len([f for f in selected_features if f.startswith('tfidf_')])}")
        print(f"  - NER features: {len([f for f in selected_features if f in self.ner_feature_names])}")

        return selected_features

    def train_ner_enhanced(self, nlp_features_df: pd.DataFrame, claims_df: pd.DataFrame,
                          notes_df: pd.DataFrame, test_size: float = 0.2, progress_callback=None) -> Dict:
        """
        Train the NER-enhanced model with base + TF-IDF + NER features.

        Args:
            nlp_features_df: NLP features
            claims_df: Claims data with status
            notes_df: Raw notes data for TF-IDF and NER
            test_size: Proportion for test set
            progress_callback: Optional callback for progress updates

        Returns:
            NER-enhanced training results dictionary
        """
        def update_progress(percent, message):
            if progress_callback:
                progress_callback(percent, message)
            else:
                print(f"{message} ({percent:.0%})")

        update_progress(0.05, "🚀 Starting NER-enhanced model training...")

        # Prepare NER-enhanced data (this takes time due to spaCy processing)
        update_progress(0.1, "🔍 Extracting NER features from notes (this may take a few minutes)...")
        df = self.prepare_ner_enhanced_features(nlp_features_df, claims_df, notes_df)

        if len(df) < 10:
            raise ValueError("Insufficient data for training (need at least 10 samples)")

        update_progress(0.4, "✅ NER feature extraction completed, preparing training data...")

        # Select NER-enhanced features
        self.ner_enhanced_feature_columns = self.select_ner_enhanced_features(df)
        self.feature_columns = self.ner_enhanced_feature_columns  # Update base class attribute

        # Prepare X and y
        X = df[self.ner_enhanced_feature_columns].fillna(0)
        y = df['granular_status']

        update_progress(0.5, "🔢 Encoding labels and scaling features...")

        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_encoded, test_size=test_size,
            random_state=self.random_state, stratify=y_encoded
        )

        update_progress(0.6, "🌲 Training Random Forest model with enhanced features...")

        # Train NER-enhanced model with optimized hyperparameters
        self.model = RandomForestClassifier(
            n_estimators=200,  # More trees for more features
            max_depth=20,  # Deeper trees
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=self.random_state,
            class_weight='balanced',
            max_features='sqrt',  # Use sqrt for feature selection
            n_jobs=-1  # Use all processors
        )

        self.model.fit(X_train, y_train)
        self.model_trained = True

        update_progress(0.8, "📊 Evaluating model performance...")

        # Evaluate model
        train_pred = self.model.predict(X_train)
        train_accuracy = accuracy_score(y_train, train_pred)

        test_pred = self.model.predict(X_test)
        test_accuracy = accuracy_score(y_test, test_pred)

        # Cross-validation
        cv_scores = cross_val_score(self.model, X_train, y_train, cv=5, scoring='accuracy')

        update_progress(0.9, "🎯 Analyzing feature importance and generating report...")

        # NER-enhanced feature importance analysis
        feature_importance = pd.DataFrame({
            'feature': self.ner_enhanced_feature_columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)

        # Separate feature importance by type
        base_importance = feature_importance[
            (~feature_importance['feature'].str.startswith('tfidf_')) &
            (~feature_importance['feature'].isin(self.ner_feature_names))
        ].head(10)

        tfidf_importance = feature_importance[
            feature_importance['feature'].str.startswith('tfidf_')
        ].head(10)

        ner_importance = feature_importance[
            feature_importance['feature'].isin(self.ner_feature_names)
        ].head(10)

        # Store NER-enhanced training results
        self.training_history = {
            'model_type': 'ner_enhanced',
            'timestamp': datetime.now(),
            'n_samples': len(df),
            'n_features': len(self.ner_enhanced_feature_columns),
            'n_base_features': len([f for f in self.ner_enhanced_feature_columns if not f.startswith('tfidf_') and f not in self.ner_feature_names]),
            'n_tfidf_features': len([f for f in self.ner_enhanced_feature_columns if f.startswith('tfidf_')]),
            'n_ner_features': len([f for f in self.ner_enhanced_feature_columns if f in self.ner_feature_names]),
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'feature_importance': feature_importance,
            'base_importance': base_importance,
            'tfidf_importance': tfidf_importance,
            'ner_importance': ner_importance,
            'classification_report': classification_report(
                y_test, test_pred,
                target_names=self.label_encoder.classes_,
                output_dict=True
            ),
            'confusion_matrix': confusion_matrix(y_test, test_pred),
            'class_names': self.label_encoder.classes_,
            'test_actual': self.label_encoder.inverse_transform(y_test),
            'test_predicted': self.label_encoder.inverse_transform(test_pred)
        }

        update_progress(1.0, "🎉 NER-enhanced model training completed successfully!")

        print(f"NER-enhanced training completed!")
        print(f"Train Accuracy: {train_accuracy:.3f}")
        print(f"Test Accuracy: {test_accuracy:.3f}")
        print(f"CV Accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")

        return self.training_history

    def predict_ner_enhanced(self, nlp_features_df: pd.DataFrame, notes_df: pd.DataFrame,
                            claims_df: pd.DataFrame = None) -> pd.DataFrame:
        """
        Make predictions using NER-enhanced model.

        Args:
            nlp_features_df: NLP features for prediction
            notes_df: Raw notes for TF-IDF and NER feature extraction
            claims_df: Claims DataFrame with financial features (optional)

        Returns:
            DataFrame with NER-enhanced predictions and probabilities
        """
        if not self.model_trained or self.tfidf_vectorizer is None or self.nlp_model is None:
            raise ValueError("NER-enhanced model must be trained before making predictions")

        # First get enhanced predictions (base + TF-IDF)
        prediction_df = super().predict_enhanced(nlp_features_df, notes_df, claims_df)

        # Extract NER features for prediction claims
        pred_notes = notes_df[notes_df['clmNum'].isin(nlp_features_df['clmNum'])]
        pred_claim_texts = pred_notes.groupby('clmNum')['note'].apply(
            lambda x: ' '.join(x.fillna('').astype(str))
        ).reset_index()

        # Extract NER features
        ner_features = defaultdict(lambda: defaultdict(int))
        relevant_entities = {
            'PERSON': 'person_count', 'ORG': 'organization_count', 'GPE': 'location_count',
            'MONEY': 'money_count', 'DATE': 'date_count', 'TIME': 'time_count',
            'PERCENT': 'percent_count', 'FACILITY': 'facility_count',
            'EVENT': 'event_count', 'LAW': 'law_count', 'PRODUCT': 'product_count'
        }

        for idx, row in pred_claim_texts.iterrows():
            claim_num = row['clmNum']
            text = row['note']

            if pd.isna(text) or len(text.strip()) == 0:
                continue

            text_limited = text[:10000]
            doc = self.nlp_model(text_limited)

            entity_counts = Counter([ent.label_ for ent in doc.ents])

            for ent_type, feature_name in relevant_entities.items():
                ner_features[claim_num][feature_name] = entity_counts.get(ent_type, 0)

            ner_features[claim_num]['total_entities'] = len(doc.ents)
            ner_features[claim_num]['unique_entity_types'] = len(set([ent.label_ for ent in doc.ents]))

            if len(doc) > 0:
                ner_features[claim_num]['entity_density'] = (len(doc.ents) / len(doc)) * 100
            else:
                ner_features[claim_num]['entity_density'] = 0

            if len(doc.ents) > 0:
                unique_entities = len(set([ent.text.lower() for ent in doc.ents]))
                ner_features[claim_num]['entity_diversity'] = unique_entities / len(doc.ents)
            else:
                ner_features[claim_num]['entity_diversity'] = 0

        # Convert NER features to DataFrame
        ner_df = pd.DataFrame.from_dict(ner_features, orient='index').fillna(0)
        ner_df.reset_index(inplace=True)
        ner_df.rename(columns={'index': 'clmNum'}, inplace=True)

        # The predict_enhanced method from parent already handles the full pipeline
        # We just need to return the existing prediction_df since the model was trained
        # with NER features and the parent method uses the same feature columns
        return prediction_df

    def save_model(self, filepath: str):
        """Save the NER-enhanced model including TF-IDF vectorizer and spaCy model info"""
        if not self.model_trained:
            raise ValueError("NER-enhanced model must be trained before saving")

        # Save spaCy model name (not the model itself, too large)
        spacy_model_name = None
        if self.nlp_model is not None:
            spacy_model_name = self.nlp_model.meta.get('name', 'en_core_web_sm')

        model_data = {
            'model': self.model,
            'label_encoder': self.label_encoder,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'enhanced_feature_columns': getattr(self, 'enhanced_feature_columns', None),
            'ner_enhanced_feature_columns': getattr(self, 'ner_enhanced_feature_columns', None),
            'training_history': self.training_history,
            'tfidf_vectorizer': self.tfidf_vectorizer,
            'tfidf_feature_names': self.tfidf_feature_names,
            'ner_feature_names': self.ner_feature_names,
            'max_tfidf_features': self.max_tfidf_features,
            'spacy_model_name': spacy_model_name
        }

        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(model_data, filepath)
        print(f"NER-enhanced model saved to {filepath}")

    def load_model(self, filepath: str):
        """Load the NER-enhanced model including TF-IDF vectorizer and spaCy model"""
        model_data = joblib.load(filepath)

        self.model = model_data['model']
        self.label_encoder = model_data['label_encoder']
        self.scaler = model_data['scaler']
        self.feature_columns = model_data['feature_columns']
        self.enhanced_feature_columns = model_data.get('enhanced_feature_columns', self.feature_columns)
        self.ner_enhanced_feature_columns = model_data.get('ner_enhanced_feature_columns', self.feature_columns)
        self.training_history = model_data['training_history']
        self.tfidf_vectorizer = model_data['tfidf_vectorizer']
        self.tfidf_feature_names = model_data['tfidf_feature_names']
        self.ner_feature_names = model_data.get('ner_feature_names', [])
        self.max_tfidf_features = model_data.get('max_tfidf_features', 50)

        # Reload spaCy model
        spacy_model_name = model_data.get('spacy_model_name', 'en_core_web_sm')
        if spacy_model_name:
            try:
                self._load_spacy_model()
                print(f"Reloaded spaCy model: {spacy_model_name}")
            except Exception as e:
                print(f"Warning: Could not reload spaCy model {spacy_model_name}: {e}")
                self.nlp_model = None

        self.model_trained = True
        print(f"NER-enhanced model loaded from {filepath}")

def get_available_models(model_type: str = "all") -> pd.DataFrame:
    """
    Get information about available saved models

    Args:
        model_type: "baseline", "enhanced", "ner", or "all"

    Returns:
        DataFrame with model information
    """
    import glob
    import os
    from datetime import datetime

    model_dir = './_data/models/'
    if not os.path.exists(model_dir):
        return pd.DataFrame()

    # Define model patterns
    patterns = {
        'baseline': 'baseline_rf_model*.joblib',
        'enhanced': 'enhanced_rf_model*.joblib',
        'ner': 'ner_rf_model*.joblib'
    }

    if model_type == "all":
        search_patterns = patterns.values()
    else:
        search_patterns = [patterns.get(model_type, '*.joblib')]

    models_info = []

    for pattern in search_patterns:
        model_files = glob.glob(os.path.join(model_dir, pattern))

        for model_path in model_files:
            try:
                # Load model metadata without full model
                model_data = joblib.load(model_path)

                # Extract key information
                training_history = model_data.get('training_history', {})

                # Determine model type from filename or content
                filename = os.path.basename(model_path)
                if 'baseline' in filename:
                    detected_type = 'Baseline'
                elif 'enhanced' in filename:
                    detected_type = 'Enhanced'
                elif 'ner' in filename:
                    detected_type = 'NER'
                else:
                    detected_type = training_history.get('model_type', 'Unknown')

                model_info = {
                    'Filename': filename,
                    'Type': detected_type,
                    'File Size (MB)': round(os.path.getsize(model_path) / (1024*1024), 2),
                    'Modified': datetime.fromtimestamp(os.path.getmtime(model_path)).strftime('%Y-%m-%d %H:%M'),
                    'Test Accuracy': f"{training_history.get('test_accuracy', 0):.3f}",
                    'CV Accuracy': f"{training_history.get('cv_mean', 0):.3f}",
                    'Features': training_history.get('n_features', 0),
                    'Samples': training_history.get('n_samples', 0),
                    'Classes': len(training_history.get('class_names', [])),
                    'Full Path': model_path
                }

                # Add type-specific info
                if detected_type == 'Enhanced':
                    model_info['TF-IDF Features'] = training_history.get('n_tfidf_features', 0)
                elif detected_type == 'NER':
                    model_info['TF-IDF Features'] = training_history.get('n_tfidf_features', 0)
                    model_info['NER Features'] = training_history.get('n_ner_features', 0)

                models_info.append(model_info)

            except Exception as e:
                # If model can't be loaded, still show basic file info
                models_info.append({
                    'Filename': os.path.basename(model_path),
                    'Type': 'Unknown (Error)',
                    'File Size (MB)': round(os.path.getsize(model_path) / (1024*1024), 2),
                    'Modified': datetime.fromtimestamp(os.path.getmtime(model_path)).strftime('%Y-%m-%d %H:%M'),
                    'Error': str(e)[:50] + "..." if len(str(e)) > 50 else str(e),
                    'Full Path': model_path
                })

    if not models_info:
        return pd.DataFrame({'Message': ['No saved models found']})

    return pd.DataFrame(models_info)

def create_model_performance_summary(training_history: Dict) -> pd.DataFrame:
    """Create a summary of model performance metrics"""

    # Extract classification report
    class_report = training_history['classification_report']

    # Create summary for each class
    summary_data = []
    for class_name in training_history['class_names']:
        if class_name in class_report:
            summary_data.append({
                'Class': class_name,
                'Precision': class_report[class_name]['precision'],
                'Recall': class_report[class_name]['recall'],
                'F1-Score': class_report[class_name]['f1-score'],
                'Support': class_report[class_name]['support']
            })

    # Add overall metrics
    summary_data.append({
        'Class': 'Overall',
        'Precision': class_report['weighted avg']['precision'],
        'Recall': class_report['weighted avg']['recall'],
        'F1-Score': class_report['weighted avg']['f1-score'],
        'Support': class_report['weighted avg']['support']
    })

    return pd.DataFrame(summary_data)

def plot_feature_importance(feature_importance_df: pd.DataFrame, top_n: int = 15):
    """Plot feature importance"""

    plt.figure(figsize=(10, 8))
    top_features = feature_importance_df.head(top_n)

    sns.barplot(data=top_features, y='feature', x='importance', orient='h')
    plt.title(f'Top {top_n} Feature Importances - Baseline Model')
    plt.xlabel('Importance')
    plt.tight_layout()

    return plt.gcf()

def plot_confusion_matrix(confusion_matrix, class_names):
    """Plot confusion matrix"""

    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix - Baseline Model')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()

    return plt.gcf()

def plot_prediction_comparison(test_actual, test_predicted, real_predictions=None):
    """
    Plot comparison between test predictions and real data predictions

    Args:
        test_actual: Actual labels from test set
        test_predicted: Predicted labels from test set
        real_predictions: Predictions on real/open claims (optional)
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Test set actual distribution
    test_actual_counts = pd.Series(test_actual).value_counts()
    axes[0].pie(test_actual_counts.values, labels=test_actual_counts.index, autopct='%1.1f%%')
    axes[0].set_title('Test Set - Actual Status Distribution')

    # Test set predicted distribution
    test_pred_counts = pd.Series(test_predicted).value_counts()
    axes[1].pie(test_pred_counts.values, labels=test_pred_counts.index, autopct='%1.1f%%')
    axes[1].set_title('Test Set - Predicted Status Distribution')

    # Real predictions distribution (if provided)
    if real_predictions is not None:
        real_pred_counts = real_predictions.value_counts()
        axes[2].pie(real_pred_counts.values, labels=real_pred_counts.index, autopct='%1.1f%%')
        axes[2].set_title('Open Claims - Predicted Status Distribution')
    else:
        axes[2].text(0.5, 0.5, 'No open claims\npredictions available',
                    horizontalalignment='center', verticalalignment='center',
                    transform=axes[2].transAxes, fontsize=14)
        axes[2].set_title('Open Claims - Predicted Status Distribution')

    plt.tight_layout()
    return fig

def plot_prediction_confidence_analysis(predictions_df, status_col='predicted_status', confidence_col='prediction_confidence'):
    """
    Plot prediction confidence analysis

    Args:
        predictions_df: DataFrame with predictions and confidence scores
        status_col: Column name for predicted status
        confidence_col: Column name for prediction confidence
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Confidence distribution by predicted status
    for i, status in enumerate(predictions_df[status_col].unique()):
        status_data = predictions_df[predictions_df[status_col] == status]
        axes[0, 0].hist(status_data[confidence_col], alpha=0.6, label=status, bins=20)
    axes[0, 0].set_xlabel('Prediction Confidence')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].set_title('Confidence Distribution by Predicted Status')
    axes[0, 0].legend()

    # Box plot of confidence by status
    status_confidence = []
    status_labels = []
    for status in predictions_df[status_col].unique():
        status_data = predictions_df[predictions_df[status_col] == status]
        status_confidence.append(status_data[confidence_col].values)
        status_labels.append(status)

    axes[0, 1].boxplot(status_confidence, labels=status_labels)
    axes[0, 1].set_ylabel('Prediction Confidence')
    axes[0, 1].set_title('Confidence Distribution by Status (Box Plot)')
    axes[0, 1].tick_params(axis='x', rotation=45)

    # Confidence vs prediction count
    conf_bins = pd.cut(predictions_df[confidence_col], bins=10)
    conf_counts = conf_bins.value_counts().sort_index()
    axes[1, 0].bar(range(len(conf_counts)), conf_counts.values)
    axes[1, 0].set_xlabel('Confidence Bins')
    axes[1, 0].set_ylabel('Number of Predictions')
    axes[1, 0].set_title('Prediction Count by Confidence Level')

    # High confidence predictions summary
    high_conf = predictions_df[predictions_df[confidence_col] > 0.8]
    if len(high_conf) > 0:
        high_conf_counts = high_conf[status_col].value_counts()
        axes[1, 1].pie(high_conf_counts.values, labels=high_conf_counts.index, autopct='%1.1f%%')
        axes[1, 1].set_title(f'High Confidence Predictions (>80%)\nTotal: {len(high_conf)} claims')
    else:
        axes[1, 1].text(0.5, 0.5, 'No high confidence\npredictions (>80%)',
                       horizontalalignment='center', verticalalignment='center',
                       transform=axes[1, 1].transAxes, fontsize=12)
        axes[1, 1].set_title('High Confidence Predictions (>80%)')

    plt.tight_layout()
    return fig

def get_open_claims_for_prediction(claims_df):
    """
    Get open claims that should be predicted (including INITIAL_REVIEW)

    Args:
        claims_df: Claims DataFrame

    Returns:
        DataFrame with open claims
    """
    # Define open claim statuses (now including INITIAL_REVIEW)
    open_statuses = ['OPEN', 'ESTABLISHED', 'INITIAL_REVIEW', 'FUTURE_PAY_POTENTIAL']

    # Filter for open claims
    open_claims = claims_df[claims_df['clmStatus'].isin(open_statuses)]

    print(f"Found {len(open_claims)} open claims for prediction")
    print(f"Open status breakdown: {open_claims['clmStatus'].value_counts().to_dict()}")
    print("Note: INITIAL_REVIEW claims are now included in open claims prediction")

    return open_claims

def plot_enhanced_feature_importance(training_history: Dict, top_n: int = 20):
    """
    Plot feature importance for enhanced model with separate TF-IDF analysis

    Args:
        training_history: Enhanced model training history
        top_n: Number of top features to show
    """
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))

    # All features
    all_features = training_history['feature_importance'].head(top_n)
    sns.barplot(data=all_features, y='feature', x='importance', orient='h', ax=axes[0])
    axes[0].set_title(f'Top {top_n} All Features - Enhanced Model')
    axes[0].set_xlabel('Importance')

    # TF-IDF features only
    tfidf_features = training_history['tfidf_importance']
    if len(tfidf_features) > 0:
        # Clean TF-IDF feature names for display
        tfidf_display = tfidf_features.copy()
        tfidf_display['clean_feature'] = tfidf_display['feature'].str.replace('tfidf_', '')
        sns.barplot(data=tfidf_display, y='clean_feature', x='importance', orient='h', ax=axes[1])
        axes[1].set_title('Top TF-IDF Features')
        axes[1].set_xlabel('Importance')
    else:
        axes[1].text(0.5, 0.5, 'No TF-IDF features\nin top importance',
                    ha='center', va='center', transform=axes[1].transAxes)
        axes[1].set_title('Top TF-IDF Features')

    # Base features (non-TF-IDF)
    base_features = training_history['base_importance']
    if len(base_features) > 0:
        sns.barplot(data=base_features, y='feature', x='importance', orient='h', ax=axes[2])
        axes[2].set_title('Top Base Features (NLP + Financial)')
        axes[2].set_xlabel('Importance')
    else:
        axes[2].text(0.5, 0.5, 'No base features\nin top importance',
                    ha='center', va='center', transform=axes[2].transAxes)
        axes[2].set_title('Top Base Features')

    plt.tight_layout()
    return fig

def plot_model_comparison(baseline_history: Dict, enhanced_history: Dict):
    """
    Compare baseline vs enhanced model performance

    Args:
        baseline_history: Baseline model training history
        enhanced_history: Enhanced model training history
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Accuracy comparison
    models = ['Baseline', 'Enhanced']
    train_acc = [baseline_history['train_accuracy'], enhanced_history['train_accuracy']]
    test_acc = [baseline_history['test_accuracy'], enhanced_history['test_accuracy']]
    cv_acc = [baseline_history['cv_mean'], enhanced_history['cv_mean']]

    x = np.arange(len(models))
    width = 0.25

    axes[0, 0].bar(x - width, train_acc, width, label='Train', alpha=0.8)
    axes[0, 0].bar(x, test_acc, width, label='Test', alpha=0.8)
    axes[0, 0].bar(x + width, cv_acc, width, label='CV', alpha=0.8)
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].set_title('Model Accuracy Comparison')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(models)
    axes[0, 0].legend()
    axes[0, 0].set_ylim(0, 1)

    # Feature count comparison
    baseline_features = baseline_history['n_features']
    enhanced_features = enhanced_history['n_features']
    tfidf_features = enhanced_history.get('n_tfidf_features', 0)
    base_features_enhanced = enhanced_features - tfidf_features

    feature_data = {
        'Baseline': [baseline_features, 0],
        'Enhanced': [base_features_enhanced, tfidf_features]
    }

    bottom = np.zeros(2)
    colors = ['skyblue', 'orange']
    labels = ['Base Features', 'TF-IDF Features']

    for i, (label, color) in enumerate(zip(labels, colors)):
        values = [feature_data['Baseline'][i], feature_data['Enhanced'][i]]
        axes[0, 1].bar(models, values, bottom=bottom, label=label, color=color, alpha=0.8)
        bottom += values

    axes[0, 1].set_ylabel('Number of Features')
    axes[0, 1].set_title('Feature Count Comparison')
    axes[0, 1].legend()

    # F1-Score comparison by class
    baseline_report = baseline_history['classification_report']
    enhanced_report = enhanced_history['classification_report']

    classes = [cls for cls in baseline_report.keys()
              if isinstance(baseline_report[cls], dict) and 'f1-score' in baseline_report[cls]]

    baseline_f1 = [baseline_report[cls]['f1-score'] for cls in classes]
    enhanced_f1 = [enhanced_report[cls]['f1-score'] for cls in classes]

    x = np.arange(len(classes))
    width = 0.35

    axes[1, 0].bar(x - width/2, baseline_f1, width, label='Baseline', alpha=0.8)
    axes[1, 0].bar(x + width/2, enhanced_f1, width, label='Enhanced', alpha=0.8)
    axes[1, 0].set_ylabel('F1-Score')
    axes[1, 0].set_title('F1-Score by Class Comparison')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(classes, rotation=45, ha='right')
    axes[1, 0].legend()

    # Improvement metrics
    improvement_data = {
        'Metric': ['Train Accuracy', 'Test Accuracy', 'CV Accuracy'],
        'Baseline': [baseline_history['train_accuracy'],
                    baseline_history['test_accuracy'],
                    baseline_history['cv_mean']],
        'Enhanced': [enhanced_history['train_accuracy'],
                    enhanced_history['test_accuracy'],
                    enhanced_history['cv_mean']]
    }

    improvement_df = pd.DataFrame(improvement_data)
    improvement_df['Improvement'] = improvement_df['Enhanced'] - improvement_df['Baseline']
    improvement_df['Improvement_Pct'] = (improvement_df['Improvement'] / improvement_df['Baseline']) * 100

    colors = ['green' if x > 0 else 'red' for x in improvement_df['Improvement_Pct']]
    axes[1, 1].bar(improvement_df['Metric'], improvement_df['Improvement_Pct'],
                   color=colors, alpha=0.7)
    axes[1, 1].set_ylabel('Improvement (%)')
    axes[1, 1].set_title('Performance Improvement (Enhanced vs Baseline)')
    axes[1, 1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
    axes[1, 1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    return fig

if __name__ == "__main__":
    # Example usage
    print("Baseline ML Model for Claim Status Prediction")
    print("This module provides Random Forest baseline classification.")