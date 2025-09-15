import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
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

        Args:
            nlp_features_df: DataFrame with NLP features
            claims_df: DataFrame with claim information including status

        Returns:
            DataFrame ready for training
        """
        # Get available financial columns
        financial_cols = ['clmNum', 'clmStatus', 'clmCause']
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

        # Filter to eligible statuses only
        eligible_statuses = ['PAID', 'DENIED', 'CLOSED', 'INITIAL_REVIEW']
        merged_df = merged_df[merged_df['clmStatus'].isin(eligible_statuses)]

        print(f"Prepared dataset with {len(merged_df)} samples")
        print(f"Status distribution:\n{merged_df['clmStatus'].value_counts()}")

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
        y = df['clmStatus']

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
            'class_names': self.label_encoder.classes_
        }

        print(f"Training completed!")
        print(f"Train Accuracy: {train_accuracy:.3f}")
        print(f"Test Accuracy: {test_accuracy:.3f}")
        print(f"CV Accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")

        return self.training_history

    def predict(self, nlp_features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Make predictions on new data.

        Args:
            nlp_features_df: NLP features for prediction

        Returns:
            DataFrame with predictions and probabilities
        """
        if not self.model_trained:
            raise ValueError("Model must be trained before making predictions")

        # Prepare features
        X = nlp_features_df[self.feature_columns].fillna(0)
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

if __name__ == "__main__":
    # Example usage
    print("Baseline ML Model for Claim Status Prediction")
    print("This module provides Random Forest baseline classification.")