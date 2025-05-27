# -*- coding: utf-8 -*-
# AI Project Success Prediction Engine
# Complete ML system for analyzing project briefs and predicting success likelihood

import json
import pandas as pd
import numpy as np
import re
from typing import Dict, List, Tuple, Any
import warnings

warnings.filterwarnings('ignore')

# Core libraries
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
import joblib

# NLP libraries
import nltk
from textstat import flesch_reading_ease, flesch_kincaid_grade, syllable_count
from collections import Counter

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Download NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer


class ProjectBriefAnalyzer:
    """
    Core class for analyzing project briefs and extracting features
    """

    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.technical_terms = {
            'ai_ml': ['ai', 'ml', 'machine learning', 'deep learning', 'neural', 'algorithm', 'model', 'tensorflow',
                      'pytorch'],
            'cloud': ['aws', 'azure', 'cloud', 'kubernetes', 'docker', 'api', 'microservice'],
            'data': ['data', 'database', 'sql', 'analytics', 'etl', 'pipeline', 'warehouse'],
            'web': ['web', 'frontend', 'backend', 'react', 'javascript', 'html', 'css'],
            'mobile': ['mobile', 'ios', 'android', 'app', 'flutter', 'react native'],
            'security': ['security', 'encryption', 'auth', 'oauth', 'ssl', 'vulnerability'],
            'business': ['roi', 'revenue', 'cost', 'profit', 'budget', 'stakeholder', 'kpi']
        }

    def preprocess_text(self, text: str) -> str:
        """Clean and preprocess text"""
        # Convert to lowercase
        text = text.lower()
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s\.\,\!\?]', ' ', text)
        # Remove extra whitespace
        text = ' '.join(text.split())
        return text

    def calculate_clarity_score(self, text: str) -> Dict[str, float]:
        """Calculate various clarity metrics"""
        sentences = sent_tokenize(text)
        words = word_tokenize(text)

        # Basic metrics
        avg_sentence_length = len(words) / len(sentences) if sentences else 0
        unique_words = len(set(words))
        total_words = len(words)
        lexical_diversity = unique_words / total_words if total_words > 0 else 0

        # Readability scores
        flesch_score = flesch_reading_ease(text)
        fk_grade = flesch_kincaid_grade(text)

        # Grammar errors (simplified)
        grammar_errors = self.estimate_grammar_errors(text)

        # Normalize scores to 0-1 range
        clarity_metrics = {
            'avg_sentence_length': min(avg_sentence_length / 20, 1.0),
            'lexical_diversity': lexical_diversity,
            'flesch_score': max(0, flesch_score) / 100,
            'fk_grade': max(0, min(20 - fk_grade, 20)) / 20,
            'grammar_score': max(0, 1 - grammar_errors / 10)
        }

        # Weighted average for final clarity score
        weights = [0.2, 0.2, 0.3, 0.2, 0.1]
        clarity_score = sum(score * weight for score, weight in zip(clarity_metrics.values(), weights))

        return {
            'clarity_score': clarity_score,
            **clarity_metrics
        }

    def estimate_grammar_errors(self, text: str) -> int:
        """Estimate grammar errors using simple heuristics"""
        errors = 0

        # Check for common issues
        sentences = sent_tokenize(text)
        for sentence in sentences:
            # Check if sentence starts with lowercase
            if sentence and sentence[0].islower() and sentence[0].isalpha():
                errors += 0.5

            # Check for repeated words
            words = sentence.split()
            for i in range(len(words) - 1):
                if words[i] == words[i + 1]:
                    errors += 1

        return int(errors)

    def calculate_completeness_score(self, text: str) -> Dict[str, float]:
        """Calculate completeness based on presence of key sections"""
        text_lower = text.lower()

        # Key sections to look for
        required_sections = {
            'problem_definition': ['brief', 'problem', 'challenge', 'issue', 'need'],
            'solution_approach': ['solution', 'approach', 'what needs', 'how to', 'method'],
            'success_criteria': ['success', 'criteria', 'metric', 'kpi', 'measure', 'goal'],
            'scope': ['scope', 'in scope', 'out of scope', 'include', 'exclude'],
            'technical_details': ['tech', 'technology', 'stack', 'framework', 'tool', 'platform'],
            'opportunity': ['opportunity', 'market', 'business', 'value', 'benefit']
        }

        section_scores = {}
        for section, keywords in required_sections.items():
            score = 0
            for keyword in keywords:
                if keyword in text_lower:
                    score += 1
            section_scores[section] = min(score / len(keywords), 1.0)

        # Calculate overall completeness
        completeness_score = sum(section_scores.values()) / len(section_scores)

        return {
            'completeness_score': completeness_score,
            **section_scores
        }

    def extract_technical_features(self, text: str) -> Dict[str, float]:
        """Extract technical domain features"""
        text_lower = text.lower()

        domain_scores = {}
        for domain, terms in self.technical_terms.items():
            score = sum(1 for term in terms if term in text_lower)
            domain_scores[f'{domain}_mentions'] = score

        # Calculate technical complexity
        total_technical_terms = sum(domain_scores.values())
        technical_density = total_technical_terms / len(text.split()) if text.split() else 0

        return {
            'technical_density': technical_density,
            'total_technical_terms': total_technical_terms,
            **domain_scores
        }

    def extract_all_features(self, text: str) -> Dict[str, float]:
        """Extract all features for a project brief"""
        preprocessed_text = self.preprocess_text(text)

        # Input validation for nonsense text
        words = preprocessed_text.split()
        unique_words = set(words)

        # Check for meaningful content
        meaningful_words = ['project', 'system', 'build', 'create', 'develop', 'implement',
                            'problem', 'solution', 'need', 'goal', 'success', 'scope', 'in', 'out',
                            'brief', 'what', 'needs', 'happen', 'opportunity', 'hypothesis']
        has_meaningful_content = any(word in preprocessed_text.lower() for word in meaningful_words)

        # Quality penalty for nonsense text
        content_quality_penalty = 1.0
        if not has_meaningful_content or len(unique_words) < 5 or len(words) < 10:
            content_quality_penalty = 0.1  # Severe penalty for nonsense
            print(f"⚠️ Applying content quality penalty: {content_quality_penalty}")

        # Get all feature sets
        clarity_features = self.calculate_clarity_score(preprocessed_text)
        completeness_features = self.calculate_completeness_score(preprocessed_text)
        technical_features = self.extract_technical_features(preprocessed_text)

        # Apply quality penalty to completeness features (most important)
        completeness_features = {k: v * content_quality_penalty for k, v in completeness_features.items()}

        # Basic text statistics
        basic_features = {
            'word_count': len(words),
            'sentence_count': len(sent_tokenize(preprocessed_text)),
            'char_count': len(preprocessed_text),
            'avg_word_length': np.mean([len(word) for word in words]) if words else 0
        }

        # Combine all features
        all_features = {
            **clarity_features,
            **completeness_features,
            **technical_features,
            **basic_features
        }

        return all_features


class ProjectSuccessPredictor:
    """
    ML model for predicting project success
    """

    def __init__(self):
        self.analyzer = ProjectBriefAnalyzer()
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None

    def prepare_data(self, data: List[Dict]) -> Tuple[pd.DataFrame, np.ndarray]:
        """Prepare training data from project briefs"""
        features_list = []
        labels = []

        for item in data:
            # Extract structured features
            features = self.analyzer.extract_all_features(item['text'])
            features_list.append(features)

            # Create success label (1 if both clarity and feasibility are 1, else 0)
            success_label = 1 if (item['clarity'] == 1 and item['feasibility'] == 1) else 0
            labels.append(success_label)

        # Convert to DataFrame
        features_df = pd.DataFrame(features_list)

        # Handle any missing values
        features_df = features_df.fillna(0)

        # Store feature names
        self.feature_names = features_df.columns.tolist()

        return features_df, np.array(labels)

    def train_model(self, data: List[Dict], test_size: float = 0.2) -> Dict[str, Any]:
        """Train the prediction model"""
        # Prepare data
        X, y = self.prepare_data(data)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Train multiple models and select best
        models = {
            'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
            'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000)
        }

        best_model = None
        best_score = 0
        results = {}

        for name, model in models.items():
            model.fit(X_train_scaled, y_train)
            score = model.score(X_test_scaled, y_test)
            results[name] = {
                'accuracy': score,
                'model': model,
                'predictions': model.predict(X_test_scaled),
                'probabilities': model.predict_proba(X_test_scaled) if hasattr(model, 'predict_proba') else None
            }

            if score > best_score:
                best_score = score
                best_model = model

        self.model = best_model

        # Generate detailed results
        y_pred = self.model.predict(X_test_scaled)

        training_results = {
            'accuracy': accuracy_score(y_test, y_pred),
            'classification_report': classification_report(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'feature_importance': self.get_feature_importance(),
            'model_comparison': results,
            'test_predictions': y_pred,
            'test_labels': y_test
        }

        return training_results

    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from trained model"""
        if self.model is None:
            return {}

        if hasattr(self.model, 'feature_importances_'):
            importance = self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            importance = np.abs(self.model.coef_[0])
        else:
            return {}

        feature_importance = dict(zip(self.feature_names, importance))
        # Sort by importance
        return dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))

    def predict_success(self, project_text: str) -> Dict[str, Any]:
        """Predict success probability for a new project brief"""
        if self.model is None:
            raise ValueError("Model not trained yet!")

        # Extract features
        features = self.analyzer.extract_all_features(project_text)

        # DEBUG: Print all feature values
        print("\n🔍 DEBUG: Feature Analysis")
        print("-" * 40)
        for feature_name, value in features.items():
            print(f"{feature_name:<25}: {value:.4f}")

        # Convert to DataFrame with correct column order
        features_df = pd.DataFrame([features])
        features_df = features_df.reindex(columns=self.feature_names, fill_value=0)

        # DEBUG: Print scaled features
        features_scaled = self.scaler.transform(features_df)
        print(f"\n📊 Scaled features (first 5): {features_scaled[0][:5]}")

        # Make prediction
        prediction = self.model.predict(features_scaled)[0]
        if hasattr(self.model, 'predict_proba'):
            probability = self.model.predict_proba(features_scaled)[0]
            success_probability = probability[1]  # Probability of success (class 1)
            print(f"🎯 Raw model probabilities: {probability}")
            print(f"🎯 Success probability: {success_probability:.4f}")
        else:
            success_probability = prediction

        # Get feature importance for recommendations
        feature_importance = self.get_feature_importance()

        # Generate recommendations
        recommendations = self.generate_recommendations(features, feature_importance)

        # MANUAL OVERRIDE LOGIC
        critical_features = ['completeness_score', 'success_criteria', 'scope', 'problem_definition']
        critical_score = sum(features.get(feat, 0) for feat in critical_features) / len(critical_features)

        # If critical features are too low, override the ML prediction
        if critical_score < 0.1:  # All critical features basically zero
            success_probability = min(success_probability, 0.15)  # Cap at 15%
            print(f"🚨 Business logic override: capped probability at {success_probability:.3f}")

            # After getting ML prediction
            if critical_score < 0.05:  # Essentially no content
                success_probability = min(success_probability, 0.10)  # Max 10%
            elif critical_score < 0.2:  # Very poor content
                success_probability = min(success_probability, 0.30)  # Max 30%
            elif critical_score < 0.4:  # Poor content
                success_probability = min(success_probability, 0.60)  # Max 60%

            if success_probability > 0.10:
                success_probability = success_probability + 0.15
            if success_probability > 0.8:
                success_probability = min(success_probability, 0.95)

            comprehensive_msg = "✅ Your brief is comprehensive. No major gaps detected!"
            if recommendations == [comprehensive_msg]:
                success_probability = min(success_probability, 0.7)

        return {
            'prediction': int(prediction),
            'success_probability': float(success_probability),
            'features': features,
            'recommendations': recommendations,
            'feature_importance': feature_importance
        }

    def generate_recommendations(self, features: Dict[str, float], feature_importance: Dict[str, float]) -> List[str]:
        """Generate actionable recommendations based on feature analysis"""
        recommendations = []

        # Calculate overall quality score
        critical_features = ['completeness_score', 'success_criteria', 'scope', 'problem_definition']
        critical_score = sum(features.get(feat, 0) for feat in critical_features) / len(critical_features)

        print(f"🔍 Critical features average: {critical_score:.3f}")

        # If ALL critical features are basically zero, flag as very poor
        if critical_score < 0.1:
            recommendations.append(
                "⚠️ CRITICAL: This brief lacks essential project elements - problem definition, success criteria, and scope")

        # Get top important features
        top_features = list(feature_importance.keys())[:5]

        # Analyze key areas for improvement
        if features.get('completeness_score', 0) < 0.3:
            recommendations.append(
                "Add comprehensive project details: problem definition, solution approach, success criteria")

        if features.get('success_criteria', 0) < 0.3:
            recommendations.append("Define specific, measurable success criteria and KPIs")

        if features.get('scope', 0) < 0.3:
            recommendations.append("Clearly define what is in-scope and out-of-scope for this project")

        if features.get('technical_density', 0) < 0.01:
            recommendations.append("Include specific technical implementation details and technology choices")

        if features.get('problem_definition', 0) < 0.3:
            recommendations.append("Provide clear problem statement with business context and impact")

        # Special case for nonsense text
        word_count = features.get('word_count', 0)
        clarity_score = features.get('clarity_score', 0)
        if word_count > 0 and clarity_score < 0.3 and critical_score < 0.1:
            recommendations.insert(0, "🚨 This appears to be invalid text. Please provide a proper project brief.")

        if not recommendations:
            recommendations.append("✅ Your brief is comprehensive. No major gaps detected!")

        return recommendations[:3]  # Return top 3 recommendations

    def save_model(self, filepath: str):
        """Save trained model and components"""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            # Don't save the analyzer object, recreate it on load
        }
        joblib.dump(model_data, filepath)

    def load_model(self, filepath: str):
        """Load trained model and components"""
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        # Recreate analyzer instead of loading it
        self.analyzer = ProjectBriefAnalyzer()


def load_project_data(data_string: str) -> List[Dict]:
    """Load project brief data from string"""
    lines = data_string.strip().split('\n')
    data = []
    for line in lines:
        if line.strip():
            try:
                item = json.loads(line)
                data.append(item)
            except json.JSONDecodeError:
                continue
    return data


def create_visualizations(training_results: Dict, predictor: ProjectSuccessPredictor) -> None:
    """Create visualization plots"""

    # Feature importance plot
    feature_importance = training_results['feature_importance']
    top_features = dict(list(feature_importance.items())[:10])

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # 1. Feature Importance
    features = list(top_features.keys())
    importance = list(top_features.values())

    axes[0, 0].barh(features, importance)
    axes[0, 0].set_title('Top 10 Feature Importance')
    axes[0, 0].set_xlabel('Importance Score')

    # 2. Confusion Matrix
    cm = training_results['confusion_matrix']
    sns.heatmap(cm, annot=True, fmt='d', ax=axes[0, 1],
                xticklabels=['Failure', 'Success'],
                yticklabels=['Failure', 'Success'])
    axes[0, 1].set_title('Confusion Matrix')
    axes[0, 1].set_ylabel('Actual')
    axes[0, 1].set_xlabel('Predicted')

    # 3. Model Comparison
    model_names = list(training_results['model_comparison'].keys())
    accuracies = [training_results['model_comparison'][name]['accuracy'] for name in model_names]

    axes[1, 0].bar(model_names, accuracies)
    axes[1, 0].set_title('Model Performance Comparison')
    axes[1, 0].set_ylabel('Accuracy')
    axes[1, 0].set_ylim(0, 1)

    # 4. Prediction Distribution
    y_pred = training_results['test_predictions']
    y_test = training_results['test_labels']

    prediction_counts = pd.Series(y_pred).value_counts().sort_index()
    actual_counts = pd.Series(y_test).value_counts().sort_index()

    x = ['Failure', 'Success']
    width = 0.35
    x_pos = np.arange(len(x))

    axes[1, 1].bar(x_pos - width / 2, actual_counts.values, width, label='Actual', alpha=0.7)
    axes[1, 1].bar(x_pos + width / 2, prediction_counts.values, width, label='Predicted', alpha=0.7)
    axes[1, 1].set_title('Actual vs Predicted Distribution')
    axes[1, 1].set_xticks(x_pos)
    axes[1, 1].set_xticklabels(x)
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig('model_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()


# Main execution function
def main():
    """Main function to demonstrate the complete system"""

    print("🚀 AI Project Success Prediction Engine")
    print("=" * 50)

    # Load data from external JSON file
    print("Loading project data from file...")
    try:
        with open('project_data.json', 'r', encoding='utf-8') as f:
            project_data = json.load(f)
        print(f"Loaded {len(project_data)} project briefs")
    except FileNotFoundError:
        print("❌ Error: project_data.json file not found!")
        print("Please create project_data.json with your project brief data.")
        return
    except json.JSONDecodeError as e:
        print(f"❌ Error: Invalid JSON format in project_data.json: {e}")
        return

    if len(project_data) == 0:
        print("❌ Error: No project briefs found in project_data.json")
        return

    # Initialize and train predictor
    print("\nInitializing predictor...")
    predictor = ProjectSuccessPredictor()

    print("Training model...")
    training_results = predictor.train_model(project_data)

    print(f"\nModel Training Results:")
    print(f"Accuracy: {training_results['accuracy']:.3f}")
    print("\nClassification Report:")
    print(training_results['classification_report'])

    # Display feature importance
    print("\nTop 10 Most Important Features:")
    feature_importance = training_results['feature_importance']
    for i, (feature, importance) in enumerate(list(feature_importance.items())[:10], 1):
        print(f"{i:2d}. {feature:<25} {importance:.4f}")

    # Save the trained model
    print("\nSaving trained model...")
    predictor.save_model('project_success_model.pkl')

    # Create visualizations
    print("\nCreating visualizations...")
    create_visualizations(training_results, predictor)

    # Test prediction on a sample brief
    print("\nTesting prediction on sample brief...")
    sample_brief = """
    In Brief
    Build a machine learning system to predict customer churn using historical data.

    What Needs to Happen
    - Collect customer interaction data from CRM
    - Train predictive models using scikit-learn
    - Deploy API for real-time predictions
    - Create dashboard for business users

    Success Criteria
    - Model accuracy >= 85%
    - API response time < 200ms
    - 90% user adoption by sales team

    Scope
    - In: Python ML pipeline, REST API, React dashboard
    - Out: Real-time streaming, mobile app
    """

    prediction_result = predictor.predict_success(sample_brief)

    print(f"\nSample Brief Analysis:")
    print(f"Success Probability: {prediction_result['success_probability']:.3f}")
    print(f"Prediction: {'SUCCESS' if prediction_result['prediction'] == 1 else 'NEEDS IMPROVEMENT'}")
    print("\nTop Recommendations:")
    for i, rec in enumerate(prediction_result['recommendations'], 1):
        print(f"{i}. {rec}")

    print("\n" + "=" * 50)
    print("🎉 System Setup Complete!")
    print("\nNext Steps:")
    print("1. Run: streamlit run streamlit_dashboard.py")
    print("2. Open your browser to analyze project briefs")
    print("=" * 50)


if __name__ == "__main__":
    main()
