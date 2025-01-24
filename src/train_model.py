import argparse
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, 
                            confusion_matrix, 
                            accuracy_score,
                            roc_auc_score)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# Fixed feature order to match prediction requirements
FEATURE_ORDER = [
    'num_links',
    'suspicious_domain',
    'reply_to_mismatch',
    'num_urgent_keywords',
    'contains_html',
    'link_domain_mismatch',
    'text_entropy',
    'body_length'
]

MODELS = {
    'random_forest': RandomForestClassifier(
        class_weight='balanced',
        n_estimators=150,
        max_depth=10,
        random_state=42
    ),
    'logistic_regression': LogisticRegression(
        class_weight='balanced',
        max_iter=1000,
        random_state=42
    ),
    'svm': SVC(
        class_weight='balanced',
        probability=True,
        kernel='rbf',
        random_state=42
    )
}

def train_model(data_path, model_output, model_type='random_forest', test_size=0.2):
    """
    Train a phishing detection model and save it to disk
    """
    # Load and validate data
    df = pd.read_csv(data_path)
    assert 'label' in df.columns, "Dataset missing 'label' column"
    assert all(col in df.columns for col in FEATURE_ORDER), "Missing required features"
    
    # Prepare data
    X = df[FEATURE_ORDER]
    y = df['label']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=42
    )
    
    # Initialize and train model
    model = MODELS[model_type]
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    print(f"\n{'='*40}")
    print(f"{model_type.upper()} Performance")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"ROC AUC: {roc_auc_score(y_test, y_proba):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print('='*40)
    
    # Save model
    joblib.dump(model, model_output)
    print(f"\nModel saved to {model_output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a phishing email classifier')
    parser.add_argument('--data', required=True, 
                       help='Path to processed CSV file')
    parser.add_argument('--model_output', required=True,
                       help='Output path for trained model (.pkl)')
    parser.add_argument('--model_type', default='random_forest',
                       choices=MODELS.keys(),
                       help='Model type to train')
    parser.add_argument('--test_size', type=float, default=0.2,
                       help='Test set proportion (default: 0.2)')
    
    args = parser.parse_args()
    
    train_model(
        data_path=args.data,
        model_output=args.model_output,
        model_type=args.model_type,
        test_size=args.test_size
    )
