import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, 
    precision_recall_fscore_support, 
    classification_report, 
    confusion_matrix,
    ConfusionMatrixDisplay
)
from sklearn.model_selection import GridSearchCV, cross_val_score

def preprocess_dataframe(df):
    """
    Preprocess the input dataframe for complex word identification
    """
    # Validate column names
    required_cols = [
        "HIT_ID", "Sentence", "Start_Offset", "End_Offset", 
        "Target_Word", "Native_Annotators", "Non_Native_Annotators", 
        "Binary_Label"
    ]

    # Ensure only the required columns are present
    df = df[required_cols]

    # Convert to numeric
    numeric_cols = [
        "Start_Offset", "End_Offset", 
        "Native_Annotators", "Non_Native_Annotators", 
        "Binary_Label"
    ]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')

    # Basic text cleaning
    df['Sentence'] = df['Sentence'].str.strip()

    # Drop rows with invalid data
    df = df.dropna(subset=numeric_cols)

    return df

def extract_features(df):
    """
    Extract meaningful features for complex word identification
    """
    # Calculate annotator features
    df['total_annotators'] = df['Native_Annotators'] + df['Non_Native_Annotators']

    # Word length feature
    df['target_word_length'] = df['Target_Word'].str.len()

    # Positional features
    df['word_start_ratio'] = df['Start_Offset'] / df['Sentence'].str.len()

    # Annotator ratio feature
    df['annotator_ratio'] = df['Native_Annotators'] / df['total_annotators']

    return df

def prepare_data(df):
    """
    Prepare data for machine learning model
    """
    # Extract features
    df = extract_features(df)

    # Prepare feature matrix
    features = [
        'annotator_ratio', 
        'total_annotators',
        'target_word_length', 
        'word_start_ratio', 
        'Start_Offset', 
        'Native_Annotators', 
        'Non_Native_Annotators'
    ]

    X = df[features].values
    y = df['Binary_Label'].values

    return X, y

def plot_confusion_matrix(y_true, y_pred, title='Confusion Matrix'):
    """
    Plot and save confusion matrix
    """
    plt.figure(figsize=(8, 6))
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred)
    plt.title(title)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.close()

def perform_cross_validation(X, y, pipeline, param_grid):
    """
    Perform stratified cross-validation to get more robust performance estimate
    """
    # Use Stratified K-Fold for cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Perform grid search with cross-validation
    grid_search = GridSearchCV(
        pipeline, 
        param_grid, 
        cv=cv, 
        scoring='f1', 
        n_jobs=-1
    )

    # Fit the grid search
    grid_search.fit(X, y)

    # Perform cross-validation with best estimator
    best_model = grid_search.best_estimator_
    cv_scores = cross_val_score(best_model, X, y, cv=cv, scoring='f1')

    return grid_search, cv_scores

def main():
    # Load training and development datasets
    try:
        frames = []
        base_path = f'{os.getcwd()}/cwishareddataset/traindevset/'
        for folder in os.listdir(base_path):
            folder_path = os.path.join(base_path, folder)
            if os.path.isdir(folder_path):
                for file in os.listdir(folder_path):
                    file_path = os.path.join(folder_path, file)
                    frames.append(pd.read_csv(file_path, delimiter='\t', header=None))

        df = pd.concat(frames)
        df.columns = [
            "HIT_ID", "Sentence", "Start_Offset", "End_Offset", 
            "Target_Word", "Native_Annotators", "Non_Native_Annotators", 
            "Native_Marked", "Non_Native_Marked", "Binary_Label", "Prob_Label"
        ]
    except Exception as e:
        print(f"Error loading training datasets: {e}")
        return

    # Preprocess data
    df = preprocess_dataframe(df)

    # Split data
    X, y = prepare_data(df)

    # Create a pipeline with scaling and Random Forest
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(random_state=42))
    ])

    # Define parameter grid for GridSearchCV
    param_grid = {
        'classifier__n_estimators': [50, 100, 200],
        'classifier__max_depth': [None, 10, 20, 30],
        'classifier__min_samples_split': [2, 5, 10],
        'classifier__min_samples_leaf': [1, 2, 4]
    }

    # Perform cross-validation
    grid_search, cv_scores = perform_cross_validation(X, y, pipeline, param_grid)

    # Print cross-validation results
    print("Cross-Validation Scores:", cv_scores)
    print("Mean CV Score:", cv_scores.mean())
    print("Standard Deviation of CV Scores:", cv_scores.std())

    # Best model and parameters
    best_model = grid_search.best_estimator_
    print("\nBest Parameters:", grid_search.best_params_)

    # Feature importance
    feature_names = [
        'annotator_ratio', 
        'total_annotators',
        'target_word_length', 
        'word_start_ratio', 
        'Start_Offset', 
        'Native_Annotators', 
        'Non_Native_Annotators'
    ]
    importances = best_model.named_steps['classifier'].feature_importances_
    feature_importances = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)

    print("\nFeature Importances:")
    for name, importance in feature_importances:
        print(f"{name}: {importance}")

    # Load test datasets
    try:
        test_frames = []
        test_base_path = f'{os.getcwd()}/cwishareddataset/testset/'
        for folder in os.listdir(test_base_path):
            folder_path = os.path.join(test_base_path, folder)
            if os.path.isdir(folder_path):
                for file in os.listdir(folder_path):
                    file_path = os.path.join(folder_path, file)
                    test_frames.append(pd.read_csv(file_path, delimiter='\t', header=None))

        test_df = pd.concat(test_frames)
        test_df.columns = [
            "HIT_ID", "Sentence", "Start_Offset", "End_Offset", 
            "Target_Word", "Native_Annotators", "Non_Native_Annotators", 
            "Native_Marked", "Non_Native_Marked", "Binary_Label", "Prob_Label"
        ]
    except Exception as e:
        print(f"Error loading test datasets: {e}")
        return

    # Preprocess test data
    test_df = preprocess_dataframe(test_df)

    # Prepare test data
    test_features, test_labels = prepare_data(test_df)

    # Predict on test set
    test_predictions = best_model.predict(test_features)

    # Evaluate on test set
    print("\nTest Set Results:")
    print(classification_report(test_labels, test_predictions))

    # Plot confusion matrix for test set
    plot_confusion_matrix(test_labels, test_predictions, title='Test Set Confusion Matrix')

    # Save the model (optional)
    import joblib
    joblib.dump(best_model, 'complex_word_random_forest_model.pkl')
    print("\nModel saved to complex_word_random_forest_model.pkl")

if __name__ == '__main__':
    main()