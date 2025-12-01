import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.datasets import load_iris # Using a built-in dataset for guaranteed runnability

# Set plotting style for better visualization
sns.set_style("whitegrid")

def load_and_preprocess_data():
    """Loads the Iris dataset, separates features and target, and performs scaling."""
    print("Loading and preparing data...")
    
    # Load the built-in Iris dataset (Replace this with your pd.read_csv('your_data.csv') when needed)
    iris = load_iris(as_frame=True)
    df = iris.frame
    X = df.drop(columns=['target'])
    y = df['target']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Initialize and fit the StandardScaler (imported in your notebook)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert scaled arrays back to DataFrames (optional, but good for tracking)
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)

    print(f"Data split: Training samples={len(X_train)}, Testing samples={len(X_test)}")
    return X_train_scaled, X_test_scaled, y_train, y_test

def train_and_evaluate_model(X_train, X_test, y_train, y_test):
    """Initializes, trains, and evaluates a Decision Tree Classifier."""
    print("\nTraining Decision Tree Classifier...")
    
    # Initialize the model
    dt_classifier = DecisionTreeClassifier(max_depth=4, random_state=42)
    
    # Train the model
    dt_classifier.fit(X_train, y_train)
    
    # Make predictions
    y_pred = dt_classifier.predict(X_test)
    
    # --- Evaluation ---
    
    # 1. Accuracy Score
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nModel Accuracy: {accuracy:.4f}")
    
    # 2. Classification Report
    print("\n--- Classification Report ---")
    print(classification_report(y_test, y_pred))
    
    # 3. Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Plotting the Confusion Matrix (using seaborn and matplotlib imports)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=load_iris().target_names, yticklabels=load_iris().target_names)
    plt.title('Confusion Matrix for Decision Tree')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()
    
    # 4. Cross-Validation Score
    # Note: Cross-validation is often done on the full dataset before final split/training
    cv_scores = cross_val_score(dt_classifier, 
                                pd.concat([X_train, X_test]), 
                                pd.concat([y_train, y_test]), 
                                cv=5)
    print(f"\n5-Fold Cross-Validation Accuracy Scores: {cv_scores}")
    print(f"Mean CV Accuracy: {cv_scores.mean():.4f}")

def main():
    """Main function to run the ML workflow."""
    
    # 1. Load and Preprocess Data
    X_train, X_test, y_train, y_test = load_and_preprocess_data()
    
    # 2. Train and Evaluate Model
    train_and_evaluate_model(X_train, X_test, y_train, y_test)

if __name__ == "__main__":
    main()
