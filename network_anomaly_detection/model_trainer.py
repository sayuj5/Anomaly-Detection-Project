import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import seaborn as sns

# --- Model Training Functions ---

def train_supervised_rf(X_train, y_train, random_state=42):
    """Trains a Random Forest Classifier."""
    print("[Supervised Training] Starting Random Forest training...")
    # Using a simple setup for the Random Forest
    rf_model = RandomForestClassifier(n_estimators=100, random_state=random_state, n_jobs=-1, class_weight='balanced')
    rf_model.fit(X_train, y_train)
    print("[Supervised Training] Random Forest training complete.")
    return rf_model

def train_deep_learning_mlp(X_train, y_train, epochs, batch_size, random_state=42):
    """Trains a Keras Multi-Layer Perceptron (MLP) model."""
    
    # Model Architecture
    input_dim = X_train.shape[1]
    mlp_model = Sequential([
        Dense(64, activation='relu', input_shape=(input_dim,)),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid') # Binary classification output
    ])

    # Compile the model
    mlp_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model
    mlp_model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
    
    print("[Supervised Training] Keras MLP training complete.")
    return mlp_model

def train_unsupervised_iforest(X_train, contamination, random_state=42):
    """Trains an Unsupervised Isolation Forest model."""
    # Isolation Forest is fitted only on the features (X_train)
    iforest_model = IsolationForest(
        contamination=contamination, 
        random_state=random_state, 
        n_jobs=-1
    )
    # The fit method in Isolation Forest returns the model itself
    iforest_model.fit(X_train)
    print("[Unsupervised Training] Isolation Forest training complete.")
    return iforest_model

# --- Evaluation Function ---

def evaluate_model(model, X_test, y_test, model_name, is_unsupervised=False):
    """
    Evaluates the model and prints a classification report and confusion matrix.
    
    Includes robust prediction logic and explicit label passing to classification_report.
    """
    print(f"\n--- Evaluating {model_name} ---")

    # Define the labels for classification report and confusion matrix
    LABELS = [0, 1]
    TARGET_NAMES = ['Normal (0)', 'Anomaly (1)']

    if is_unsupervised:
        # For unsupervised models (Isolation Forest)
        y_pred = model.predict(X_test)
        # Convert IForest output: 1 (Normal) -> 0, -1 (Anomaly) -> 1
        y_pred_binary = np.where(y_pred == 1, 0, 1)
        y_prob = None # No standard probability/AUC calculation for IForest

    else:
        # For supervised models (Random Forest, MLP)
        y_pred_raw = model.predict(X_test)
        
        if hasattr(model, 'predict_proba'):
            # Random Forest and other Scikit-learn classifiers with predict_proba
            probabilities = model.predict_proba(X_test)
            
            # Robustly get probability for class 1
            if probabilities.shape[1] > 1:
                y_prob = probabilities[:, 1]
            else:
                y_prob = probabilities[:, 0]
                
            y_pred_binary = y_pred_raw
            
        elif hasattr(model, 'predict') and model_name == 'Keras MLP':
            # Keras model with sigmoid output returns a single column of probabilities for class 1
            y_prob = y_pred_raw.flatten()
            # Convert continuous probability to binary (threshold at 0.5)
            y_pred_binary = (y_prob > 0.5).astype(int)
        else:
            y_prob = None
            y_pred_binary = y_pred_raw

    # Print Classification Report
    print("Classification Report:")
    # FIX: Explicitly pass 'labels' to classification_report to force it to show both classes (0 and 1), 
    # even if one class is missing in y_test or y_pred_binary.
    print(classification_report(y_test, y_pred_binary, 
                                target_names=TARGET_NAMES, 
                                labels=LABELS))

    # Print Confusion Matrix
    cm = confusion_matrix(y_test, y_pred_binary, labels=LABELS) # Also explicitly set labels here
    plt.figure(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=TARGET_NAMES, 
                yticklabels=TARGET_NAMES)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # Calculate and print AUC if probabilities are available
    if not is_unsupervised and y_prob is not None:
        try:
            # Check if both classes are present in the test labels for AUC calculation
            if len(np.unique(y_test)) > 1:
                auc = roc_auc_score(y_test, y_prob)
                print(f"AUC Score: {auc:.4f}")
            else:
                print("Warning: Only one class present in y_test. Skipping AUC calculation.")
        except ValueError as e:
            print(f"Warning: Could not calculate AUC score. {e}")


def plot_feature_importance(model, feature_names, n_top=15):
    """
    Plots the feature importance from a tree-based model (e.g., Random Forest).
    """
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        # Create a pandas Series for easy sorting and naming
        feature_importance_series = pd.Series(importances, index=feature_names)
        
        # Sort values and select top N features
        top_features = feature_importance_series.sort_values(ascending=False).head(n_top)
        
        # Plotting
        plt.figure(figsize=(10, 6))
        sns.barplot(x=top_features.values, y=top_features.index, palette="viridis")
        plt.title(f'Top {n_top} Feature Importances - Random Forest')
        plt.xlabel('Importance Score')
        plt.ylabel('Feature')
        plt.tight_layout()
    else:
        print("Feature importance plot skipped: Model does not have 'feature_importances_'.")
