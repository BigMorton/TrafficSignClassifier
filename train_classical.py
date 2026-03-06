import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

# Import stages 1 and 2
from data_pipeline import load_and_preprocess
from feature_extraction import extract_features

def evaluate_model(model, X_test, y_test, class_names, model_name):
    # Generates predictions, prints classification report, plots confusion matrix

    # Nice formatting to show model under evaluation
    print(f"\n{'='*40}")
    print(f"Evaluating: {model_name}")
    print(f"\n{'='*40}")

    # Model makes predictions on test data
    y_pred = model.predict(X_test)

    # Calculate overall accuracty
    acc = accuracy_score(y_test, y_pred)
    print(f"Overall Accuracy: {acc * 100:.2f}%\n")  # Print accurac as percentage to 2 dec places

    # Generate and print detailed report (precision, recall, F1-Score)
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names))

    # Generate and display confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)

    # Format plot for better visual understanding
    fig, ax, = plt.subplots(figsize=(8, 6))
    disp.plot(cmap=plt.cm.Blues, ax=ax, xticks_rotation=45)
    plt.title((f"Confusion Matrix: {model_name}"))
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Stage 1: Load and preprocess data
    print("Stage 1: Loading and preprocessing data...")
    X_train, X_test, y_train, y_test, class_names = load_and_preprocess()

    # Stage 2: Extract features
    print("\nStage 2: Extracting features...")
    X_train_feat = extract_features(X_train)
    X_test_feat = extract_features(X_test)

    # Stage 3: Train Naive Bayes 
    print("\nStage 3: Training Naive Bayes Classifier...")
    nb_model = GaussianNB()
    nb_model.fit(X_train_feat, y_train) # TRAINING HAPPENS HERE

    # Stage 4: Train Lindear Discriminant Classifier (LDC)
    print("\nStage 4: Training Linear Discriminant Classifier...")
    ldc_model = LinearDiscriminantAnalysis()
    ldc_model.fit(X_train_feat, y_train) # TRAINING LINEAR BOUNDARIES

    # Stage 5: Evaluation of Models
    evaluate_model(nb_model, X_test_feat, y_test, class_names, "Naive Bayes")
    evaluate_model(ldc_model, X_test_feat, y_test, class_names, "Linear Discriminant Classifier")