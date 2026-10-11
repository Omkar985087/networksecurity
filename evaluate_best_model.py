"""
evaluate_best_model.py
======================
Comprehensive evaluation of candidate models and the best-performing model
for Network Security / Phishing Website Detection.

Complies strictly with requirements:
1. Inspects existing architecture, models, and preprocessing conventions.
2. Uses reproducible stratified train-test split (80/20, random_state=42, stratify=Result).
3. Selects best model using test-independent cross-validation F1-score on the training split.
4. Calculates Accuracy, Precision, Recall, F1-score, Confusion Matrix, Classification Report, and ROC-AUC.
   Explicitly defines target encoding and positive class (Phishing).
5. Prevents data leakage (preprocessor fitted only on train split, Result removed from features).
6. Generates evaluation outputs:
   - evaluation_results/best_model_metrics.txt
   - evaluation_results/model_comparison.csv
   - evaluation_results/confusion_matrix.png
7. Prints a clear terminal report formatted with decimals and percentages.
8. Preserves production models without unintentional overwriting.
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve
)

# Project paths
DATA_PATH = os.path.join("Network_data", "phisingData.csv")
FINAL_MODEL_PATH = os.path.join("final_model", "model.pkl")
FINAL_PREPROCESSOR_PATH = os.path.join("final_model", "preprocessor.pkl")
OUTPUT_DIR = "evaluation_results"


def load_and_split_data(data_path: str):
    """
    Loads labeled phishing dataset and creates a reproducible stratified train-test split.
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}")

    df = pd.read_csv(data_path)
    
    # Feature columns and target column
    target_col = "Result"
    X = df.drop(columns=[target_col])
    y_raw = df[target_col]

    # Target encoding:
    # Original UCI Phishing dataset: -1 = Phishing, 1 = Legitimate
    # Codebase convention (data_transformation.py): -1 -> 0, 1 -> 1
    # 0 = Phishing, 1 = Legitimate
    y = y_raw.replace(-1, 0)

    # Stratified train-test split: 80% train, 20% test, random_state=42
    X_train, X_test, y_train, y_test, y_train_raw, y_test_raw = train_test_split(
        X, y, y_raw, test_size=0.20, random_state=42, stratify=y
    )

    return df, X_train, X_test, y_train, y_test, y_train_raw, y_test_raw


def fit_preprocessor(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """
    Fits KNNImputer only on training features to prevent data leakage,
    matching project's DATA_TRANSFORMATION_IMPUTER_PARAMS (n_neighbors=3).
    """
    imputer = KNNImputer(missing_values=np.nan, n_neighbors=3, weights="uniform")
    preprocessor = Pipeline([("imputer", imputer)])
    
    preprocessor.fit(X_train)
    X_train_trans = preprocessor.transform(X_train)
    X_test_trans = preprocessor.transform(X_test)
    
    return preprocessor, X_train_trans, X_test_trans


def run_model_selection(X_train_trans: np.ndarray, y_train: pd.Series):
    """
    Evaluates candidate models using cross-validation on the training split only.
    Selects best model by test-independent cross-validation F1-score.
    """
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    # Candidate models and parameter grids from codebase conventions
    candidate_configs = {
        "Random Forest": {
            "estimator": RandomForestClassifier(random_state=42),
            "params": {"n_estimators": [16, 32, 64, 128, 256]}
        },
        "Gradient Boosting": {
            "estimator": GradientBoostingClassifier(random_state=42),
            "params": {
                "n_estimators": [64, 128, 256],
                "learning_rate": [0.05, 0.1, 0.2],
                "subsample": [0.8, 0.9, 1.0]
            }
        },
        "Decision Tree": {
            "estimator": DecisionTreeClassifier(random_state=42),
            "params": {"criterion": ["gini", "entropy", "log_loss"]}
        },
        "Logistic Regression": {
            "estimator": LogisticRegression(max_iter=1000, random_state=42),
            "params": {"C": [0.1, 1.0, 10.0]}
        },
        "AdaBoost": {
            "estimator": AdaBoostClassifier(random_state=42),
            "params": {
                "n_estimators": [32, 64, 128, 256],
                "learning_rate": [0.1, 0.5, 1.0]
            }
        }
    }

    results = {}
    fitted_models = {}

    print(">>> Running test-independent cross-validation model selection on training split...")
    for model_name, cfg in candidate_configs.items():
        print(f"    Evaluating {model_name}...")
        # Use macro F1 for cross-validation scoring to balance phishing and legitimate classes
        gs = GridSearchCV(
            cfg["estimator"],
            cfg["params"],
            cv=cv,
            scoring="f1_macro",
            n_jobs=-1
        )
        gs.fit(X_train_trans, y_train)

        best_estimator = gs.best_estimator_
        # Best model is fitted on full training split
        best_estimator.fit(X_train_trans, y_train)

        # Cross-validation scores for both classes
        # Phishing (pos_label=0) and Legitimate (pos_label=1)
        # Using cv_results_ of the best index
        best_idx = gs.best_index_
        mean_cv_macro_f1 = gs.cv_results_["mean_test_score"][best_idx]
        std_cv_macro_f1 = gs.cv_results_["std_test_score"][best_idx]

        results[model_name] = {
            "best_params": gs.best_params_,
            "cv_f1_macro": mean_cv_macro_f1,
            "cv_f1_macro_std": std_cv_macro_f1,
            "model_obj": best_estimator
        }
        fitted_models[model_name] = best_estimator
        print(f"    -> {model_name} Best Params: {gs.best_params_} | Mean CV Macro F1: {mean_cv_macro_f1:.4f} (+/- {std_cv_macro_f1:.4f})")

    # Select best model based on training CV macro F1 score
    best_model_name = max(results, key=lambda k: results[k]["cv_f1_macro"])
    best_model_obj = results[best_model_name]["model_obj"]

    return results, fitted_models, best_model_name, best_model_obj


def evaluate_model_on_test(model, X_test_trans: np.ndarray, y_test: pd.Series):
    """
    Computes comprehensive evaluation metrics on held-out test split.
    """
    y_pred = model.predict(X_test_trans)
    
    # Probability estimates for ROC-AUC
    y_proba = None
    roc_auc = None
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test_trans)[:, 1]
        roc_auc = roc_auc_score(y_test, y_proba)
    elif hasattr(model, "decision_function"):
        y_scores = model.decision_function(X_test_trans)
        roc_auc = roc_auc_score(y_test, y_scores)

    # Metrics with Phishing (class 0) as Positive Class
    acc = accuracy_score(y_test, y_pred)
    prec_phish = precision_score(y_test, y_pred, pos_label=0)
    rec_phish = recall_score(y_test, y_pred, pos_label=0)
    f1_phish = f1_score(y_test, y_pred, pos_label=0)

    # Metrics with Legitimate (class 1) as Positive Class
    prec_legit = precision_score(y_test, y_pred, pos_label=1)
    rec_legit = recall_score(y_test, y_pred, pos_label=1)
    f1_legit = f1_score(y_test, y_pred, pos_label=1)

    # Summary averages
    f1_macro = f1_score(y_test, y_pred, average="macro")
    f1_weighted = f1_score(y_test, y_pred, average="weighted")
    prec_macro = precision_score(y_test, y_pred, average="macro")
    rec_macro = recall_score(y_test, y_pred, average="macro")

    # Confusion matrix: rows = true, cols = pred
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    # TN, FP, FN, TP when Class 0 (Phishing) is Positive:
    # True Class 0 & Pred 0 -> TP (Phishing correctly detected)
    # True Class 0 & Pred 1 -> FN (Phishing missed)
    # True Class 1 & Pred 0 -> FP (Legitimate flagged as phishing)
    # True Class 1 & Pred 1 -> TN (Legitimate passed)
    tp_phish = cm[0, 0]
    fn_phish = cm[0, 1]
    fp_phish = cm[1, 0]
    tn_phish = cm[1, 1]

    clf_report = classification_report(
        y_test,
        y_pred,
        labels=[0, 1],
        target_names=["Phishing (0)", "Legitimate (1)"],
        digits=4
    )

    metrics_dict = {
        "accuracy": acc,
        "phishing_precision": prec_phish,
        "phishing_recall": rec_phish,
        "phishing_f1": f1_phish,
        "legitimate_precision": prec_legit,
        "legitimate_recall": rec_legit,
        "legitimate_f1": f1_legit,
        "macro_precision": prec_macro,
        "macro_recall": rec_macro,
        "macro_f1": f1_macro,
        "weighted_f1": f1_weighted,
        "roc_auc": roc_auc,
        "confusion_matrix": cm,
        "tp_phishing": tp_phish,
        "fn_phishing": fn_phish,
        "fp_phishing": fp_phish,
        "tn_phishing": tn_phish,
        "classification_report": clf_report,
        "y_pred": y_pred,
        "y_proba": y_proba
    }

    return metrics_dict


def save_confusion_matrix_plot(cm: np.ndarray, model_name: str, output_path: str):
    """
    Renders and saves a high-quality annotated confusion matrix plot.
    """
    plt.figure(figsize=(7, 6))
    labels = ["Phishing (0 / -1)", "Legitimate (1)"]
    
    # Custom annotations showing counts and cell meaning
    annot = np.array([
        [f"{cm[0, 0]}\n(True Phishing)", f"{cm[0, 1]}\n(False Legitimate)"],
        [f"{cm[1, 0]}\n(False Phishing)", f"{cm[1, 1]}\n(True Legitimate)"]
    ])

    sns.heatmap(
        cm,
        annot=annot,
        fmt="",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        cbar=True,
        linewidths=1.5,
        linecolor="white",
        annot_kws={"size": 12, "weight": "bold"}
    )
    plt.title(f"Confusion Matrix: {model_name} (Held-out Test Set)", fontsize=13, weight="bold", pad=15)
    plt.xlabel("Predicted Label", fontsize=11, weight="bold")
    plt.ylabel("Actual Label", fontsize=11, weight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved confusion matrix plot: {output_path}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 80)
    print("NETWORK SECURITY PHISHING DETECTION - BEST MODEL EVALUATION")
    print("=" * 80)

    # Step 1: Load and split dataset
    df, X_train, X_test, y_train, y_test, y_train_raw, y_test_raw = load_and_split_data(DATA_PATH)
    total_records = len(df)
    n_train = len(X_train)
    n_test = len(X_test)
    n_features = X_train.shape[1]

    print(f"\n[1] Dataset Summary:")
    print(f"    - Original Labeled Data: {DATA_PATH}")
    print(f"    - Total Records: {total_records}")
    print(f"    - Feature Count: {n_features}")
    print(f"    - Target Column: 'Result'")
    print(f"    - Target Distribution (Original): -1 (Phishing)={sum(y_train_raw == -1) + sum(y_test_raw == -1)}, 1 (Legitimate)={sum(y_train_raw == 1) + sum(y_test_raw == 1)}")
    print(f"    - Training Split (80%): {n_train} records (Phishing={sum(y_train == 0)}, Legitimate={sum(y_train == 1)})")
    print(f"    - Held-out Test Split (20%): {n_test} records (Phishing={sum(y_test == 0)}, Legitimate={sum(y_test == 1)})")
    print(f"    - Target Label Mapping:")
    print(f"        * Original: -1  ==>  Model Encoded: 0  ==>  Meaning: Phishing (Positive Class of Interest)")
    print(f"        * Original: +1  ==>  Model Encoded: 1  ==>  Meaning: Legitimate Website")

    # Step 2: Fit preprocessor strictly on training split
    preprocessor, X_train_trans, X_test_trans = fit_preprocessor(X_train, X_test)
    print(f"\n[2] Preprocessing Pipeline:")
    print(f"    - Preprocessor fitted strictly on training split (no leakage).")
    print(f"    - Preprocessing steps: KNNImputer(n_neighbors=3, weights='uniform')")

    # Step 3: Run model selection using CV on training split
    results, fitted_models, best_model_name, best_model_obj = run_model_selection(X_train_trans, y_train)

    print(f"\n[3] Model Selection Result:")
    print(f"    - Selected Best Model: {best_model_name}")
    print(f"    - Best Hyperparameters: {results[best_model_name]['best_params']}")
    print(f"    - Training Cross-Validation Macro F1-score: {results[best_model_name]['cv_f1_macro']:.4f} (+/- {results[best_model_name]['cv_f1_macro_std']:.4f})")

    # Step 4: Evaluate candidate models and best model on held-out test split
    print(f"\n[4] Evaluating Candidate Models on Held-out Test Split (once)...")
    comparison_records = []
    test_metrics_by_model = {}

    for model_name, model_obj in fitted_models.items():
        metrics = evaluate_model_on_test(model_obj, X_test_trans, y_test)
        test_metrics_by_model[model_name] = metrics

        comparison_records.append({
            "Model Name": model_name,
            "CV F1 (Macro, Train)": round(results[model_name]["cv_f1_macro"], 4),
            "Test Accuracy": round(metrics["accuracy"], 4),
            "Test F1 (Phishing: 0)": round(metrics["phishing_f1"], 4),
            "Test Precision (Phishing: 0)": round(metrics["phishing_precision"], 4),
            "Test Recall (Phishing: 0)": round(metrics["phishing_recall"], 4),
            "Test F1 (Legitimate: 1)": round(metrics["legitimate_f1"], 4),
            "Test Precision (Legitimate: 1)": round(metrics["legitimate_precision"], 4),
            "Test Recall (Legitimate: 1)": round(metrics["legitimate_recall"], 4),
            "Test F1 (Macro)": round(metrics["macro_f1"], 4),
            "Test ROC-AUC": round(metrics["roc_auc"], 4) if metrics["roc_auc"] is not None else "N/A",
            "Best Parameters": str(results[model_name]["best_params"])
        })

    comparison_df = pd.DataFrame(comparison_records).sort_values(
        by="CV F1 (Macro, Train)", ascending=False
    )

    # Save model comparison CSV
    model_comparison_file = os.path.join(OUTPUT_DIR, "model_comparison.csv")
    comparison_df.to_csv(model_comparison_file, index=False)
    print(f"    -> Saved model comparison table to: {model_comparison_file}")

    # Best model metrics
    best_metrics = test_metrics_by_model[best_model_name]

    # Step 5: Save Confusion Matrix Plot
    cm_plot_file = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
    save_confusion_matrix_plot(best_metrics["confusion_matrix"], best_model_name, cm_plot_file)

    # Step 6: Inspect existing saved production model (final_model/model.pkl)
    existing_model_metrics = None
    if os.path.exists(FINAL_MODEL_PATH):
        try:
            with open(FINAL_MODEL_PATH, "rb") as f:
                saved_prod_model = pickle.load(f)
            existing_model_metrics = evaluate_model_on_test(saved_prod_model, X_test_trans, y_test)
        except Exception as e:
            print(f"Notice: Could not evaluate existing final_model/model.pkl: {e}")

    # Step 7: Write Detailed Evaluation Report
    metrics_report_file = os.path.join(OUTPUT_DIR, "best_model_metrics.txt")
    with open(metrics_report_file, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("NETWORK SECURITY / PHISHING DETECTION - BEST MODEL EVALUATION REPORT\n")
        f.write("=" * 80 + "\n\n")

        f.write("1. DATASET & SPLITTING CONFIGURATION\n")
        f.write("-" * 50 + "\n")
        f.write(f"Source Dataset: {DATA_PATH}\n")
        f.write(f"Total Records: {total_records}\n")
        f.write(f"Number of Features: {n_features}\n")
        f.write(f"Splitting Strategy: Stratified Train-Test Split (test_size=0.20, random_state=42)\n")
        f.write(f"Training Split Count: {n_train} records (80.00%)\n")
        f.write(f"Held-out Test Split Count: {n_test} records (20.00%)\n\n")

        f.write("2. TARGET LABEL DEFINITION & ENCODING\n")
        f.write("-" * 50 + "\n")
        f.write("Original Dataset Target: 'Result' in {-1, 1}\n")
        f.write("Codebase Encoded Target: 'Result' in {0, 1} (via replace(-1, 0))\n")
        f.write("Mapping Table:\n")
        f.write("  - Original: -1  ==>  Model Encoded: 0  ==>  Class: Phishing Website (Positive Class)\n")
        f.write("  - Original: +1  ==>  Model Encoded: 1  ==>  Class: Legitimate Website\n\n")
        f.write(f"Class Distribution in Held-out Test Set (Total: {n_test}):\n")
        f.write(f"  - Phishing (0 / -1): {sum(y_test == 0)} records ({sum(y_test == 0) / n_test * 100:.2f}%)\n")
        f.write(f"  - Legitimate (1):    {sum(y_test == 1)} records ({sum(y_test == 1) / n_test * 100:.2f}%)\n\n")

        f.write("3. SELECTED BEST MODEL ARCHITECTURE & HYPERPARAMETERS\n")
        f.write("-" * 50 + "\n")
        f.write(f"Best Model Name: {best_model_name}\n")
        f.write(f"Model Class: {best_model_obj.__class__.__name__}\n")
        f.write(f"Model Selection Criterion: Test-independent 3-Fold Stratified Cross-Validation Macro F1\n")
        f.write(f"Best Hyperparameters: {results[best_model_name]['best_params']}\n")
        f.write(f"CV Macro F1 on Training Split: {results[best_model_name]['cv_f1_macro']:.4f} "
                f"({results[best_model_name]['cv_f1_macro'] * 100:.2f}%) +/- {results[best_model_name]['cv_f1_macro_std']:.4f}\n\n")

        f.write("4. FINAL EVALUATION METRICS ON HELD-OUT TEST SPLIT (N = 2,211)\n")
        f.write("-" * 50 + "\n")
        f.write(f"Overall Accuracy:                  {best_metrics['accuracy']:.4f}  ({best_metrics['accuracy'] * 100:.2f}%)\n")
        if best_metrics['roc_auc'] is not None:
            f.write(f"ROC-AUC Score:                     {best_metrics['roc_auc']:.4f}  ({best_metrics['roc_auc'] * 100:.2f}%)\n\n")

        f.write("A. Metrics with POSITIVE CLASS = PHISHING (0 / -1):\n")
        f.write(f"  - Precision (Phishing):          {best_metrics['phishing_precision']:.4f}  ({best_metrics['phishing_precision'] * 100:.2f}%)\n")
        f.write(f"  - Recall (Phishing):             {best_metrics['phishing_recall']:.4f}  ({best_metrics['phishing_recall'] * 100:.2f}%)\n")
        f.write(f"  - F1-score (Phishing):           {best_metrics['phishing_f1']:.4f}  ({best_metrics['phishing_f1'] * 100:.2f}%)\n\n")

        f.write("B. Metrics with POSITIVE CLASS = LEGITIMATE (1):\n")
        f.write(f"  - Precision (Legitimate):        {best_metrics['legitimate_precision']:.4f}  ({best_metrics['legitimate_precision'] * 100:.2f}%)\n")
        f.write(f"  - Recall (Legitimate):           {best_metrics['legitimate_recall']:.4f}  ({best_metrics['legitimate_recall'] * 100:.2f}%)\n")
        f.write(f"  - F1-score (Legitimate):         {best_metrics['legitimate_f1']:.4f}  ({best_metrics['legitimate_f1'] * 100:.2f}%)\n\n")

        f.write("C. Summary Averages:\n")
        f.write(f"  - Macro Precision:               {best_metrics['macro_precision']:.4f}  ({best_metrics['macro_precision'] * 100:.2f}%)\n")
        f.write(f"  - Macro Recall:                  {best_metrics['macro_recall']:.4f}  ({best_metrics['macro_recall'] * 100:.2f}%)\n")
        f.write(f"  - Macro F1-score:                {best_metrics['macro_f1']:.4f}  ({best_metrics['macro_f1'] * 100:.2f}%)\n")
        f.write(f"  - Weighted F1-score:             {best_metrics['weighted_f1']:.4f}  ({best_metrics['weighted_f1'] * 100:.2f}%)\n\n")

        f.write("5. CONFUSION MATRIX (HELD-OUT TEST SET)\n")
        f.write("-" * 50 + "\n")
        cm = best_metrics["confusion_matrix"]
        f.write(f"Confusion Matrix Array (Rows=True, Columns=Predicted):\n")
        f.write(f"                 Pred: Phishing(0)   Pred: Legit(1)\n")
        f.write(f"True: Phishing(0)       {cm[0, 0]:<15}      {cm[0, 1]:<15}\n")
        f.write(f"True: Legit(1)          {cm[1, 0]:<15}      {cm[1, 1]:<15}\n\n")
        f.write(f"Detailed Counts (Phishing = Positive Event):\n")
        f.write(f"  - True Positives (TP, Phishing correctly detected): {best_metrics['tp_phishing']}\n")
        f.write(f"  - False Negatives (FN, Phishing missed / false pass): {best_metrics['fn_phishing']}\n")
        f.write(f"  - False Positives (FP, Legitimate falsely blocked):   {best_metrics['fp_phishing']}\n")
        f.write(f"  - True Negatives (TN, Legitimate correctly passed):  {best_metrics['tn_phishing']}\n\n")

        f.write("6. SKLEARN CLASSIFICATION REPORT\n")
        f.write("-" * 50 + "\n")
        f.write(best_metrics["classification_report"] + "\n\n")

        f.write("7. CANDIDATE MODELS COMPARISON\n")
        f.write("-" * 50 + "\n")
        f.write(comparison_df.to_string(index=False) + "\n\n")

        if existing_model_metrics:
            f.write("8. EXISTING PRODUCTION MODEL (final_model/model.pkl) VALIDATION\n")
            f.write("-" * 50 + "\n")
            f.write(f"Model Type: {saved_prod_model.__class__.__name__}\n")
            f.write(f"Held-out Test Accuracy:          {existing_model_metrics['accuracy']:.4f} ({existing_model_metrics['accuracy'] * 100:.2f}%)\n")
            f.write(f"Held-out Test Phishing F1:        {existing_model_metrics['phishing_f1']:.4f} ({existing_model_metrics['phishing_f1'] * 100:.2f}%)\n")
            f.write(f"Held-out Test Legitimate F1:      {existing_model_metrics['legitimate_f1']:.4f} ({existing_model_metrics['legitimate_f1'] * 100:.2f}%)\n")
            f.write(f"Held-out Test Macro F1:           {existing_model_metrics['macro_f1']:.4f} ({existing_model_metrics['macro_f1'] * 100:.2f}%)\n")
            f.write(f"Held-out Test ROC-AUC:            {existing_model_metrics['roc_auc']:.4f} ({existing_model_metrics['roc_auc'] * 100:.2f}%)\n\n")

        f.write("9. DATA LEAKAGE PREVENTION & INTEGRITY NOTES\n")
        f.write("-" * 50 + "\n")
        f.write("a. Preprocessing Isolation: The KNNImputer preprocessor was fitted exclusively on the 80% training set.\n")
        f.write("   No imputation statistics or patterns from the 20% test set leaked into model fitting.\n")
        f.write("b. Feature Isolation: The 'Result' target column was strictly removed prior to feature transformations.\n")
        f.write("c. Test Set Independence: The test split was kept completely untouched during hyperparameter tuning and model selection.\n")
        f.write("d. Unlabeled Data Handling: The 12-row Valid_data/test.csv lacks ground truth labels and was strictly excluded\n")
        f.write("   from evaluation to avoid guesswork and invalid benchmark scores.\n")
        f.write("e. Production Preservation: Existing files final_model/model.pkl and final_model/preprocessor.pkl were NOT\n")
        f.write("   overwritten, ensuring production deployment and FastAPI serving remain fully intact and operational.\n")

    print(f"Saved full metrics report to: {metrics_report_file}")

    # Step 8: Terminal Report
    print("\n" + "=" * 80)
    print("TERMINAL EVALUATION REPORT: BEST MODEL")
    print("=" * 80)
    print(f"Selected Best Model: {best_model_name}")
    print(f"Number of Training Records: {n_train:,} records (80.00%)")
    print(f"Number of Test Records:     {n_test:,} records (20.00%)")
    print("\nTarget Label Mapping:")
    print("  - Phishing (Positive Event): Encoded as 0  (Original Dataset label: -1)")
    print("  - Legitimate Website:        Encoded as 1  (Original Dataset label: +1)")
    print("\nKey Performance Metrics on Held-Out Test Set (N = 2,211):")
    print(f"  * Accuracy:                     {best_metrics['accuracy']:.4f}  ({best_metrics['accuracy'] * 100:.2f}%)")
    print(f"  * Precision (Phishing - pos=0): {best_metrics['phishing_precision']:.4f}  ({best_metrics['phishing_precision'] * 100:.2f}%)")
    print(f"  * Recall (Phishing - pos=0):    {best_metrics['phishing_recall']:.4f}  ({best_metrics['phishing_recall'] * 100:.2f}%)")
    print(f"  * F1-Score (Phishing - pos=0):  {best_metrics['phishing_f1']:.4f}  ({best_metrics['phishing_f1'] * 100:.2f}%)")
    print(f"  * Precision (Legitimate - pos=1): {best_metrics['legitimate_precision']:.4f}  ({best_metrics['legitimate_precision'] * 100:.2f}%)")
    print(f"  * Recall (Legitimate - pos=1):    {best_metrics['legitimate_recall']:.4f}  ({best_metrics['legitimate_recall'] * 100:.2f}%)")
    print(f"  * F1-Score (Legitimate - pos=1):  {best_metrics['legitimate_f1']:.4f}  ({best_metrics['legitimate_f1'] * 100:.2f}%)")
    print(f"  * Macro F1-Score:               {best_metrics['macro_f1']:.4f}  ({best_metrics['macro_f1'] * 100:.2f}%)")
    if best_metrics["roc_auc"] is not None:
        print(f"  * ROC-AUC:                      {best_metrics['roc_auc']:.4f}  ({best_metrics['roc_auc'] * 100:.2f}%)")

    print("\nConfusion Matrix (Held-out Test Split):")
    print(f"                 Pred: Phishing(0)   Pred: Legit(1)")
    print(f"True: Phishing(0)       {cm[0, 0]:<15}      {cm[0, 1]:<15}")
    print(f"True: Legit(1)          {cm[1, 0]:<15}      {cm[1, 1]:<15}")
    print(f"  -> True Positives (TP - Phishing detected): {best_metrics['tp_phishing']}")
    print(f"  -> False Negatives (FN - Phishing missed):   {best_metrics['fn_phishing']}")
    print(f"  -> False Positives (FP - Legit false alarm): {best_metrics['fp_phishing']}")
    print(f"  -> True Negatives (TN - Legit passed):       {best_metrics['tn_phishing']}")

    print("\nScikit-learn Classification Report:")
    print(best_metrics["classification_report"])

    print("\nGenerated Output Files:")
    print(f"  1. Metrics Report:      {os.path.abspath(metrics_report_file)}")
    print(f"  2. Model Comparison:    {os.path.abspath(model_comparison_file)}")
    print(f"  3. Confusion Matrix:    {os.path.abspath(cm_plot_file)}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
