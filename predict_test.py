
import os
import pickle
import pandas as pd

from networksecurity.utils.ml_utils.model.estimator import NetworkModel

TEST_FILE = "Valid_data/test.csv"
MODEL_FILE = "final_model/model.pkl"
PREPROCESSOR_FILE = "final_model/preprocessor.pkl"
OUTPUT_FILE = "prediction_output/test_predictions.csv"


def load_pickle(path):
    with open(path, "rb") as file:
        return pickle.load(file)


def main():
    # Load test data
    X_test = pd.read_csv(TEST_FILE)
    print("Test data shape:", X_test.shape)

    # Remove target column if present
    X_test = X_test.drop(columns=["Result"], errors="ignore")

    # Load saved preprocessing pipeline and model
    preprocessor = load_pickle(PREPROCESSOR_FILE)
    model = load_pickle(MODEL_FILE)

    # Generate predictions
    network_model = NetworkModel(preprocessor, model)
    predictions = network_model.predict(X_test)

    # Save predictions
    os.makedirs("prediction_output", exist_ok=True)
    results = X_test.copy()
    results["predicted_column"] = predictions
    results.to_csv(OUTPUT_FILE, index=False)

    print("\nPredictions generated successfully!")
    print("Saved to:", OUTPUT_FILE)
    print("\nPrediction counts:")
    print(results["predicted_column"].value_counts())
    print("\nPredictions for all 12 records:")
    print(results[["predicted_column"]].to_string(index=True))


if __name__ == "__main__":
    main()
