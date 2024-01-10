import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib
import os

def generate_dummy_data(num_samples=1000):
    """
    Generates dummy transaction data for fraud detection.
    """
    np.random.seed(42)
    data = {
        'transaction_amount': np.random.rand(num_samples) * 1000 + 10,
        'age': np.random.randint(18, 80, num_samples),
        'location_risk': np.random.rand(num_samples),
        'transaction_frequency': np.random.randint(1, 20, num_samples),
        'is_fraud': np.random.choice([0, 1], num_samples, p=[0.95, 0.05]) # 5% fraud
    }
    df = pd.DataFrame(data)
    # Introduce some correlation for fraud
    df.loc[df['is_fraud'] == 1, 'transaction_amount'] = np.random.rand(df['is_fraud'].sum()) * 2000 + 500
    df.loc[df['is_fraud'] == 1, 'location_risk'] = np.random.rand(df['is_fraud'].sum()) * 0.5 + 0.5
    return df

def train_fraud_model(data_path=None, model_output_path="models/fraud_detector.joblib"):
    """
    Trains a RandomForestClassifier for fraud detection.

    Args:
        data_path (str, optional): Path to the CSV data file. If None, dummy data is generated.
        model_output_path (str): Path to save the trained model.
    """
    if data_path:
        df = pd.read_csv(data_path)
    else:
        print("Generating dummy data for training...")
        df = generate_dummy_data()

    X = df.drop("is_fraud", axis=1)
    y = df["is_fraud"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print("Training RandomForestClassifier...")
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight=\"balanced\")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print("\n--- Model Evaluation ---")
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("ROC AUC Score:", roc_auc_score(y_test, y_proba))

    # Save the model
    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    joblib.dump(model, model_output_path)
    print(f"\nModel saved to {model_output_path}")

if __name__ == "__main__":
    # Example usage:
    # To train with dummy data:
    train_fraud_model()

    # To train with a specific CSV file (uncomment and provide path):
    # train_fraud_model(data_path="path/to/your/transactions.csv")

    # Create requirements.txt
    with open("requirements.txt", "w") as f:
        f.write("pandas\n")
        f.write("scikit-learn\n")
        f.write("numpy\n")
        f.write("joblib\n")
    print("requirements.txt created.")
