from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import joblib
import pandas as pd

def train_and_select_model(X_train, y_train, X_test, y_test, save_path='models/model.pkl'):
    models = {
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
        "SVM": SVC(probability=True, kernel='rbf', random_state=42),
        "GradientBoosting": GradientBoostingClassifier(n_estimators=100, random_state=42)
    }

    results = {}
    best_model_name = None
    best_model = None
    best_accuracy = 0.0

    for name, model in models.items():
        print(f"\n🔹 Training {name}...")
        model.fit(X_train, y_train)

        # Predictions
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        results[name] = acc

        # Print accuracy & classification report
        print(f"{name} Accuracy: {acc:.4f}")
        print(f"Classification Report for {name}:\n")
        print(classification_report(y_test, y_pred))

        # Update best model
        if acc > best_accuracy:
            best_accuracy = acc
            best_model_name = name
            best_model = model

    print("\n✅ Model Comparison Completed!")
    print(f"Best Model: {best_model_name} with Accuracy: {best_accuracy:.4f}")

    # Save best model
    joblib.dump(best_model, save_path)
    print(f"✅ Best model saved to {save_path}")

    return results, best_model_name

if __name__ == "__main__":
    df = pd.read_csv("data/emotions.csv")
    X = df.drop(columns=['label'])
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    results, best = train_and_select_model(X_train, y_train, X_test, y_test)
    print("\n📊 Accuracy Summary:", results)
