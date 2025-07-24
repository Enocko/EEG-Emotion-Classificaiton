from src.load_data import load_dataset
from src.preprocess import preprocess
from src.train_models import train_and_select_model
from src.visualize import plot_accuracy_bar, plot_confusion_matrix
from sklearn.model_selection import train_test_split
import joblib

def run_pipeline():
    # Load and preprocess
    df = load_dataset('data/emotions.csv')
    X, y = preprocess(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train models, select best
    results, best_model_name = train_and_select_model(X_train, y_train, X_test, y_test)
    print("\n📊 Accuracy Summary:", results)
    print(f"✅ Best Model: {best_model_name}")

    # Visualize results
    plot_accuracy_bar(results)

    # Confusion Matrix for best model
    best_model = joblib.load('models/model.pkl')
    plot_confusion_matrix(best_model, X_test, y_test)

if __name__ == "__main__":
    run_pipeline()
