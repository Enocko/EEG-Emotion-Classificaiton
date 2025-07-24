import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

# Plot accuracy comparison
def plot_accuracy_bar(results):
    plt.figure(figsize=(8, 5))
    bars = plt.bar(results.keys(), results.values(), color=['blue', 'green', 'orange'])
    plt.title("Model Accuracy Comparison")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    # Add labels on bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.01, f"{yval:.2f}", ha='center', va='bottom')
    plt.show()

# Plot confusion matrix
def plot_confusion_matrix(model, X_test, y_test):
    cm = confusion_matrix(y_test, model.predict(X_test), labels=model.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix - Best Model")
    plt.show()

if __name__ == "__main__":
    # Load data and split
    df = pd.read_csv("data/emotions.csv")
    X = df.drop(columns=['label'])
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Load the best model
    model = joblib.load("models/model.pkl")

    # Load results from training (manually enter or pass via file)
    results = {
        "RandomForest": 0.9883,
        "SVM": 0.3841,
        "GradientBoosting": 0.9977
    }

    print("✅ Displaying Accuracy Comparison and Confusion Matrix...")
    plot_accuracy_bar(results)
    plot_confusion_matrix(model, X_test, y_test)
