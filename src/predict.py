import joblib
import pandas as pd

def make_prediction(csv_path, model_path='models/model.pkl'):
    df = pd.read_csv(csv_path)
    if 'label' in df.columns:
        df = df.drop(columns=['label'])

    model = joblib.load(model_path)
    predictions = model.predict(df)
    proba = model.predict_proba(df)

    for i, row in enumerate(predictions):
        print(f"Prediction: {row}, Confidence: {proba[i]}")