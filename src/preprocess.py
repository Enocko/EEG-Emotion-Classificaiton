import pandas as pd

def preprocess(df):
    if 'label' in df.columns:
        X = df.drop(columns=['label'])
        y = df['label']
    else:
        X = df
        y = None
    return X, y

if __name__ == "__main__":
    df = pd.read_csv("data/emotions.csv")
    X, y = preprocess(df)
    print("✅ Preprocessing Complete!")
    print(f"Features Shape: {X.shape}")
    if y is not None:
        print(f"Labels Shape: {y.shape}")
