import pandas as pd

def load_dataset(path):
    df = pd.read_csv(path)
    return df

if __name__ == "__main__":
    # Example standalone execution
    df = load_dataset("data/emotions.csv")
    print("✅ Data Loaded Successfully!")
    print(df.head())
