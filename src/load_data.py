from sklearn.datasets import fetch_california_housing
import pandas as pd

def load_data():
    housing = fetch_california_housing(as_frame=True)
    df = housing.frame  # pandas DataFrame including target column 'MedHouseVal'
    df.to_csv("data/raw/california_housing.csv", index=False)
    return df

if __name__ == "__main__":
    df = load_data()
    print("Dataset shape:", df.shape)
    print(df.head())