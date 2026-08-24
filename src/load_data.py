import os
import pandas as pd
from sklearn.datasets import fetch_california_housing

def load_data():
    housing = fetch_california_housing(as_frame=True)
    df = housing.frame
    
    # Ensure the output directory exists
    os.makedirs("data/raw", exist_ok=True)
    
    df.to_csv("data/raw/california_housing.csv", index=False)
    return df

if __name__ == "__main__":
    df = load_data()
    print("Dataset shape:", df.shape)
    print(df.head())