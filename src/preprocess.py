import pandas as pd
from pathlib import Path


INPUT_PATH = Path("data/dataset.csv")
OUTPUT_PATH = Path("data/processed/processed.csv")


def preprocess_data():

    print("Loading dataset...")

    df = pd.read_csv(INPUT_PATH)

    print(f"Original dataset shape: {df.shape}")
    print("Original columns:")
    print(df.columns.tolist())

    # Remove unnecessary Order ID
    if "Order_ID" in df.columns:
        df = df.drop(columns=["Order_ID"])

    # Target column
    target_column = "Delivery_Time_min"

    if target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' not found."
        )

    # Remove rows where target is missing
    df = df.dropna(subset=[target_column])

    # Identify categorical columns
    categorical_columns = df.select_dtypes(
        include=["object"]
    ).columns.tolist()

    print("Categorical columns:")
    print(categorical_columns)

    # One-hot encode categorical variables
    if categorical_columns:
        df = pd.get_dummies(
            df,
            columns=categorical_columns,
            drop_first=True
        )

    # Convert boolean columns to integers
    boolean_columns = df.select_dtypes(
        include=["bool"]
    ).columns

    for column in boolean_columns:
        df[column] = df[column].astype(int)

    # Fill numerical missing values
    numerical_columns = df.select_dtypes(
        include=["number"]
    ).columns

    for column in numerical_columns:

        if df[column].isnull().any():

            df[column] = df[column].fillna(
                df[column].median()
            )

    # Remove duplicate records
    before_duplicates = len(df)

    df = df.drop_duplicates()

    after_duplicates = len(df)

    print(
        f"Removed {before_duplicates - after_duplicates} "
        f"duplicate rows."
    )

    # Save processed dataset
    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print(
        f"Processed dataset shape: {df.shape}"
    )

    print(
        f"Processed dataset saved to: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    preprocess_data()