import pandas as pd

def clean_data(filepath):
    df = pd.read_csv(filepath)

    # Drop rows with missing prices or nonsensical entries
    df = df[df['price'] > 0]
    df = df[df['minimum_nights'] < 365]

    # Encode categorical features
    df = pd.get_dummies(df, columns=['neighborhood', 'room_type'], drop_first=True)

    return df
