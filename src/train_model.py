import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib

from data_cleaning import clean_data
from model_pipeline import build_pipeline

df = clean_data('data/nyc_rentals.csv')

X = df.drop(['id', 'price'], axis=1)
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

pipeline = build_pipeline()
pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)
print(f"Test RMSE: {mean_squared_error(y_test, y_pred, squared=False):.2f}")

joblib.dump(pipeline, 'model.pkl')
