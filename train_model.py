import pandas as pd
import numpy as np
import kagglehub
import os
import pickle

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Download dataset
path = kagglehub.dataset_download("nehalbirla/vehicle-dataset-from-cardekho")

data = pd.read_csv(os.path.join(path,"car data.csv"))

# Feature engineering
current_year = 2024
data["CarAge"] = current_year - data["Year"]

data.drop(["Car_Name","Year"], axis=1, inplace=True)

# Convert categorical variables
data = pd.get_dummies(data, drop_first=True)

# Features and target
X = data.drop("Selling_Price", axis=1)
y = data["Selling_Price"]

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestRegressor(n_estimators=200)
model.fit(X_train, y_train)

# Save model
pickle.dump(model, open("car_price_model.pkl","wb"))

print("Model trained and saved successfully!")
