import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tempfile

mlflow.set_tracking_uri('http://localhost:5000')

# Load data
USAhousing = pd.read_csv(r"C:\Users\SubhamSahu\Desktop\MlOps\mlops_venv\USA_Housing1.csv")
print("USAhousing shape", USAhousing.shape)

# Check for missing values
print("USAhousing.isnull().sum()")

# Features and target
X = USAhousing[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms', 'Avg. Area Number of Bedrooms', 'Area Population']]
y = USAhousing['Price']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocessing pipeline
pipeline = Pipeline([
    ('std_scalar', MinMaxScaler())
])

X_train = pipeline.fit_transform(X_train)
X_test = pipeline.transform(X_test)

# Define a function to log the model and metrics
def log_model(model, model_name, X_train, y_train, X_test, y_test):
    with mlflow.start_run():
        # Log the model's hyperparameters
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("n_features", X_train.shape[1])
        
        # Fit the model
        model.fit(X_train, y_train)

        # Predict and calculate metrics
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Log metrics
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)

        print(f"{model_name} - MSE: {mse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")

        # Log the model parameters
        mlflow.log_param("model_parameters", model.get_params())

        # Provide an input example for signature (using the first test sample)
        example_input = X_test[0].reshape(1, -1)

        # Log the model with input example and signature
        mlflow.sklearn.log_model(model, model_name, input_example=example_input)

# Linear Regression Model
log_model(LinearRegression(), "Linear Regression", X_train, y_train, X_test, y_test)

# Decision Tree Model
log_model(DecisionTreeRegressor(), "Decision Tree", X_train, y_train, X_test, y_test)
