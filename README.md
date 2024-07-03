# Business-Strategy-Student-Performance-XBG-Model-Training-and-ML-Flow-Deployment

# Student Performance Prediction using MLflow and Ngrok

This project aims to predict the performance of students in the BADM 449 course using machine learning models. The project includes data preprocessing, model training, evaluation, and tracking using MLflow and Ngrok for remote access.

## Table of Contents

- [Introduction](#introduction)
- [Prerequisites](#prerequisites)
- [Setup](#setup)
- [Usage](#usage)
- [Results](#results)
- [Acknowledgements](#acknowledgements)

## Introduction

This project involves predicting student performance in the BADM 449 course using various machine learning models, including Random Forest, Gradient Boosting, and XGBoost. The project utilizes MLflow for model tracking and evaluation and Ngrok for creating a public URL to access the MLflow tracking server.

## Prerequisites

Before you begin, ensure you have met the following requirements:

- Python 3.6 or later
- Jupyter Notebook or Google Colab
- Ngrok account (for creating a public URL for MLflow)
- Libraries: `pandas`, `scikit-learn`, `mlflow`, `pyngrok`, `xgboost`

## Setup

1. **Clone the repository:**

    ```bash
    git clone https://github.com/yourusername/student-performance-prediction.git
    cd student-performance-prediction
    ```

2. **Install the required libraries:**

    ```bash
    pip install pandas scikit-learn mlflow pyngrok xgboost
    ```

3. **Set up Ngrok:**

    - Sign up at [Ngrok](https://ngrok.com/) and get your authtoken.
    - Set up Ngrok with your authtoken:

    ```python
    from pyngrok import ngrok
    ngrok.set_auth_token('your_ngrok_authtoken')
    ```

## Usage

### Running the Project in Google Colab

1. **Load the dataset:**

    - Upload the dataset (`Student Performance BADM 449.xlsx`) to your Google Colab environment.

2. **Set up Ngrok and start the MLflow server:**

    ```python
    from pyngrok import ngrok

    ngrok.set_auth_token('your_ngrok_authtoken')
    ngrok_tunnel = ngrok.connect(5000)
    print('Public URL:', ngrok_tunnel.public_url)

    get_ipython().system_raw("mlflow ui --port 5000 &")
    ```

3. **Run the following script to train and evaluate the models:**

    ```python
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from xgboost import XGBRegressor
    from sklearn.metrics import mean_squared_error, r2_score
    import mlflow
    import mlflow.sklearn
    import mlflow.xgboost

    # Set tracking URI to the Ngrok public URL
    mlflow.set_tracking_uri(ngrok_tunnel.public_url)

    # Load the dataset
    file_path = '/mnt/data/Student Performance BADM 449.xlsx'
    data = pd.read_excel(file_path)

    # Function to clean column names
    def clean_column_name(col_name):
        return col_name.strip().replace('#', '').replace(' ', '_').replace(':', '').replace('(', '').replace(')', '')

    # Apply the function to all column names
    data.columns = [clean_column_name(col) for col in data.columns]

    # Remove the second row
    data = data.drop(index=1)

    # Replace missing values with 0 for numeric columns
    for column in data.columns:
        if pd.api.types.is_numeric_dtype(data[column]):
            data[column].fillna(0, inplace=True)

    # Ensure 'Final_Score' exists in the cleaned columns
    target_variable = 'Final_Score'
    if target_variable not in data.columns:
        raise ValueError(f"{target_variable} not found in the columns")

    # Assuming 'Final_Score' is the target variable
    X = data.drop(columns=[target_variable])
    y = data[target_variable]

    # Identify categorical and numerical columns
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Preprocessing for numerical data
    numerical_transformer = SimpleImputer(strategy='constant', fill_value=0)

    # Preprocessing for categorical data
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Bundle preprocessing for numerical and categorical data
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])

    # Create a pipeline that includes the preprocessor and the model
    def create_model_pipeline(model):
        return Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('model', model)
        ])

    # Initialize models
    rf_model = RandomForestRegressor(random_state=42)
    gbm_model = GradientBoostingRegressor(random_state=42)
    xgb_model = XGBRegressor(random_state=42)

    # Create pipelines
    rf_pipeline = create_model_pipeline(rf_model)
    gbm_pipeline = create_model_pipeline(gbm_model)
    xgb_pipeline = create_model_pipeline(xgb_model)

    # Train the models
    rf_pipeline.fit(X_train, y_train)
    gbm_pipeline.fit(X_train, y_train)
    xgb_pipeline.fit(X_train, y_train)

    print("Model Training Complete")

    # Predict and evaluate models
    def evaluate_model(pipeline, X_test, y_test):
        predictions = pipeline.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        return mse, r2

    rf_mse, rf_r2 = evaluate_model(rf_pipeline, X_test, y_test)
    gbm_mse, gbm_r2 = evaluate_model(gbm_pipeline, X_test, y_test)
    xgb_mse, xgb_r2 = evaluate_model(xgb_pipeline, X_test, y_test)

    print(f"RandomForest MSE: {rf_mse}, R2: {rf_r2}")
    print(f"GradientBoosting MSE: {gbm_mse}, R2: {gbm_r2}")
    print(f"XGBoost MSE: {xgb_mse}, R2: {xgb_r2}")

    # Function to train and log model
    def train_and_log_model(pipeline, model_name, X_train, y_train, X_test, y_test):
        try:
            with mlflow.start_run(run_name=model_name):
                pipeline.fit(X_train, y_train)
                predictions = pipeline.predict(X_test)
                mse = mean_squared_error(y_test, predictions)
                r2 = r2_score(y_test, predictions)

                mlflow.log_param("model_name", model_name)
                mlflow.log_metric("mse", mse)
                mlflow.log_metric("r2", r2)

                if model_name == "XGBRegressor":
                    mlflow.xgboost.log_model(pipeline.named_steps['model'], artifact_path="model")
                else:
                    mlflow.sklearn.log_model(pipeline.named_steps['model'], artifact_path="model")

                print(f"{model_name} MSE: {mse}, R2: {r2}")
        except Exception as e:
            print(f"Failed to log {model_name} model: {e}")

    # Log models with MLflow
    train_and_log_model(rf_pipeline, "RandomForestRegressor", X_train, y_train, X_test, y_test)
    train_and_log_model(gbm_pipeline, "GradientBoostingRegressor", X_train, y_train, X_test, y_test)
    train_and_log_model(xgb_pipeline, "XGBRegressor", X_train, y_train, X_test, y_test)

    print("Model Tracking Complete")
    ```

## Results

The models were successfully trained and evaluated, with the following performance metrics:

- **RandomForestRegressor**:
  - MSE: 12.915658909130464
  - R2: 0.827437876907103

- **GradientBoostingRegressor**:
  - MSE: 6.8877532607157015
  - R2: 0.9079748594809287

- **XGBRegressor**:
  - MSE: 6.408346774486543
  - R2: 0.9143800612341066

## Acknowledgements

- This project uses the MLflow and Ngrok libraries.
- The dataset used is `Student Performance BADM 449.xlsx`.

## License

This project is licensed under the MIT License - see the LICENSE.md file for details.
