import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

#TODO: Will need more imports!
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    mean_absolute_error, 
    mean_squared_error, 
    r2_score, 
    accuracy_score, 
    precision_score, 
    recall_score, 
    roc_auc_score
)
from sklearn.model_selection import train_test_split


class ModelTrainer:
    def __init__(self, df: pd.DataFrame, target: str, features: list[str]):
        self.df = df
        self.target = target
        self.features = features
        self.model = None
        self.metrics = None

    def build_preprocessor(self, X: pd.DataFrame) -> ColumnTransformer:
        """
        DO NOT ALTER THIS METHOD!

        A preprocessing pipeline that automates the cleaning and formatting 
        of your data so it's ready for a machine learning model!

        Splits data into numeric features and categorical features.
        SimpleImputer fills any remaining missing data with column median.
        OneHotEncoder turns text categories into digit categories (to be read by machine).
        Apply the numeric_transformer to numeric data and categorical_transformer to categorical data.
        """
        numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
        categorical_features = X.select_dtypes(exclude=["number"]).columns.tolist()

        numeric_transformer = Pipeline(
            steps=[("imputer", SimpleImputer(strategy="median"))]
        )

        categorical_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]
        )

        return ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, numeric_features),
                ("cat", categorical_transformer, categorical_features),
            ]
        )

    def train_linear(self):
        X = self.df[self.features]
        y = self.df[self.target]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=134340)
        
        preprocessor = self.build_preprocessor(X_train)

        self.model = Pipeline(
            steps=[
                ("preprocessor", preprocessor), 
                ("regressor", LinearRegression())
            ]
        )

        self.model.fit(X_train, y_train)
        preds = self.model.predict(X_test)

        self.metrics = {
            "mse": mean_squared_error(y_test, preds), 
            "mae": mean_absolute_error(y_test, preds), 
            "r2": r2_score(y_test, preds)
        }

        return self.model, self.metrics
    def train_logistic(self):
        #TODO: Train and evaluate a logistic model
        #      Return metrics: accuracy, precision, recall, roc_auc
        pass

    def evaluate(self):
        if self.metrics is None:
            raise ValueError("No metrics available yet. Train a model first.")
        return self.metrics