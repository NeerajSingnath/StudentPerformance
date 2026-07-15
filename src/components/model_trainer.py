from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
import sys
import os

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
from src.utils import evaluate_models
from dataclasses import dataclass


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifact", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    # pyrefly: ignore [implicit-any-parameter]
    def initiate_model_training(self, train_array, test_array, preprocessor_path):
        try:
            logging.info("Splitting train and test Array")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1],
            )

            logging.info("Training the Model")
            models = {
                "Linear Regression": LinearRegression(),
                "Random Forest Regressor": RandomForestRegressor(),
                "Gradient Boosting Regressor": GradientBoostingRegressor(),
                "K Neighbors Regressor": KNeighborsRegressor(),
                "Decision Tree Regressor": DecisionTreeRegressor(),
                "XGBoost Regressor": XGBRegressor(),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }

            model_record: dict = evaluate_models(
                X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models
            )

            logging.info("Best model searching")
            best_model_score = max(sorted(model_record.values()))

            best_model_name =  list(model_record.keys())[list(model_record.values()).index(best_model_score)]

            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No best model found", sys)

            logging.info("Best model found")
            save_object(file_path=self.model_trainer_config.trained_model_file_path, obj=best_model)


            predicted = best_model.predict(X_test)

            r2 = r2_score(y_test, predicted)
            mse = mean_squared_error(y_test, predicted)
            return r2
        except Exception as e:
            raise CustomException(e, sys)
