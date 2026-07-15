from scipy.optimize._trustregion_constr import report
import os
import sys
import joblib
from src.exception import CustomException
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        joblib.dump(obj, file_path)

    except Exception as e:
        raise CustomException(e, sys)


def evaluate_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    models: dict,
) -> dict:
    try:
        report ={}
        for i in range(len(list(models))):
            model = list(models.values())[i]
            model_name = list(models.keys())[i]

            model.fit(X_train, y_train)

            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)


            report[model_name] = test_model_score
        return report
    except Exception as e:
        raise CustomException(e, sys)

