from sklearn.model_selection import GridSearchCV
import os
import sys
import joblib
from src.exception import CustomException
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score


def save_object(file_path: str, obj: object) -> None:
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
    params: dict
) -> tuple:
    try:
        report = {}
        best_models = {}
        for i in range(len(list(models))):
            model = list(models.values())[i]
            model_name = list(models.keys())[i]

            param = params[model_name]


            gs = GridSearchCV(
                estimator=model,
                param_grid=param,
                cv=5,
                scoring='r2',
                n_jobs=-1,
                verbose=1
            )
            gs.fit(X_train, y_train)

            best_model = gs.best_estimator_
            best_models[model_name] = best_model

            y_train_pred = best_model.predict(X_train)
            y_test_pred = best_model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)


            report[model_name] = test_model_score
        return report, best_models
    except Exception as e:
        raise CustomException(e, sys)

