import os
import dill
import pickle
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score

def save_object(file_path, obj):
    """
    Save an object to the specified file path using dill.
    """
    dir_path = os.path.dirname(file_path)
    os.makedirs(dir_path, exist_ok=True)
    with open(file_path, "wb") as file_obj:
        dill.dump(obj, file_obj)

def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    """
    Evaluate multiple models using GridSearchCV and return a dictionary of test scores.
    """
    report = {}
    for i in range(len(list(models))):
        model = list(models.values())[i]
        para = param[list(models.keys())[i]]

        gs = GridSearchCV(model, para, cv=3)
        gs.fit(X_train, y_train)

        model.set_params(**gs.best_params_)
        model.fit(X_train, y_train)

        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        train_model_score = r2_score(y_train, y_train_pred)
        test_model_score = r2_score(y_test, y_test_pred)

        report[list(models.keys())[i]] = test_model_score

    return report

def load_object(file_path):
    """
    Load a saved object from the specified file path using dill or pickle.
    Tries both methods to handle different serialization formats.
    """
    try:
        # First try loading with pickle (standard method used in training)
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)
    except Exception as pickle_error:
        try:
            # If pickle fails, try dill
            with open(file_path, "rb") as file_obj:
                return dill.load(file_obj)
        except Exception as dill_error:
            # If both fail, raise a more informative error
            raise Exception(f"Failed to load object from {file_path}. "
                          f"Pickle error: {str(pickle_error)}. "
                          f"Dill error: {str(dill_error)}. "
                          f"This might be due to scikit-learn version incompatibility. "
                          f"Consider retraining the models with the current sklearn version.")
