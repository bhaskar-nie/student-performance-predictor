import os
import sys
import pandas as pd
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        # Load model and preprocessor from saved locations
        model_path = os.path.join("IPYNB Notebooks", "linear_regression_model.pkl")
        preprocessor_path = os.path.join('IPYNB Notebooks', 'preprocessor_linear_regression.pkl')

        # Load the model and preprocessor using load_object function
        model = load_object(file_path=model_path)
        preprocessor = load_object(file_path=preprocessor_path)

        # Apply preprocessor transformation
        data_scaled = preprocessor.transform(features)

        # Make predictions using the model
        preds = model.predict(data_scaled)

        return preds
    

class CustomData:
    def __init__(self,
                 gender: str,
                 learning_support: str,
                 parent_education,
                 had_lunch: str,
                 course_completed: str,
                 percentage_in_test1: int,
                 percentage_in_test2: int,
                ):

        # Initialize the custom data features
        self.gender = gender
        self.learning_support = learning_support
        self.parent_education = parent_education
        self.had_lunch = had_lunch
        self.course_completed = course_completed
        self.percentage_in_test1 = percentage_in_test1
        self.percentage_in_test2 = percentage_in_test2

    def get_data_as_data_frame(self):
        # Create a DataFrame for input data based on user inputs
        custom_data_input_dict = {
            "Gender": [self.gender],
            "Learning Support": [self.learning_support],
            "Parent's Education": [self.parent_education],
            "Had Lunch": [self.had_lunch],
            "Course Completed": [self.course_completed],
            "Percentage in Test 1": [self.percentage_in_test1],
            "Percentage in Test 2": [self.percentage_in_test2],
        }

        return pd.DataFrame(custom_data_input_dict)
