#!/usr/bin/env python3
"""
Script to retrain and save models with current sklearn version
"""
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import pickle

def retrain_models():
    """Retrain all models and save them with current sklearn version"""
    
    print("Loading data...")
    # Load the dataset
    df = pd.read_csv('data.csv')
    
    print("Preparing features...")
    # Prepare X and y
    X = df.drop(columns=['Percentage in Test 3'], axis=1)
    y = df['Percentage in Test 3']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Identify feature types
    num_features = X.select_dtypes(exclude="object").columns
    cat_features = X.select_dtypes(include="object").columns
    
    print(f"Numerical features: {list(num_features)}")
    print(f"Categorical features: {list(cat_features)}")
    
    # Create preprocessor
    numeric_transformer = StandardScaler()
    oh_transformer = OneHotEncoder(drop='first', sparse_output=False)
    
    preprocessor = ColumnTransformer(
        [
            ("OneHotEncoder", oh_transformer, cat_features),
            ("StandardScaler", numeric_transformer, num_features),        
        ]
    )
    
    print("Fitting preprocessor...")
    # Fit the preprocessor on training data
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    # Define models
    models = {
        "Linear Regression": LinearRegression(),
        "K-Neighbors Regressor": KNeighborsRegressor(),
        "Decision Tree": DecisionTreeRegressor(random_state=42),
        "Random Forest Regressor": RandomForestRegressor(random_state=42),
    }
    
    # Create pkl files directory if it doesn't exist
    os.makedirs("pkl files", exist_ok=True)
    
    print("\nTraining and saving models...")
    print("=" * 60)
    
    for model_name, model in models.items():
        print(f"\nTraining {model_name}...")
        
        # Train the model
        model.fit(X_train_processed, y_train)
        
        # Make predictions and calculate R2 score
        y_pred = model.predict(X_test_processed)
        r2 = r2_score(y_test, y_pred)
        
        print(f"R² Score: {r2:.4f}")
        
        # Save the trained model using pickle
        model_filename = f"pkl files/{model_name.replace(' ', '_').lower()}_model.pkl"
        with open(model_filename, 'wb') as f:
            pickle.dump(model, f)
        print(f"✅ Saved model: {model_filename}")
        
        # Save the preprocessor (same for all models but saved separately for each)
        preprocessor_filename = f"pkl files/preprocessor_{model_name.replace(' ', '_').lower()}.pkl"
        with open(preprocessor_filename, 'wb') as f:
            pickle.dump(preprocessor, f)
        print(f"✅ Saved preprocessor: {preprocessor_filename}")
    
    print(f"\n🎉 All models retrained and saved successfully!")
    print("You can now run the Flask app without compatibility issues.")

def test_loading():
    """Test if the newly saved models can be loaded"""
    print("\n" + "=" * 60)
    print("Testing model loading...")
    
    try:
        from src.utils import load_object
        
        # Test loading linear regression model and preprocessor
        model = load_object("pkl files/linear_regression_model.pkl")
        preprocessor = load_object("pkl files/preprocessor_linear_regression.pkl")
        
        print("✅ Successfully loaded Linear Regression model and preprocessor")
        print(f"Model type: {type(model)}")
        print(f"Preprocessor type: {type(preprocessor)}")
        
        # Test with sample data
        sample_data = pd.DataFrame({
            "Gender": ["Female"],
            "Learning Support": ["Private Coaching"],
            "Parent's Education": ["Bachelor"],
            "Had Lunch": ["Yes"],
            "Course Completed": ["Complete"],
            "Percentage in Test 1": [78],
            "Percentage in Test 2": [84],
        })
        
        # Transform and predict
        processed_data = preprocessor.transform(sample_data)
        prediction = model.predict(processed_data)
        
        print(f"✅ Test prediction successful: {prediction[0]:.2f}")
        
    except Exception as e:
        print(f"❌ Error during testing: {str(e)}")

if __name__ == "__main__":
    import sklearn
    print(f"Using scikit-learn version: {sklearn.__version__}")
    print("=" * 60)
    
    retrain_models()
    test_loading() 