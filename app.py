from flask import Flask, request, render_template, jsonify
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application=Flask(__name__)

app=application

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        # Retrieve form inputs
        percentage_in_test1 = float(request.form.get('percentage_in_test1'))
        percentage_in_test2 = float(request.form.get('percentage_in_test2'))

        # Create a CustomData instance and predict
        data = CustomData(
            gender=request.form.get('gender'),
            learning_support=request.form.get('learning_support'),
            parent_education=request.form.get('parent_education'),
            had_lunch=request.form.get('had_lunch'),
            course_completed=request.form.get('course_completed'),
            percentage_in_test1=percentage_in_test1,
            percentage_in_test2=percentage_in_test2
        )
        pred_df = data.get_data_as_data_frame()
        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)

        # Ensure predicted results are within bounds
        predicted_test3_score = min(max(results[0], 0), 100)

        return render_template(
            'home.html',
            results=predicted_test3_score,
            percentage_in_test1=percentage_in_test1,
            percentage_in_test2=percentage_in_test2
        )


@app.route('/predictapi', methods=['POST'])
def predict_api():
    try:
        # Get data from the request body (expecting JSON)
        data = request.get_json()

        # Extract values from the JSON
        gender = data.get('gender')
        learning_support = data.get('learning_support')
        parent_education = data.get('parent_education')
        had_lunch = data.get('had_lunch')
        course_completed = data.get('course_completed')
        percentage_in_test1 = float(data.get('percentage_in_test1'))
        percentage_in_test2 = float(data.get('percentage_in_test2'))

        # Prepare the data in the correct format
        custom_data = CustomData(
            gender=gender,
            learning_support=learning_support,
            parent_education=parent_education,
            had_lunch=had_lunch,
            course_completed=course_completed,
            percentage_in_test1=percentage_in_test1,
            percentage_in_test2=percentage_in_test2
        )
        pred_df = custom_data.get_data_as_data_frame()

        # Make prediction
        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)

        # Constrain the predicted score to the range [0, 100]
        predicted_score = max(0, min(100, results[0]))

        # Generate the graph
        test_scores = [percentage_in_test1, percentage_in_test2, predicted_score]
        test_labels = ['Test 1', 'Test 2', 'Test 3']

        plt.figure(figsize=(6, 4))
        plt.bar(test_labels, test_scores, color=['blue', 'green', 'orange'])
        plt.title("Student's Test Performance")
        plt.ylabel("Percentage")
        plt.ylim(0, 100)

        # Save the graph to a bytes buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        encoded_graph = base64.b64encode(buf.getvalue()).decode('utf-8')
        buf.close()

        # Return prediction and graph as JSON
        return jsonify({
            "prediction": predicted_score,
            "graph": f"data:image/png;base64,{encoded_graph}"
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 400
    
@app.route('/insights')
def insights():
    return render_template('insights.html')

    
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=1000, debug=True)
