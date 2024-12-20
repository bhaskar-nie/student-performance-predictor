# **Student Upcoming Test Marks Prediction Using Flask and Machine Learning**

## **Overview**
This project predicts a student's upcoming test marks based on historical data using a machine learning model. The prediction leverages a Linear Regression model served via a Flask API. The project aims to assist educators and institutions in identifying students who may need additional support to improve their academic outcomes.

You can access the deployed application here: **[Student Upcoming Test Marks Predictor](https://student-performance-predictor-bsa6.onrender.com/)**.
---

## **1. Dataset Details**

### **Features**
#### **Categorical Features**
- **Gender**: Male/Female.
- **Learning Support**: Types of support like Limited Tutoring, Private Coaching, etc.
- **Parent's Education**: Levels such as Bachelor's, Master's, High School, etc.
- **Lunch Status**: Whether the student had lunch before the test (Yes/No).
- **Course Completion**: Indicates if the student completed a preparation course (Complete/None).

#### **Numerical Features**
- **Test Scores**: Percentages in two consecutive tests (Test 1 and Test 2).

---

## **2. Machine Learning Model**

### **Model Selection**
The project trained and evaluated several models:
- Linear Regression
- Decision Trees
- Random Forest
- K-Nearest Neighbors

#### **Chosen Model**: **Linear Regression**
- Achieved an accuracy of **93.7% (R² score)**.
- Selected for its simplicity, speed, and interpretability.

### **Model Evaluation**
- **Metrics**: R² score, Mean Absolute Error (MAE).
- **Rationale**: Linear Regression performed consistently well across all evaluation metrics, making it ideal for this problem.

---

## **3. Flask API**

### **API Endpoint**
- **URL**: `/predictapi`
- **Method**: POST
- **Input**: JSON
- **Output**: JSON

#### **Example Input**
```json
{
  "gender": "Female",
  "learning_support": "Private Coaching",
  "parent_education": "Bachelor",
  "had_lunch": "Yes",
  "course_completed": "Complete",
  "percentage_in_test1": 78,
  "percentage_in_test2": 84
}
```

#### **Example Output**
```json
{
  "prediction": 82.5,
  "graph": "data:image/png;base64,..."
}
```
#### **Screenshots**:

![Student Performance Graph Screenshot](https://github.com/user-attachments/assets/df834208-a61b-4097-ab6a-60ced9f3c4b4)

![POSTMAN Screenshot](https://github.com/user-attachments/assets/5c69520e-356c-48de-8dc2-5711caeafe25)

### **API Error Handling**
- **Invalid Inputs**: Returns error messages for missing or incorrect data.
- **Response Format**: Consistent JSON output with clear messages.
- **Visualization**: Generates a bar graph comparing test scores.

---

## **4. Project Setup**

### **Setup Instructions**
1. Clone the repository:
   ```bash
   git clone https://github.com/bhaskar-nie/student-performance-predictor.git
   ```
2. Navigate to the project directory:
   ```bash
   cd student-performance-predictor
   ```
3. Create a virtual environment (recommended):
   - On Windows:
     ```bash
     python -m venv venv
     venv\Scripts\activate
     ```
   - On macOS/Linux:
     ```bash
     python3 -m venv venv
     source venv/bin/activate
     ```
4. Install dependencies within the virtual environment:
   ```bash
   pip install -r requirements.txt
   ```
5. Run the Flask app:
   ```bash
   python app.py
   ```

---

## **5. Navigating the Web Application**

### **Home Page**
- **URL**: `https://student-performance-predictor-bsa6.onrender.com`
- The **Home Page** introduces the project and its purpose. It provides navigation links to the **Insights Page** and **Prediction Page**.
- **Screenshots**:
    
![Home Page Screenshot](https://github.com/user-attachments/assets/70b99710-e16b-4da4-adb3-9bc2577c166f)

![Home Page Screenshot2](https://github.com/user-attachments/assets/e2c3a7d1-c6ea-4b76-883c-503531c53469)

---

### **Insights Page**
- **URL**: `https://student-performance-predictor-bsa6.onrender.com/insights`
- The **Insights Page** provides visualizations of student performance trends, relationships between features, and distribution of test scores.
- **Features**:
  - Graphs and charts for test score distributions.
  - Insights into categorical feature impacts on performance.
- **Screenshots**:
  
![Insights Page Screenshot](https://github.com/user-attachments/assets/944c1868-199f-482d-99f8-7d29787b1527)

![Insights Page Screenshot2](https://github.com/user-attachments/assets/98a1ad01-9991-4c7a-9e87-ff6c14c8f650)

---

### **Prediction Page**
- **URL**: `https://student-performance-predictor-bsa6.onrender.com/predictdata`
- The **Prediction Page** allows users to input student details and test scores to receive a predicted performance score.
- **Features**:
  - Form fields for all required input data.
  - Displays the predicted score and a bar graph visualization.
- **Screenshot**:
   
![Prediction Page Screenshot](https://github.com/user-attachments/assets/9ca98513-d959-467a-b330-b49d1720a06b)

---

## **6. API Testing**

The detailed API testing steps, including Postman instructions and `curl` commands, are provided in the **API Testing Documentation**.  
Please refer to the uploaded PDF (`API_Testing_Documentation.pdf`) in the repository for comprehensive details.

---

## **7. Docker Deployment**

### **Steps**
1. Build the Docker image:
   ```bash
   docker build -t my-ml-app .
   ```
2. Run the container:
   ```bash
   docker run -p 1000:1000 my-ml-app
   ```
3. Access the web application at `http://127.0.0.1:1000/`.

---

## **8. Conclusion**
This project demonstrates the use of machine learning and Flask for a practical problem in education. It provides a robust and user-friendly interface for predicting student performance, with potential applications in personalized learning and academic intervention strategies.
