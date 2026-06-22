# AI-Driven-HR-Attrition-Prediction-System
AI-Driven HR Attrition Prediction using Machine Learning and Explainable AI (SHAP). The system predicts employee attrition risk, provides retention insights, and includes an interactive GUI for real-time predictions


📌 Overview

AI-Driven HR Attrition Prediction is a Machine Learning project that predicts whether an employee is likely to leave an organization. The system helps HR departments identify employees at risk of attrition and take proactive retention measures.

The project uses Ensemble Learning techniques along with Explainable AI (SHAP) to provide accurate and interpretable predictions.

🎯 Objectives
Predict employee attrition using Machine Learning.
Compare the performance of multiple classification models.
Improve prediction accuracy using Ensemble Learning.
Provide model transparency through Explainable AI (SHAP).
Develop an interactive GUI for real-time attrition prediction.
🚀 Features
Employee Attrition Prediction
Ensemble Learning Model
Explainable AI using SHAP Waterfall Plot
Feature Importance Analysis
Correlation Heatmap Visualization
Confusion Matrix and ROC Curve Evaluation
Interactive GUI for User Inputs
Business Insights for HR Decision Making
🛠 Technologies Used
Programming Language
Python
Libraries
Pandas
NumPy
Scikit-Learn
Matplotlib
Seaborn
SHAP
Imbalanced-Learn (SMOTE)
Tkinter
📊 Machine Learning Models Used
Logistic Regression
Decision Tree
Random Forest
Gradient Boosting
Ensemble Model (Random Forest + Gradient Boosting)
📈 Model Performance
Model	Accuracy
Logistic Regression	0.769231
Decision Tree	0.811741
Random Forest	0.908907
Gradient Boosting	0.894737
Ensemble Model	0.914980

The Ensemble Model achieved the highest accuracy and was selected as the final prediction model.

🔄 Project Workflow
Data Collection
Data Preprocessing
Label Encoding
Correlation Analysis
Data Balancing using SMOTE
Model Training
Hyperparameter Tuning
Ensemble Model Development
Model Evaluation
Explainable AI using SHAP
GUI Development
Business Insight Generation
📊 Explainable AI (SHAP)

SHAP (SHapley Additive exPlanations) is used to explain model predictions.

The project includes:

SHAP Waterfall Plot
Individual Employee Prediction Explanation
Feature Contribution Analysis

This helps HR professionals understand why a particular employee is predicted to stay or leave.

💻 GUI Module

The graphical user interface allows users to enter employee details such as:

Age
Monthly Income
Distance From Home
Overtime Status
Job Satisfaction
Work-Life Balance

The system then predicts:

Employee likely to Stay or Leave
Attrition Risk Score
Possible Reasons for Prediction

📊 Key Business Insights
Employees working overtime show higher attrition risk.
Low job satisfaction increases resignation probability.
Poor work-life balance contributes to employee turnover.
Long commuting distances increase attrition likelihood.
Higher income and better work-life balance improve employee retention.
