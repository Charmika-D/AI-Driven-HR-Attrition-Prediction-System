#===========================================
# STEP 1 : IMPORT LIBRARIES
#===========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier
from imblearn.over_sampling import SMOTE

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

import shap

# ===============================================
# STEP 2 : LOAD DATASET
# ===============================================

df = pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")

# Remove unnecessary columns
df.drop(
    ['EmployeeNumber', 'EmployeeCount', 'StandardHours'],
    axis=1,
    inplace=True
)

print("Dataset Loaded Successfully")

print("\nSample Employee Records:\n")

print(df[['Age',
          'Attrition',
          'Department',
          'JobRole',
          'MonthlyIncome',
          'JobSatisfaction',
          'WorkLifeBalance',
          'OverTime']].head().to_string(index=False))

# ============================================
# STEP 3 : DATA PREPROCESSING
# ============================================

# Convert categorical columns into numerical

le = LabelEncoder()

for col in df.columns:
    if df[col].dtype == 'object' or df[col].dtype == 'string':
        df[col] = le.fit_transform(df[col])

print("\nData Preprocessing Completed")

"""# ======================================================
# CORRELATION HEATMAP(all features are considered in this)
# =========================================================

print("\nGenerating Correlation Heatmap...")

plt.figure(figsize=(14,10))

sns.heatmap(df.corr(),
            cmap='coolwarm',
            annot=False)

plt.title("Correlation Heatmap")

plt.tight_layout()
plt.show()"""



# ============================================
# STEP 4 : SPLIT INPUT AND OUTPUT
# ============================================

X = df.drop("Attrition", axis=1)
y = df["Attrition"]

# ============================================
# SMOTE BALANCING
# ============================================

smote = SMOTE(random_state=42)

X, y = smote.fit_resample(X, y)

print("Dataset balanced using SMOTE")


# ============================================
# STEP 5 : TRAIN TEST SPLIT
# ============================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTrain Test Split Completed")

# ============================================
# STEP 6 : LOGISTIC REGRESSION
# ============================================

lr = LogisticRegression(max_iter=5000)

lr.fit(X_train, y_train)

lr_pred = lr.predict(X_test)

lr_accuracy = accuracy_score(y_test, lr_pred)

print("\nLogistic Regression Accuracy :", lr_accuracy)

# ============================================
# STEP 7 : DECISION TREE
# ============================================

dt = DecisionTreeClassifier(random_state=42)

dt.fit(X_train, y_train)

dt_pred = dt.predict(X_test)

dt_accuracy = accuracy_score(y_test, dt_pred)

print("\nDecision Tree Accuracy :", dt_accuracy)

"""# =======================================================================
# STEP 8 : RANDOM FOREST(without hyperparamter tuning, accuracy is less)
# ===========================================================================

rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf.fit(X_train, y_train)

rf_pred = rf.predict(X_test)

rf_accuracy = accuracy_score(y_test, rf_pred)

print("\nRandom Forest Accuracy :", rf_accuracy)"""

# ============================================
# RANDOM FOREST WITH HYPERPARAMETER TUNING
# ============================================

print("\nApplying Hyperparameter Tuning for Random Forest...")

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [5, 10, None],
    'min_samples_split': [2, 5]
}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=3,
    scoring='accuracy',
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

rf = grid_search.best_estimator_

rf_pred = rf.predict(X_test)

rf_accuracy = accuracy_score(y_test, rf_pred)

print("Best Parameters :", grid_search.best_params_)

print("Random Forest Accuracy :", rf_accuracy)

"""# =============================================================
# STEP 9 : GRADIENT BOOSTING(accuracy is less without tuning)
# ================================================================

gb = GradientBoostingClassifier(random_state=42)

gb.fit(X_train, y_train)

gb_pred = gb.predict(X_test)

gb_accuracy = accuracy_score(y_test, gb_pred)

print("\nGradient Boosting Accuracy :", gb_accuracy)"""

# ===============================================
# STEP 9 : GRADIENT BOOSTING
# ===============================================

gb = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)

gb.fit(X_train, y_train)

gb_pred = gb.predict(X_test)

gb_accuracy = accuracy_score(y_test, gb_pred)

print("\nGradient Boosting Accuracy :", gb_accuracy)


# ============================================
# STEP 10 : ENSEMBLE MODEL
# ============================================

ensemble_model = VotingClassifier(
    estimators=[
        ('rf', rf),
        ('gb', gb)
    ],
    voting='soft'  
)

ensemble_model.fit(X_train, y_train)

ensemble_pred = ensemble_model.predict(X_test)

ensemble_accuracy = accuracy_score(y_test, ensemble_pred)

print("\nEnsemble Model Accuracy :", ensemble_accuracy)

# ============================================
# STEP 11 : MODEL COMPARISON
# ============================================

models = ['Logistic Regression',
          'Decision Tree',
          'Random Forest',
          'Gradient Boosting',
          'Ensemble Model']

accuracies = [lr_accuracy,
              dt_accuracy,
              rf_accuracy,
              gb_accuracy,
              ensemble_accuracy]

comparison_df = pd.DataFrame({
    'Model': models,
    'Accuracy': accuracies
})

print("\nModel Comparison")
print(comparison_df)

# ============================================
# STEP 12 : VISUALIZATION
# ============================================

plt.figure(figsize=(10,6))

sns.barplot(
    x='Model',
    y='Accuracy',
    data=comparison_df
)

plt.title("Model Accuracy Comparison")
plt.xticks(rotation=15)

plt.tight_layout()
plt.show()


# ============================================
# STEP 13 : CONFUSION MATRIX
# ============================================

cm = confusion_matrix(y_test, ensemble_pred)

plt.figure(figsize=(6,5))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues'
)

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.tight_layout()
plt.show()

# ============================================
# STEP 14 : CLASSIFICATION REPORT
# ============================================

print("\nClassification Report")
print(classification_report(y_test, ensemble_pred))

# ============================================
# STEP 15 : PERFORMANCE METRICS
# ============================================

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

precision = precision_score(y_test, ensemble_pred)

recall = recall_score(y_test, ensemble_pred)

f1 = f1_score(y_test, ensemble_pred)

roc_auc = roc_auc_score(y_test, ensemble_pred)

print("\nPerformance Metrics")
print("Precision :", precision)
print("Recall :", recall)
print("F1 Score :", f1)
print("ROC-AUC Score :", roc_auc)


# ============================================
# STEP 16 : ROC CURVE
# ============================================

fpr, tpr, thresholds = roc_curve(y_test, ensemble_pred)

plt.figure(figsize=(6,5))

plt.plot(fpr, tpr, label='ROC Curve')

plt.plot([0,1], [0,1], linestyle='--')

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title("ROC-AUC Curve")

plt.legend()

plt.show()

# ============================================
# STEP 17 : FEATURE IMPORTANCE GRAPH
# ============================================

print("\nGenerating Feature Importance Graph...")

importance = rf.feature_importances_

feature_names = X.columns

feature_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importance
})

feature_df = feature_df.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10,6))

plt.barh(feature_df['Feature'][:10],
         feature_df['Importance'][:10])

plt.xlabel("Importance Score")

plt.ylabel("Features")

plt.title("Top 10 Important Features Affecting Attrition")

plt.gca().invert_yaxis()

plt.tight_layout()
plt.show()

"""# ===============================================
# CORRELATION HEATMAP(manually imp features)
# ===============================================

print("\nGenerating Correlation Heatmap...")

# Top important features from Feature Importance graph
important_features = [
    'Attrition',
    'StockOptionLevel',
    'MonthlyIncome',
    'JobSatisfaction',
    'JobInvolvement',
    'EnvironmentSatisfaction',
    'YearsWithCurrManager',
    'JobLevel',
    'DistanceFromHome'
]

# Correlation matrix
corr_matrix = df[important_features].corr()

# Plot heatmap
plt.figure(figsize=(10, 7))

sns.heatmap(
    corr_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    linewidths=0.5,
    square=True
)

plt.title("Correlation Heatmap of Important Features", fontsize=14)
plt.tight_layout()
plt.show()"""


# ==================================================
# CORRELATION HEATMAP(HR Focused)
# ==================================================

print("\nGenerating HR Correlation Heatmap...")

important_features = [
    'MonthlyIncome',
    'JobSatisfaction',
    'WorkLifeBalance',
    'DistanceFromHome',
    'OverTime',
    'EnvironmentSatisfaction',
    'JobInvolvement',
    'Age',
    'TotalWorkingYears',
    'Attrition'
]

# Create correlation matrix
corr_matrix = df[important_features].corr()

# Plot heatmap
plt.figure(figsize=(10,8))

sns.heatmap(
    corr_matrix,
    annot=True,
    cmap='coolwarm',
    fmt='.2f',
    linewidths=0.5
)

plt.title(
    "HR Factors Correlation Heatmap",
    fontsize=14
)

plt.tight_layout()
plt.show()


# ==================================================
# STEP 18 : EXPLAINABLE AI (SHAP WATERFALL PLOT)
# ==================================================

print("\nGenerating Explainable AI Graph...")

import shap

# Create SHAP Explainer
explainer = shap.TreeExplainer(rf)

# Select employee for explanation
sample_index = 5

# Employee details
print("\n====================================")
print("EMPLOYEE SELECTED FOR EXPLANATION")
print("====================================")
print(X_test.iloc[sample_index])

# Prediction
prediction = rf.predict(X_test.iloc[[sample_index]])[0]

print("\nPrediction Result:")

if prediction == 1:
    print("Employee likely to leave")
else:
    print("Employee likely to stay")

# ==================================================
# CREATE SINGLE EXPLANATION
# ==================================================

single_explanation = explainer(
    X_test.iloc[[sample_index]]
)

# ==================================================
# WATERFALL PLOT
# ==================================================

sample_data = X_test.iloc[[sample_index]]

# Create explanation
exp = explainer(sample_data)

# Select the "Leave" class explanation
shap.plots.waterfall(
    exp[0, :, 1],
    max_display=10,
    show=False
)

plt.gcf().set_size_inches(12, 6)
plt.subplots_adjust(left=0.28)

plt.show()
# ===============================================
#  STEP 20 : BUSINESS INSIGHTS
# ===============================================

print("\nBusiness Insights:")

print("- Employees working overtime show higher attrition risk")
print("- Low job satisfaction increases resignation probability")
print("- Poor work-life balance contributes to employee attrition")
print("- Employees living far from office are more likely to leave")
print("- Higher income and better work-life balance improve retention")

print("\n....Project Execution Completed Successfully....")
print("======THANK YOU======")


# ===============================================
# STEP 21 : TKINTER GUI FOR ATTRITION PREDICTION
# ===============================================

import tkinter as tk
from tkinter import ttk

# ===============================================s
# CREATE WINDOW
# ===============================================

root = tk.Tk()

root.title("Employee Attrition Prediction System")

root.geometry("550x700")

root.config(bg="white")

# ===============================================
# TITLE
# ===============================================

title = tk.Label(
    root,
    text="Employee Attrition Prediction System",
    font=("Arial", 18, "bold"),
    bg="white",
    fg="blue"
)

title.pack(pady=15)

# ===============================================
# AGE
# ===============================================

tk.Label(
    root,
    text="Age",
    bg="white",
    font=("Arial", 11)
).pack()

age_entry = tk.Entry(root, width=35)

age_entry.pack(pady=5)

# ===============================================
# MONTHLY INCOME
# ===============================================

tk.Label(
    root,
    text="Monthly Income",
    bg="white",
    font=("Arial", 11)
).pack()

income_entry = tk.Entry(root, width=35)

income_entry.pack(pady=5)

# ===============================================
# DISTANCE
# ===============================================

tk.Label(
    root,
    text="Distance From Home (km)",
    bg="white",
    font=("Arial", 11)
).pack()

distance_entry = tk.Entry(root, width=35)

distance_entry.pack(pady=5)

# ===============================================
# OVERTIME
# ===============================================

tk.Label(
    root,
    text="Overtime",
    bg="white",
    font=("Arial", 11)
).pack()

overtime_combo = ttk.Combobox(root, width=32)

overtime_combo['values'] = ("Yes", "No")

overtime_combo.pack(pady=5)

# ===============================================
# JOB SATISFACTION
# ===============================================

tk.Label(
    root,
    text="Job Satisfaction",
    bg="white",
    font=("Arial", 11)
).pack()

job_combo = ttk.Combobox(root, width=32)

job_combo['values'] = (
    "1-Low",
    "2-Medium",
    "3-High",
    "4-Very High"
)

job_combo.pack(pady=5)

# ===============================================
# WORK LIFE BALANCE
# ===============================================

tk.Label(
    root,
    text="Work-Life Balance",
    bg="white",
    font=("Arial", 11)
).pack()

wlb_combo = ttk.Combobox(root, width=32)

wlb_combo['values'] = (
    "1-Low",
    "2-Good",
    "3-Better",
    "4-Best"
)

wlb_combo.pack(pady=5)

# ===============================================
# RESULT LABEL
# ===============================================

result_label = tk.Label(
    root,
    text="",
    font=("Arial", 14, "bold"),
    bg="white",
    fg="red"
)

result_label.pack(pady=20)

# ===============================================
# REASONS LABEL
# ===============================================

reasons_label = tk.Label(
    root,
    text="",
    font=("Arial", 11),
    bg="white",
    justify="left",
    wraplength=450,
    fg="darkgreen"
)

reasons_label.pack(pady=10)

# ===============================================
# PREDICTION FUNCTION
# ===============================================

def predict_attrition():

    employee_data = X.iloc[0:1].copy()

    # ===========================================
    # GET VALUES FROM GUI
    # ===========================================

    age = int(age_entry.get())

    income = int(income_entry.get())

    distance = int(distance_entry.get())

    overtime = overtime_combo.get()

    job_satisfaction = int(job_combo.get()[0])

    work_life = int(wlb_combo.get()[0])

    # ===========================================
    # ASSIGN VALUES
    # ===========================================

    employee_data["Age"] = age

    employee_data["MonthlyIncome"] = income

    employee_data["DistanceFromHome"] = distance

    employee_data["OverTime"] = 1 if overtime == "Yes" else 0

    employee_data["JobSatisfaction"] = job_satisfaction

    employee_data["WorkLifeBalance"] = work_life

    # ===========================================
    # MODEL PREDICTION
    # ===========================================

    prediction = ensemble_model.predict(employee_data)
    # ATTRITION RISK PERCENTAGE
    probability = ensemble_model.predict_proba(employee_data)[0][1]

    risk_percentage = round(probability * 100, 2)

    # ===========================================
    # RESULT + REASONS
    # ===========================================

    reasons = ""

    # ===========================================
    # EMPLOYEE LIKELY TO LEAVE
    # ===========================================

    if prediction[0] == 1:

        result = "Employee likely to leave"

        if income < 40000:
            reasons += "• Low monthly income\n"

        if distance > 10:
            reasons += "• Employee lives far from office\n"

        if overtime == "Yes":
            reasons += "• Employee works overtime\n"

        if job_satisfaction <=2:
            reasons += "• Low job satisfaction\n"

        if work_life <=2:
            reasons += "• Poor work-life balance\n"

        result_label.config(
            text=f"{result}\nAttrition Risk Score : {risk_percentage}%",
            fg="red"
        )

    # ===========================================
    # EMPLOYEE LIKELY TO STAY
    # ===========================================

    else:

        result = "Employee likely to stay"

        if income >= 40000:
            reasons += "• Good monthly income\n"

        if distance <= 10:
            reasons += "• Employee lives near office\n"

        if overtime == "No":
            reasons += "• No overtime work\n"

        if job_satisfaction >= 3:
            reasons += "• Good job satisfaction\n"

        if work_life >= 3:
            reasons += "• Good work-life balance\n"

        result_label.config(
            text=f"{result}\nAttrition Risk Score : {100 - risk_percentage}%",
            fg="green"
        )

    # ===========================================
    # HANDLE EMPTY REASONS
    # ===========================================

    if reasons == "":
        reasons = "• No major factors detected"

    # ===========================================
    # DISPLAY REASONS
    # ===========================================

    reasons_label.config(
        text="Possible Reasons:\n\n" + reasons
    )

# ===============================================
# PREDICT BUTTON
# ===============================================

predict_button = tk.Button(
    root,
    text="Predict Attrition",
    command=predict_attrition,
    bg="blue",
    fg="white",
    font=("Arial", 12, "bold"),
    width=22,
    height=2
)

predict_button.pack(pady=20)

# ===============================================
# RUN GUI
# ===============================================

root.mainloop()

