 Diabetes Prediction using Random Forest Classifier

This project uses the Pima Indians Diabetes Dataset to predict whether a patient has diabetes based on several medical features. The model is built using the **Random Forest Classifier** from scikit-learn, and the performance is evaluated using accuracy, classification reports, and confusion matrices.


 Dataset
The dataset used is `diabetes.csv`, which contains the following features:

- Pregnancies
- Glucose
- BloodPressure
- SkinThickness
- Insulin
- BMI
- DiabetesPedigreeFunction
- Age
- Outcome (Target Variable: 0 = No Diabetes, 1 = Diabetes)

---
- Handled hidden missing values (zeros in non-logical columns)
- Replaced missing values with column-wise median
- Correlation heatmap using Seaborn
- Model training with `RandomForestClassifier`
- Eva
