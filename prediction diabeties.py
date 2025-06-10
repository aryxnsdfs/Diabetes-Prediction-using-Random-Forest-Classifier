import pandas as pd, numpy as np
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score

df=pd.read_csv('diabetes.csv')

exc=['Pregnancies','Outcome']
exclude=df.columns.difference(exc)
df[exclude]=df[exclude].replace(0,np.nan)
df[exclude]=df[exclude].fillna(df[exclude].median())

corr=df.corr()
plt.figure(figsize=(10,8))
sns.heatmap(corr,annot=True,cmap='coolwarm',linewidths=0.5)
plt.title('Feature correlation HeatMap')
plt.show()

x=df[['Glucose','BloodPressure','Insulin','SkinThickness','DiabetesPedigreeFunction','Age','BMI']]
y=df['Outcome']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

model=RandomForestClassifier(n_estimators=100,random_state=42,class_weight='balanced')
model.fit(x_train,y_train)

y_pred = model.predict(x_test)
print(accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))


cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

importances=model.feature_importances_
features=x.columns
plt.figure(figsize=(8,6))
sns.barplot(x=importances,y=features)
plt.show()