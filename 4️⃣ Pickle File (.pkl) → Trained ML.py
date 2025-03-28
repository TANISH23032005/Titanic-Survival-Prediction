
# 🚢 Titanic Survival Prediction (with Model Saving)
# This script performs a complete ML pipeline and saves the trained model as a .pkl file

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib
import warnings
warnings.filterwarnings('ignore')

# 📊 1. Data Loading
file_path = 'tested.csv'
df = pd.read_csv(file_path)
print("Dataset loaded successfully!")
print(df.info())
print(df.head())

# 🧹 2. Data Cleaning & Preprocessing
df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Fare'].fillna(df['Fare'].median(), inplace=True)

le = LabelEncoder()
df['Sex'] = le.fit_transform(df['Sex'])
df['Embarked'] = le.fit_transform(df['Embarked'])

print("\nMissing values after cleaning:")
print(df.isnull().sum())

# 📊 3. Exploratory Data Analysis (EDA)
sns.countplot(x='Survived', data=df)
plt.title('Survival Distribution')
plt.savefig('survival_distribution.png')
plt.close()

plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(), annot=True, cmap='Blues')
plt.title('Feature Correlation Heatmap')
plt.savefig('correlation_heatmap.png')
plt.close()

print("\nEDA plots saved as images!")

# 🤖 4. Model Training & Evaluation
X = df.drop('Survived', axis=1)
y = df['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\nModel Performance:")
print("Accuracy Score:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png')
plt.close()

print("\nConfusion matrix saved as an image!")

# 🗂️ 5. Save Model as .pkl
joblib.dump(model, 'titanic_model.pkl')
print("\n✅ Model saved successfully as 'titanic_model.pkl'!")

# ✅ 6. Conclusion
print("\nTask Completed: Titanic Survival Prediction Pipeline Executed Successfully!")
