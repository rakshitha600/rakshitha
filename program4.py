# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load dataset
df = pd.read_csv("https://calmcode.io/static/data/titanic.csv")

# Preprocessing
df['age'].fillna(df['age'].median(), inplace=True)
df['sex'] = df['sex'].map({'male':0,'female':1})

# Features and target
X = df.select_dtypes(include=['int64','float64']).drop('survived', axis=1)
y = df['survived']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Feature importance plot
plt.barh(X.columns, model.feature_importances_)
plt.title("Feature Importance - Random Forest")
plt.xlabel("Importance")
plt.show()