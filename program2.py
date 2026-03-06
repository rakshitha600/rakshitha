# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures

%matplotlib inline

# Load California housing dataset
california = fetch_california_housing(as_frame=True)
df = california.frame

# Display first few rows
print(df.head())

# Check for missing values
print("Missing values in each column:")
print(df.isnull().sum())

# Plot histogram of target variable
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.histplot(df['MedHouseVal'], bins=30, kde=True)
plt.title("Distribution of Median House Value")
plt.xlabel("MedHouseVal")
plt.ylabel("Frequency")
plt.show()

# Compute correlation matrix and plot heatmap
correlation_matrix = df.corr().round(2)
plt.figure(figsize=(12,10))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()

# Select two features most correlated with target
top_features = correlation_matrix['MedHouseVal'].drop('MedHouseVal').abs().sort_values(ascending=False).head(2).index.tolist()
print("Top 2 features most correlated with MedHouseVal:", top_features)

X = df[top_features]
Y = df['MedHouseVal']

# Scatter plots for selected features
plt.figure(figsize=(12,5))
for i, col in enumerate(top_features):
    plt.subplot(1, len(top_features), i+1)
    plt.scatter(X[col], Y, marker='o', alpha=0.5)
    plt.xlabel(col)
    plt.ylabel("MedHouseVal")
    plt.title(f"{col} vs MedHouseVal")
plt.tight_layout()
plt.show()

# Split dataset into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=5)
print(f"Training set shape: {X_train.shape}")
print(f"Testing set shape: {X_test.shape}")

# Linear Regression
lin_model = LinearRegression()
lin_model.fit(X_train, Y_train)

# Evaluate on training set
y_train_pred = lin_model.predict(X_train)
rmse_train = np.sqrt(mean_squared_error(Y_train, y_train_pred))
r2_train = r2_score(Y_train, y_train_pred)

# Evaluate on testing set
y_test_pred = lin_model.predict(X_test)
rmse_test = np.sqrt(mean_squared_error(Y_test, y_test_pred))
r2_test = r2_score(Y_test, y_test_pred)

print("\nLinear Regression Performance:")
print(f"Training set -> RMSE: {rmse_train:.4f}, R2: {r2_train:.4f}")
print(f"Testing set -> RMSE: {rmse_test:.4f}, R2: {r2_test:.4f}")

# Plot actual vs predicted
plt.figure(figsize=(6,6))
plt.scatter(Y_test, y_test_pred, alpha=0.5)
plt.xlabel("Actual MedHouseVal")
plt.ylabel("Predicted MedHouseVal")
plt.title("Actual vs Predicted Values")
plt.plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], 'r--')  # diagonal line
plt.show()

# Polynomial Regression function
def create_polynomial_regression_model(degree):
    poly_features = PolynomialFeatures(degree=degree)
    X_train_poly = poly_features.fit_transform(X_train)
    
    poly_model = LinearRegression()
    poly_model.fit(X_train_poly, Y_train)
    
    y_train_pred = poly_model.predict(X_train_poly)
    y_test_pred = poly_model.predict(poly_features.transform(X_test))
    
    rmse_train = np.sqrt(mean_squared_error(Y_train, y_train_pred))
    r2_train = r2_score(Y_train, y_train_pred)
    
    rmse_test = np.sqrt(mean_squared_error(Y_test, y_test_pred))
    r2_test = r2_score(Y_test, y_test_pred)
    
    print(f"\nPolynomial Regression (degree={degree}) Performance:")
    print(f"Training set -> RMSE: {rmse_train:.4f}, R2: {r2_train:.4f}")
    print(f"Testing set -> RMSE: {rmse_test:.4f}, R2: {r2_test:.4f}")

# Test polynomial regression with degree 2
create_polynomial_regression_model(2)
