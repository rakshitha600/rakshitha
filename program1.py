import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load California Housing dataset
housing = fetch_california_housing(as_frame=True)
boston = housing.frame  # keep same variable name for compatibility
boston['MEDV'] = boston['MedHouseVal']  # rename target column

# Check missing values
print("Missing values per column:")
print(boston.isnull().sum())

# Histogram of MEDV
sns.set(rc={'figure.figsize': (11.7, 8.27)})
sns.histplot(boston['MEDV'], bins=30, kde=True)
plt.title('Distribution of MEDV')
plt.xlabel('MEDV')
plt.ylabel('Frequency')
plt.show()

# Correlation heatmap
correlation_matrix = boston.corr().round(2)
sns.heatmap(data=correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Scatter plots for two features vs target
# Choose two features most correlated with MEDV
features = ['AveRooms', 'AveOccup']  # example features
target = boston['MEDV']

plt.figure(figsize=(20, 5))
for i, col in enumerate(features):
    plt.subplot(1, len(features), i + 1)
    plt.scatter(boston[col], target, marker='o')
    plt.title(f'{col} vs MEDV')
    plt.xlabel(col)
    plt.ylabel('MEDV')
plt.show()

# Prepare features and target for modeling
X = boston[features]
Y = boston['MEDV']

# Train-test split (80% train, 20% test)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=5)

# Train Linear Regression model
lin_model = LinearRegression()
lin_model.fit(X_train, Y_train)

# Evaluate model on training set
y_train_predict = lin_model.predict(X_train)
train_rmse = np.sqrt(mean_squared_error(Y_train, y_train_predict))
train_r2 = r2_score(Y_train, y_train_predict)
print("Training Set Performance")
print("------------------------")
print("RMSE:", train_rmse)
print("R²:", train_r2)

# Evaluate model on testing set
y_test_predict = lin_model.predict(X_test)
test_rmse = np.sqrt(mean_squared_error(Y_test, y_test_predict))
test_r2 = r2_score(Y_test, y_test_predict)
print("\nTesting Set Performance")
print("------------------------")
print("RMSE:", test_rmse)
print("R²:", test_r2)

# Plot predicted vs actual values
plt.scatter(Y_test, y_test_predict)
plt.plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], 'r--')  # ideal line
plt.xlabel('Actual MEDV')
plt.ylabel('Predicted MEDV')
plt.title('Actual vs Predicted MEDV')
plt.show()