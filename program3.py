# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.decomposition import PCA

# Load the Iris dataset
iris = datasets.load_iris()
X = iris.data  # All four features (sepal length, sepal width, petal length, petal width)
y = iris.target

# Split the dataset into training and testing sets (70% training, 30% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Apply PCA to reduce the data to 2 dimensions for visualization
pca = PCA(n_components=2)
X_train_2d = pca.fit_transform(X_train)
X_test_2d = pca.transform(X_test)

# Function to train and evaluate SVM with different kernels
def train_and_evaluate_svm(kernel_type):
    # Create the SVM model with the specified kernel type
    svm_model = SVC(kernel=kernel_type)
    
    # Train the model on the training data
    svm_model.fit(X_train_2d, y_train)
    
    # Make predictions on the test set
    y_pred = svm_model.predict(X_test_2d)
    
    # Evaluate the model's performance
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy for {kernel_type} kernel: {accuracy * 100:.2f}%")
    print(f"Classification Report for {kernel_type} kernel:")
    print(classification_report(y_test, y_pred))

# Function to plot decision boundary for each kernel
def plot_decision_boundary(kernel_type):
    # Train the SVM model on the training data using the specified kernel
    svm_model = SVC(kernel=kernel_type)
    svm_model.fit(X_train_2d, y_train)

    # Create a meshgrid for plotting decision boundaries
    h = .02  # Step size in meshgrid
    x_min, x_max = X_train_2d[:, 0].min() - 1, X_train_2d[:, 0].max() + 1
    y_min, y_max = X_train_2d[:, 1].min() - 1, X_train_2d[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Predict the labels for the entire meshgrid
    Z = svm_model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plot the decision boundary
    plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.Paired)

    # Plot the training points
    plt.scatter(X_train_2d[:, 0], X_train_2d[:, 1], c=y_train, edgecolors='k', marker='o', s=100, cmap=plt.cm.Paired, label="Train data")

    # Plot the test points
    plt.scatter(X_test_2d[:, 0], X_test_2d[:, 1], c=y_test, edgecolors='k', marker='x', s=100, cmap=plt.cm.Paired, label="Test data")

    # Set plot details
    plt.title(f"SVM with {kernel_type} kernel on Iris Dataset (PCA Reduced)")
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend()

    # Show the plot
    plt.show()

# Train and evaluate SVM with different kernels
kernels = ['linear', 'poly', 'rbf']
for kernel in kernels:
    train_and_evaluate_svm(kernel)

    # Plot decision boundaries for each kernel
    plot_decision_boundary(kernel)
