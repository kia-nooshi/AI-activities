import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score

# Load the dataset
iris = load_iris()
X = iris.data[:, :2]  # Use only the first two features for simplicity
y = iris.target

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize and train the logistic regression model
logistic_model = LogisticRegression(random_state=42)
logistic_model.fit(X_train_scaled, y_train)

# Predict the labels for the test set
y_pred = logistic_model.predict(X_test_scaled)

# Calculate the accuracy of the logistic model
accuracy = accuracy_score(y_test, y_pred)
print(f"Classification accuracy: {accuracy*100:.2f}%")

# Logistic Regression Decision Boundaries
x_min, x_max = X_train_scaled[:, 0].min() - 1, X_train_scaled[:, 0].max() + 1
y_min, y_max = X_train_scaled[:, 1].min() - 1, X_train_scaled[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))
Z = logistic_model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.contourf(xx, yy, Z, alpha=0.4)
plt.scatter(X_train_scaled[:, 0], X_train_scaled[:, 1], c=y_train, s=20, edgecolor='k')
plt.title('Logistic Regression Decision Boundaries')
plt.xlabel('Sepal length (standardized)')
plt.ylabel('Sepal width (standardized)')

# Linear Regression Part
# For simplicity, using the same feature as input and output (not ideal in practice)
linear_model = LinearRegression()
linear_model.fit(X_train_scaled[:, 0].reshape(-1, 1), X_train_scaled[:, 1])
x_line = np.linspace(X_train_scaled[:, 0].min(), X_train_scaled[:, 0].max(), 100)
y_line = linear_model.predict(x_line.reshape(-1, 1))

plt.subplot(1, 2, 2)
plt.scatter(X_train_scaled[:, 0], X_train_scaled[:, 1], c='blue')
plt.plot(x_line, y_line, color='red')  # Linear regression line
plt.title('Linear Regression')
plt.xlabel('Sepal length (standardized)')
plt.ylabel('Sepal width (predicted)')

plt.tight_layout()
plt.show()

# Print the weights (coefficients) and biases (intercepts) of the logistic model
print("Logistic model weights (coefficients):", logistic_model.coef_)
print("Logistic model biases (intercepts):", logistic_model.intercept_)

# Print the coefficient and intercept of the linear model
print("Linear model coefficient:", linear_model.coef_)
print("Linear model intercept:", linear_model.intercept_)