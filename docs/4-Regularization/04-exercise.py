import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline

np.random.seed(42)

X = np.linspace(0, 10, 28).reshape(-1, 1)
y_true = np.sin(X).ravel()
y = y_true + np.random.normal(0, 0.45, size=y_true.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.35, random_state=42
)

x_plot = np.linspace(0, 10, 500).reshape(-1, 1)


#without ridge or lasso
degree = 25

model = make_pipeline(
    PolynomialFeatures(degree, include_bias=False), 
    LinearRegression()
)

model.fit(X_train, y_train)
y_plot = model.predict(x_plot)
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

train_mse = mean_squared_error(y_train, y_train_pred)
test_mse = mean_squared_error(y_test, y_test_pred)

plt.figure(figsize=(8, 5))

plt.scatter(X_train, y_train, color='blue', label='Train Data', s=70)
plt.scatter(X_test, y_test, color='red', label='Test Data', s=70)

plt.plot(x_plot, y_plot, color='green', label='Model Prediction', linewidth=3)

plt.title(
    f"Without Regularization\n"
    f"Train MSE: {train_mse:.2f}, Test MSE: {test_mse:.2f}, Degree: {degree}"
)

plt.ylim(-3, 3)
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()

#with ridge
alpha = 0.0005

model_ridge = make_pipeline(
    PolynomialFeatures(degree, include_bias=False),
    StandardScaler(),
    Ridge(alpha=alpha)
)

model_ridge.fit(X_train, y_train)

y_plot_ridge = model_ridge.predict(x_plot)
y_train_pred_ridge = model_ridge.predict(X_train)
y_test_pred_ridge = model_ridge.predict(X_test)

train_mse_ridge = mean_squared_error(y_train, y_train_pred_ridge)
test_mse_ridge = mean_squared_error(y_test, y_test_pred_ridge)

plt.figure(figsize=(8, 5))

plt.scatter(X_train, y_train, color='blue', label='Train Data', s=70)
plt.scatter(X_test, y_test, color='red', label='Test Data', s=70)

plt.plot(x_plot, y_plot_ridge, color='green', label='Ridge Prediction', linewidth=3)

plt.title(
    f"With Ridge Regularization (alpha={alpha})\n"
    f"Train MSE: {train_mse_ridge:.2f}, Test MSE: {test_mse_ridge:.2f}, Degree: {degree}, Alpha: {alpha}"
)

plt.ylim(-3, 3)
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()

#with lasso
from sklearn.linear_model import Lasso

alpha = 0.0005

max_iter = 10000

model_lasso = make_pipeline(
    PolynomialFeatures(degree, include_bias=False),
    StandardScaler(),
    Lasso(alpha=alpha, max_iter=max_iter)
)

model_lasso.fit(X_train, y_train)

y_plot_lasso = model_lasso.predict(x_plot)
y_train_pred_lasso = model_lasso.predict(X_train)
y_test_pred_lasso = model_lasso.predict(X_test)

train_mse_lasso = mean_squared_error(y_train, y_train_pred_lasso)
test_mse_lasso = mean_squared_error(y_test, y_test_pred_lasso)

plt.figure(figsize=(8, 5))

plt.scatter(X_train, y_train, label="Training data", s=70)
plt.scatter(X_test, y_test, label="Test data", s=70)

plt.plot(x_plot, y_plot_lasso, linewidth=3, label="Lasso model prediction")

plt.title(
    f"With Lasso Regularization\n"
    f"Degree = {degree} | Alpha = {alpha} | "
    f"Train MSE = {train_mse_lasso:.3f} | Test MSE = {test_mse_lasso:.3f}"
)

plt.ylim(-3, 3)
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()