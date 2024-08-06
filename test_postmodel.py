import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_classification
import os
from post_model_viz import PostModelViz
import pandas as pd

# Create a directory for plots if it doesn't exist
output_path = 'PostModelVizPlots/'
os.makedirs(output_path, exist_ok=True)

# Mock Data
# For simplicity, generate some mock data for residuals, Cook's distance, leverage, and F1 scores
residuals = np.random.normal(0, 1, 100)
cooks_d = np.random.uniform(0, 1, 100)
leverage = np.random.uniform(0, 1, 100)
f1_scores = {'Class 0': 0.8, 'Class 1': 0.75}

# Mock Confusion Matrix
y_true = [0, 1, 0, 1, 0, 1, 0, 0, 1, 1]
y_pred = [0, 1, 0, 1, 1, 0, 0, 0, 1, 1]
cm = confusion_matrix(y_true, y_pred)

# Mock ROC Curve data
fpr, tpr, _ = roc_curve(y_true, y_pred)

# Mock Learning Curve Data
train_sizes = [1, 2, 3, 4, 5]
train_scores = [0.5, 0.6, 0.7, 0.8, 0.9]
test_scores = [0.4, 0.5, 0.6, 0.7, 0.8]

# Regression Line Plot Example
# Create mock data
X, y = np.arange(10).reshape(-1, 1), np.arange(10) + np.random.normal(size=10)
model = LinearRegression().fit(X, y)
regression_line_data = {'X': X.flatten(), 'y': y}

# Use PostModelViz methods
PostModelViz.residual_plot(pd.Series(residuals), output_path)
PostModelViz.cooks_distance_plot(pd.Series(cooks_d), output_path)
PostModelViz.leverage_plot(pd.Series(leverage), output_path)
PostModelViz.confusion_matrix_plot(cm, output_path)
PostModelViz.learning_curve_plot(train_sizes, train_scores, test_scores, output_path)
PostModelViz.roc_curve_plot(fpr, tpr, output_path)
PostModelViz.f1_score_plot(f1_scores, output_path)
PostModelViz.regression_line_plot(pd.DataFrame(regression_line_data), 'X', 'y', output_path)

print(f"Post-model plots have been saved in the '{output_path}' directory.")
