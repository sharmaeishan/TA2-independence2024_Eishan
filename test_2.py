from csv_analysis import CSVAnalysis 
from post_model_viz import PostModelViz
import pandas as pd
import os
from sklearn.model_selection import train_test_split

df = pd.read_csv("Copy of heart.csv")

output_path = 'Plots/'

os.makedirs(output_path, exist_ok=True)

# Use the class methods
print("Head of DataFrame:")
print(CSVAnalysis.dataframe_head(df))

print("\nTail of DataFrame:")
print(CSVAnalysis.dataframe_tail(df))


#print("\nShape of DataFrame:")
#print(CSVAnalysis.dataframe_shape(df))


print("\nInfo of DataFrame:")
CSVAnalysis.dataframe_info(df)

print("\nDescribe DataFrame:")
print(CSVAnalysis.dataframe_describe(df))

print("\nData Types of DataFrame:")
print(CSVAnalysis.dataframe_dtypes(df))

# Plotting examples
CSVAnalysis.histplot(df, 'Age', output_path)
CSVAnalysis.boxplot(df, 'Sex', output_path)
CSVAnalysis.scatterplot(df, 'Age', 'Sex', output_path)
CSVAnalysis.heatmap(df, output_path)
CSVAnalysis.pairplot(df, output_path)
CSVAnalysis.lineplot(df, 'Age', 'Cholesterol', output_path)
CSVAnalysis.barplot(df, 'Age', 'HeartDisease', output_path)
CSVAnalysis.violinplot(df, 'Age', 'Cholesterol', output_path)
CSVAnalysis.density_plot(df, 'Cholesterol', output_path)


print(f"Plots have been saved in the '{output_path}' directory.")

heart_x = df.drop("HeartDisease",axis=1)
heart_y= df["HeartDisease"]

heart_x_encoded=pd.get_dummies(heart_x,drop_first=True)
heart_x_encoded= heart_x_encoded.astype(int)

X_train,X_test,y_train,y_test= train_test_split(heart_x_encoded,heart_y,test_size=0.3)

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,confusion_matrix

dtree= DecisionTreeClassifier(max_depth=2)
dtree.fit(X_train,y_train)
train_predictions=dtree.predict(X_train)
test_predictions=dtree.predict(X_test)
train_acc= accuracy_score(y_train,train_predictions)
test_acc= accuracy_score(y_test,test_predictions)
print('train_acc',train_acc)
print('test acc', test_acc)


import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, f1_score, precision_recall_fscore_support

# Compute predictions for test set probabilities and confusion matrix
test_probabilities = dtree.predict_proba(X_test)[:, 1]  # Probabilities for the positive class
test_predictions = dtree.predict(X_test)
cm = confusion_matrix(y_test, test_predictions)

# Generate and save confusion matrix plot
PostModelViz.confusion_matrix_plot(cm, output_path)

# Calculate ROC curve
fpr, tpr, _ = roc_curve(y_test, test_probabilities)
PostModelViz.roc_curve_plot(fpr, tpr, output_path)

# Calculate F1 scores
f1_scores = precision_recall_fscore_support(y_test, test_predictions, average=None)[2]
f1_scores_dict = {f'Class {i}': f1_scores[i] for i in range(len(f1_scores))}
PostModelViz.f1_score_plot(f1_scores_dict, output_path)

print(f"Post-model visualizations have been saved in the '{output_path}' directory.")
