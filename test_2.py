from csv_analysis import CSVAnalysis 
from post_model_viz import PostModelViz
import pandas as pd
import os

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
