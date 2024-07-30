from csv_analysis import CSVAnalysis 
import pandas as pd
import os

# Create a sample DataFrame
data = {
    'A': [1, 2, 3, 4, 5],
    'B': [5, 4, 3, 2, 1],
    'C': [2, 3, 4, 5, 6],
    'D': [5, 6, 7, 8, 9]
}
df = pd.DataFrame(data)

# Define a filepath for saving plots
output_path = '/TA2-independence2024_Eishan/'
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
CSVAnalysis.histplot(df, 'A', output_path)
CSVAnalysis.boxplot(df, 'B', output_path)
CSVAnalysis.scatterplot(df, 'A', 'B', output_path)
CSVAnalysis.heatmap(df, output_path)
CSVAnalysis.pairplot(df, output_path)
CSVAnalysis.lineplot(df, 'A', 'C', output_path)
CSVAnalysis.barplot(df, 'A', 'D', output_path)
CSVAnalysis.violinplot(df, 'A', 'C', output_path)
CSVAnalysis.density_plot(df, 'C', output_path)

print(f"Plots have been saved in the '{output_path}' directory.")
