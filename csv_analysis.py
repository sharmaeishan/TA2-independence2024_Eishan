import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

class CSVAnalysis:
    def dataframe_head(merged_df):
        """Return the first 5 rows of the DataFrame."""
        return merged_df.head()

    def dataframe_tail(merged_df):
        """Return the last 5 rows of the DataFrame."""
        return merged_df.tail()

    def dataframe_shape(merged_df):
        """Return the shape of the DataFrame."""
        return merged_df.shape()

    def dataframe_info(merged_df):
        """Print a concise summary of a DataFrame."""
        return merged_df.info()

    def dataframe_describe(merged_df):
        """Generate descriptive statistics."""
        return merged_df.describe()

    def dataframe_dtypes(merged_df):
        """Return the data types of the columns."""
        return merged_df.dtypes

    def save_fig(filename):
        """Save the current figure to the specified filename."""
        plt.savefig(filename)
        plt.close()

    def histplot(merged_df, column, filepath):
        """Create a histogram plot."""
        sns.histplot(merged_df[column])
        CSVAnalysis.save_fig(os.path.join(filepath, f'{column}_histplot.png'))

    def boxplot(merged_df, column, filepath):
        """Create a box plot."""
        sns.boxplot(y=merged_df[column])
        CSVAnalysis.save_fig(os.path.join(filepath, f'{column}_boxplot.png'))

    def scatterplot(merged_df, x, y, filepath):
        """Create a scatter plot."""
        sns.scatterplot(x=merged_df[x], y=merged_df[y])
        CSVAnalysis.save_fig(os.path.join(filepath, f'{x}_vs_{y}_scatterplot.png'))

    def heatmap(merged_df, filepath):
        """Create a heatmap."""
        sns.heatmap(merged_df.corr(), annot=True)
        CSVAnalysis.save_fig(os.path.join(filepath, 'heatmap.png'))

    def pairplot(merged_df, filepath):
        """Create a pair plot."""
        sns.pairplot(merged_df)
        CSVAnalysis.save_fig(os.path.join(filepath, 'pairplot.png'))

    def lineplot(merged_df, x, y, filepath):
        """Create a line plot."""
        sns.lineplot(x=merged_df[x], y=merged_df[y])
        CSVAnalysis.save_fig(os.path.join(filepath, f'{x}_vs_{y}_lineplot.png'))

    def barplot(merged_df, x, y, filepath):
        """Create a bar plot."""
        sns.barplot(x=merged_df[x], y=merged_df[y])
        CSVAnalysis.save_fig(os.path.join(filepath, f'{x}_vs_{y}_barplot.png'))

    def violinplot(merged_df, x, y, filepath):
        """Create a violin plot."""
        sns.violinplot(x=merged_df[x], y=merged_df[y])
        CSVAnalysis.save_fig(os.path.join(filepath, f'{x}_vs_{y}_violinplot.png'))

    def density_plot(merged_df, column, filepath):
        """Create a density plot."""
        sns.kdeplot(merged_df[column], shade=True)
        CSVAnalysis.save_fig(os.path.join(filepath, f'{column}_densityplot.png'))



    
