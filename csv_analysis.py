import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

class CSVAnalysis:
    @staticmethod
    def dataframe_head(merged_df):
        """Return the first 5 rows of the DataFrame."""
        return merged_df.head()

    @staticmethod
    def dataframe_tail(merged_df):
        """Return the last 5 rows of the DataFrame."""
        return merged_df.tail()

    @staticmethod
    def dataframe_shape(merged_df):
        """Return the shape of the DataFrame."""
        return merged_df.shape

    @staticmethod
    def dataframe_info(merged_df):
        """Print a concise summary of a DataFrame."""
        return merged_df.info()

    @staticmethod
    def dataframe_describe(merged_df):
        """Generate descriptive statistics."""
        return merged_df.describe()

    @staticmethod
    def dataframe_dtypes(merged_df):
        """Return the data types of the columns."""
        return merged_df.dtypes

    @staticmethod
    def save_fig(filename):
        """Save the current figure to the specified filename."""
        plt.savefig(filename)
        plt.close()

    @staticmethod
    def histplot(merged_df, column, filepath):
        """Create a histogram plot."""
        sns.histplot(merged_df[column])
        CSVAnalysis.save_fig(os.path.join(filepath, f'{column}_histplot.png'))

    @staticmethod
    def boxplot(merged_df, column, filepath):
        """Create a box plot."""
        sns.boxplot(y=merged_df[column])
        CSVAnalysis.save_fig(os.path.join(filepath, f'{column}_boxplot.png'))

    @staticmethod
    def scatterplot(merged_df, x, y, filepath):
        """Create a scatter plot."""
        sns.scatterplot(x=merged_df[x], y=merged_df[y])
        CSVAnalysis.save_fig(os.path.join(filepath, f'{x}_vs_{y}_scatterplot.png'))

    @staticmethod
    def heatmap(merged_df, filepath):
        """Create a heatmap."""
        numeric_df = merged_df.select_dtypes(include='number')
        sns.heatmap(numeric_df.corr(), annot=True)
        CSVAnalysis.save_fig(os.path.join(filepath, 'heatmap.png'))

    @staticmethod
    def pairplot(merged_df, filepath):
        """Create a pair plot."""
        sns.pairplot(merged_df)
        CSVAnalysis.save_fig(os.path.join(filepath, 'pairplot.png'))

    @staticmethod
    def lineplot(merged_df, x, y, filepath):
        """Create a line plot."""
        sns.lineplot(x=merged_df[x], y=merged_df[y])
        CSVAnalysis.save_fig(os.path.join(filepath, f'{x}_vs_{y}_lineplot.png'))

    @staticmethod
    def barplot(merged_df, x, y, filepath):
        """Create a bar plot."""
        sns.barplot(x=merged_df[x], y=merged_df[y])
        CSVAnalysis.save_fig(os.path.join(filepath, f'{x}_vs_{y}_barplot.png'))

    @staticmethod
    def violinplot(merged_df, x, y, filepath):
        """Create a violin plot."""
        sns.violinplot(x=merged_df[x], y=merged_df[y])
        CSVAnalysis.save_fig(os.path.join(filepath, f'{x}_vs_{y}_violinplot.png'))

    @staticmethod
    def density_plot(merged_df, column, filepath):
        """Create a density plot."""
        sns.kdeplot(merged_df[column], shade=True)
        CSVAnalysis.save_fig(os.path.join(filepath, f'{column}_densityplot.png'))

    @staticmethod
    def perform_eda(merged_df, filepath):
        """Perform comprehensive EDA and save plots."""
        # Print EDA summaries
        print("DataFrame Head:")
        print(CSVAnalysis.dataframe_head(merged_df))
        print("\nDataFrame Tail:")
        print(CSVAnalysis.dataframe_tail(merged_df))
        print("\nDataFrame Shape:")
        print(CSVAnalysis.dataframe_shape(merged_df))
        print("\nDataFrame Info:")
        CSVAnalysis.dataframe_info(merged_df)
        print("\nDataFrame Describe:")
        print(CSVAnalysis.dataframe_describe(merged_df))
        print("\nDataFrame Data Types:")
        print(CSVAnalysis.dataframe_dtypes(merged_df))

        # Generate plots
        for column in merged_df.select_dtypes(include='number').columns:
            CSVAnalysis.histplot(merged_df, column, filepath)
            CSVAnalysis.boxplot(merged_df, column, filepath)
            CSVAnalysis.density_plot(merged_df, column, filepath)

        numeric_columns = merged_df.select_dtypes(include='number').columns
        for i in range(len(numeric_columns)):
            for j in range(i+1, len(numeric_columns)):
                CSVAnalysis.scatterplot(merged_df, numeric_columns[i], numeric_columns[j], filepath)

        CSVAnalysis.heatmap(merged_df, filepath)
        CSVAnalysis.pairplot(merged_df, filepath)

    @staticmethod
    def interactive_plot_selection(merged_df, filepath):
        """Display available plots and allow user to select which to generate."""
        plot_options = {
            '1': 'histplot',
            '2': 'boxplot',
            '3': 'scatterplot',
            '4': 'heatmap',
            '5': 'pairplot',
            '6': 'lineplot',
            '7': 'barplot',
            '8': 'violinplot',
            '9': 'density_plot'
        }
        
        print("Available plots:")
        for key, value in plot_options.items():
            print(f"{key}: {value}")

        choice = input("Select a plot to generate (by number): ")

        if choice not in plot_options:
            print("Invalid choice.")
            return

        plot_type = plot_options[choice]

        if plot_type in ['histplot', 'boxplot', 'density_plot']:
            column = input(f"Enter the column name for {plot_type}: ")
            if column not in merged_df.columns:
                print(f"Column '{column}' does not exist in the DataFrame.")
                return
            getattr(CSVAnalysis, plot_type)(merged_df, column, filepath)

        elif plot_type == 'scatterplot':
            x = input("Enter the column name for x-axis: ")
            y = input("Enter the column name for y-axis: ")
            if x not in merged_df.columns or y not in merged_df.columns:
                print(f"One or both columns '{x}' and '{y}' do not exist in the DataFrame.")
                return
            CSVAnalysis.scatterplot(merged_df, x, y, filepath)

        elif plot_type == 'lineplot':
            x = input("Enter the column name for x-axis: ")
            y = input("Enter the column name for y-axis: ")
            if x not in merged_df.columns or y not in merged_df.columns:
                print(f"One or both columns '{x}' and '{y}' do not exist in the DataFrame.")
                return
            CSVAnalysis.lineplot(merged_df, x, y, filepath)

        elif plot_type == 'barplot':
            x = input("Enter the column name for x-axis: ")
            y = input("Enter the column name for y-axis: ")
            if x not in merged_df.columns or y not in merged_df.columns:
                print(f"One or both columns '{x}' and '{y}' do not exist in the DataFrame.")
                return
            CSVAnalysis.barplot(merged_df, x, y, filepath)

        elif plot_type == 'violinplot':
            x = input("Enter the column name for x-axis: ")
            y = input("Enter the column name for y-axis: ")
            if x not in merged_df.columns or y not in merged_df.columns:
                print(f"One or both columns '{x}' and '{y}' do not exist in the DataFrame.")
                return
            CSVAnalysis.violinplot(merged_df, x, y, filepath)

        elif plot_type == 'heatmap':
            CSVAnalysis.heatmap(merged_df, filepath)

        elif plot_type == 'pairplot':
            CSVAnalysis.pairplot(merged_df, filepath)
