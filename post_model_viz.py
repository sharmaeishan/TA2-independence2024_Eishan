import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd

class PostModelViz:

    @staticmethod
    def save_fig(filename):
        """Save the current figure to the specified filename."""
        plt.savefig(filename)
        plt.close()

    @staticmethod
    def residual_plot(residuals, filepath):
        """Plot residuals."""
        sns.residplot(x=residuals.index, y=residuals)
        PostModelViz.save_fig(os.path.join(filepath, 'residual_plot.png'))

    @staticmethod
    def cooks_distance_plot(cooks_d, filepath):
        """Plot Cook's distance."""
        sns.scatterplot(x=cooks_d.index, y=cooks_d)
        PostModelViz.save_fig(os.path.join(filepath, 'cooks_distance_plot.png'))

    @staticmethod
    def regression_line_plot(merged_df, x, y, filepath):
        """Plot regression line."""
        sns.regplot(x=x, y=y, data=merged_df)
        PostModelViz.save_fig(os.path.join(filepath, f'{x}_vs_{y}_regression_line.png'))

    @staticmethod
    def leverage_plot(leverage, filepath):
        """Plot leverage values."""
        sns.scatterplot(x=leverage.index, y=leverage)
        PostModelViz.save_fig(os.path.join(filepath, 'leverage_plot.png'))

    @staticmethod
    def confusion_matrix_plot(cm, filepath):
        """Plot confusion matrix."""
        sns.heatmap(cm, annot=True, fmt='d')
        PostModelViz.save_fig(os.path.join(filepath, 'confusion_matrix.png'))

    @staticmethod
    def learning_curve_plot(train_sizes, train_scores, test_scores, filepath):
        """Plot learning curve."""
        plt.plot(train_sizes, train_scores, label='Train Score')
        plt.plot(train_sizes, test_scores, label='Test Score')
        plt.xlabel('Training Sizes')
        plt.ylabel('Scores')
        plt.legend()
        PostModelViz.save_fig(os.path.join(filepath, 'learning_curve.png'))

    @staticmethod
    def roc_curve_plot(fpr, tpr, filepath):
        """Plot ROC curve."""
        plt.plot(fpr, tpr, marker='.')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        PostModelViz.save_fig(os.path.join(filepath, 'roc_curve.png'))

    @staticmethod
    def f1_score_plot(f1_scores, filepath):
        """Plot F1 scores."""
        sns.barplot(x=list(f1_scores.keys()), y=list(f1_scores.values()))
        PostModelViz.save_fig(os.path.join(filepath, 'f1_scores.png'))

    @staticmethod
    def interactive_plot_selection(output_path, data):
        """Display available plots and let the user choose one to generate."""
        plot_options = {
            '1': 'residual_plot',
            '2': 'cooks_distance_plot',
            '3': 'leverage_plot',
            '4': 'confusion_matrix_plot',
            '5': 'learning_curve_plot',
            '6': 'roc_curve_plot',
            '7': 'f1_score_plot',
            '8': 'regression_line_plot'
        }

        print("Available post-model visualizations:")
        for key, value in plot_options.items():
            print(f"{key}: {value}")

        choice = input("Select a plot to generate (by number): ")

        if choice not in plot_options:
            print("Invalid choice.")
            return

        plot_type = plot_options[choice]

        # Extract relevant data for the selected plot
        if plot_type == 'residual_plot':
            residuals = data['residuals']
            PostModelViz.residual_plot(residuals, output_path)
        elif plot_type == 'cooks_distance_plot':
            cooks_d = data['cooks_d']
            PostModelViz.cooks_distance_plot(cooks_d, output_path)
        elif plot_type == 'leverage_plot':
            leverage = data['leverage']
            PostModelViz.leverage_plot(leverage, output_path)
        elif plot_type == 'confusion_matrix_plot':
            cm = data['cm']
            PostModelViz.confusion_matrix_plot(cm, output_path)
        elif plot_type == 'learning_curve_plot':
            train_sizes = data['train_sizes']
            train_scores = data['train_scores']
            test_scores = data['test_scores']
            PostModelViz.learning_curve_plot(train_sizes, train_scores, test_scores, output_path)
        elif plot_type == 'roc_curve_plot':
            fpr = data['fpr']
            tpr = data['tpr']
            PostModelViz.roc_curve_plot(fpr, tpr, output_path)
        elif plot_type == 'f1_score_plot':
            f1_scores = data['f1_scores']
            PostModelViz.f1_score_plot(f1_scores, output_path)
        elif plot_type == 'regression_line_plot':
            merged_df = data['regression_line_data']
            x = data['x']
            y = data['y']
            PostModelViz.regression_line_plot(merged_df, x, y, output_path)

        print(f"Plot '{plot_type}' has been generated and saved in the '{output_path}' directory.")
