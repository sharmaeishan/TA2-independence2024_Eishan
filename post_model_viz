import matplotlib.pyplot as plt
import seaborn as sns
import os

class PostModelViz:

    def save_fig(filename):
        """Save the current figure to the specified filename."""
        plt.savefig(filename)
        plt.close()

    def residual_plot(residuals, filepath):
        """Plot residuals."""
        sns.residplot(x=residuals.index, y=residuals)
        PostModelViz.save_fig(os.path.join(filepath, 'residual_plot.png'))

    def cooks_distance_plot(cooks_d, filepath):
        """Plot Cook's distance."""
        sns.scatterplot(x=cooks_d.index, y=cooks_d)
        PostModelViz.save_fig(os.path.join(filepath, 'cooks_distance_plot.png'))

    def regression_line_plot(merged_df, x, y, filepath):
        """Plot regression line."""
        sns.regplot(x=x, y=y, data=merged_df)
        PostModelViz.save_fig(os.path.join(filepath, f'{x}_vs_{y}_regression_line.png'))

    def leverage_plot(leverage, filepath):
        """Plot leverage values."""
        sns.scatterplot(x=leverage.index, y=leverage)
        PostModelViz.save_fig(os.path.join(filepath, 'leverage_plot.png'))

    def confusion_matrix_plot(cm, filepath):
        """Plot confusion matrix."""
        sns.heatmap(cm, annot=True, fmt='d')
        PostModelViz.save_fig(os.path.join(filepath, 'confusion_matrix.png'))

    def learning_curve_plot(train_sizes, train_scores, test_scores, filepath):
        """Plot learning curve."""
        plt.plot(train_sizes, train_scores, label='Train Score')
        plt.plot(train_sizes, test_scores, label='Test Score')
        plt.xlabel('Training Sizes')
        plt.ylabel('Scores')
        plt.legend()
        PostModelViz.save_fig(os.path.join(filepath, 'learning_curve.png'))

    def roc_curve_plot(fpr, tpr, filepath):
        """Plot ROC curve."""
        plt.plot(fpr, tpr, marker='.')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        PostModelViz.save_fig(os.path.join(filepath, 'roc_curve.png'))

    def f1_score_plot(f1_scores, filepath):
        """Plot F1 scores."""
        sns.barplot(x=list(f1_scores.keys()), y=list(f1_scores.values()))
        PostModelViz.save_fig(os.path.join(filepath, 'f1_scores.png'))
