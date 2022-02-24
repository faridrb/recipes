import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix


def pie_chart(value_counts: pd.Series):
    total_count = value_counts.sum()

    def fmt(x):
        return f'{total_count * x / 100:.0f} ({x:.2f}%)'

    plt.figure(figsize=(10, 10))
    plt.pie(value_counts, labels=value_counts.index, autopct=fmt, colors=sns.color_palette('crest'))
    plt.title(f'Total amount of data: {total_count}')
    plt.show()


def plot_confusion_matrix(y_test, preds, labels):
    with sns.plotting_context("notebook"):
        cm = confusion_matrix(y_test, preds, normalize='pred')
        plt.figure(figsize=(12, 10))
        a = sns.heatmap(cm, vmin=0, vmax=1, xticklabels=labels, yticklabels=labels)
        a.set_xlabel('Prediction')
        a.set_ylabel('True value')
        a.set_title('Normalized confusion matrix')


def plot_training_metrics(training, metric):
    plt.plot(training.history[metric])
    plt.plot(training.history[f'val_{metric}'])
    plt.xlabel('Epochs')
    plt.ylabel(metric)
    plt.legend([metric, f'val_{metric}'])
    plt.show()
