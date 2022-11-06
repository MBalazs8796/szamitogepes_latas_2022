import json
import os
import seaborn as sns
import numpy as np

from matplotlib import pyplot as plt


def create_plot(result, plot_name):
    if not os.path.isdir('./figs'):
        os.mkdir('./figs')

    plot = sns.barplot(x=np.arange(len(result)), y=result, palette='Blues')
    plot.set(xticklabels=[])
    plt.savefig(f'./figs/{plot_name}.png')
    plt.clf()


if __name__ == '__main__':
    for root, dirs, files in os.walk('./results'):
        for file in files:
            if not file == 'results-binary.json':
                with open(os.path.join(root, file), 'r') as f:
                    result = json.load(f)
                    plot_name = file.split('results_')[-1].split('.')[0]
                    create_plot(result['norms'], plot_name)
