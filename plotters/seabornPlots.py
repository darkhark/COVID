from matplotlib import pyplot as plt
import seaborn as sns


def plotWithoutClusterSns(X_set):
    # sns settings
    sns.set(rc={'figure.figsize': (15,15)})

    # colors
    palette = sns.color_palette("bright", 1)

    # plot
    sns.scatterplot(X_set[:,0], X_set[:,1], palette=palette)

    plt.title("t-SNE Covid-19 Articles")
    plt.savefig("plot_pictures/ngramsPlot.png")
    plt.show()
