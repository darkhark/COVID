import matplotlib.pyplot as plt
import seaborn as sns


def plotWithoutClusterSns(X_set, location, title):
    # sns settings
    sns.set(rc={'figure.figsize': (15,15)})

    # colors
    palette = sns.color_palette("bright", 1)

    # plot
    sns.scatterplot(X_set[:, 0], X_set[:, 1], palette=palette)

    plt.title(title)
    plt.savefig(location)
    plt.show()


def plotElbowForKmeans(kRange, distortions, name):
    X_line = [kRange[0], kRange[-1]]
    Y_line = [distortions[0], distortions[-1]]

    # Plot the elbow
    plt.plot(kRange, distortions, 'b-')
    plt.plot(X_line, Y_line, 'r')
    plt.xlabel('k')
    plt.ylabel('Distortion')
    plt.title('The Elbow Method showing the optimal k')
    location = "plot_pictures/elbows/OptimalKMeans" + name + ".png"
    plt.savefig(location)
    plt.show()


def plotSilhouetteScores(kRange, scores, name):
    X_line = [kRange[0], kRange[-1]]
    Y_line = [scores[0], scores[-1]]

    # Plot the elbow
    plt.plot(kRange, scores, 'b-')
    plt.plot(X_line, Y_line, 'r')
    plt.xlabel('k')
    plt.ylabel('Silhouette Scores')
    plt.title('Silhouette scores showing the optimal k')
    location = "plot_pictures/silhouettes/OptimalKMeans" + name + ".png"
    plt.savefig(location)
    plt.show()


def plotWithClusters(X, y_pred, k):
    # sns settings
    sns.set(rc={'figure.figsize': (15, 15)})

    # colors
    palette = sns.hls_palette(k, l=.4, s=.9)

    # plot
    sns.scatterplot(X[:, 0], X[:, 1], hue=y_pred, legend='full', palette=palette)
    title = "t-SNE with Kmeans " + str(k) + " Labels"
    plt.title(title)
    location = "plot_pictures/ClusteredTSNE" + str(k) + ".png"
    plt.savefig(location)
    plt.show()
