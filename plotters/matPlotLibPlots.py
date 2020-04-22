import matplotlib.pyplot as plt


def plotElbowForKmeans(kRange, distortions):
    X_line = [kRange[0], kRange[-1]]
    Y_line = [distortions[0], distortions[-1]]

    # Plot the elbow
    plt.plot(kRange, distortions, 'b-')
    plt.plot(X_line, Y_line, 'r')
    plt.xlabel('k')
    plt.ylabel('Distortion')
    plt.title('The Elbow Method showing the optimal k')
    plt.savefig("plot_pictures/elbows/OptimalKMeansTSNE.png")
    plt.show()


def plotSilhouetteScores(kRange, scores):
    X_line = [kRange[0], kRange[-1]]
    Y_line = [scores[0], scores[-1]]

    # Plot the elbow
    plt.plot(kRange, scores, 'b-')
    plt.plot(X_line, Y_line, 'r')
    plt.xlabel('k')
    plt.ylabel('Silhouette Scores')
    plt.title('Silhouette scores showing the optimal k')
    plt.savefig("plot_pictures/silhouettes/OptimalKMeansTSNE.png")
    plt.show()
