from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.manifold import TSNE


def getHashVectorizationMatrix(data):
    # The lambda here is like a for each
    hvec = HashingVectorizer(lowercase=False, analyzer=lambda l: l, n_features=2 ** 12)
    return hvec.fit_transform(data)


"""
From sklearn: https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html
t-SNE [1] is a tool to visualize high-dimensional data. It converts similarities between data points to joint
 probabilities and tries to minimize the Kullback-Leibler divergence between the joint probabilities of the 
 low-dimensional embedding and the high-dimensional data. t-SNE has a cost function that is not convex,
  i.e. with different initializations we can get different results.

It is highly recommended to use another dimensionality reduction method (e.g. PCA for dense data or TruncatedSVD
 for sparse data) to reduce the number of dimensions to a reasonable amount (e.g. 50) if the number of features is 
 very high. This will suppress some noise and speed up the computation of pairwise distances between samples.
"""
def reduceDimensionality(X_set):
    print("Reducing dimensionality using TSNE...")
    # t-distributed Stochastic Neighbor Embedding
    tsne = TSNE(verbose=1, perplexity=5)
    return tsne.fit_transform(X_set)
