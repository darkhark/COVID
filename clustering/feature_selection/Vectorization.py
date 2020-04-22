from sklearn.feature_extraction.text import HashingVectorizer,TfidfVectorizer
from sklearn.manifold import TSNE
import pandas as pd
from sklearn.decomposition import PCA


def getHashVectorizationMatrix(data):
    # The lambda here is like a for each for iterating through a list of lists
    hvec = HashingVectorizer(lowercase=False, analyzer=lambda l: l, n_features=2 ** 12)
    return hvec.fit_transform(data)


"""
https://www.datacamp.com/community/tutorials/introduction-t-sne
"""
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
def reduceDimensionalityWithTSNE(X_set, neighbors):
    print("Reducing dimensionality using TSNE...")
    # t-distributed Stochastic Neighbor Embedding
    # -1 n_jobs means use all the processors to try and increase speed
    tsne = TSNE(verbose=1, perplexity=neighbors, n_jobs=-1, random_state=42)
    return tsne.fit_transform(X_set)


def reduceDimensionalityWithTF_IDF(cleansedDF, calculateMaxFeatures=True):
    """
    TF-IDF measures words based on their uniqueness. The maximum number of features will be equal to
    the maximum nymber of unique words present in an article.

    :param calculateMaxFeatures: If true, the maximum number of features will be calculated
    :param cleansedDF: The cleaned Data Frame. Only the body_text column will be used.
    :return: Tf-idf-weighted document-term matrix.
    """
    print("Getting max unique body words count for maximum features...")
    body_text_values = cleansedDF["body_text"].values
    maxFeatures = 2 ** 13
    if calculateMaxFeatures:
        uniqueBodyWordsCount = pd.Series()
        uniqueBodyWordsCount["unique_body_words_count"] = cleansedDF['body_text'].apply(lambda x: len(set(str(x).split())))
        maxFeatures = uniqueBodyWordsCount["unique_body_words_count"].max()
    print("Reducing dimensionality using TF-IDF...")
    vectorizer = TfidfVectorizer(max_features=maxFeatures)
    return vectorizer.fit_transform(body_text_values)


def reduceDimensionalityWithPCA(X):
    """
    Uses Principle Component Analysis to reduce the dimensionality of our data.

    :param X: A matrix of transformed data
    :return: A reduced matrix
    """
    print("Reducing dimensionality using PCA...")
    pca = PCA(n_components=0.95, random_state=42, svd_solver="full")
    return pca.fit_transform(X.toarray())
