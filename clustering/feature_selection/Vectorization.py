from sklearn.feature_extraction.text import HashingVectorizer


def getHashVectorizationModel(data):
    # The lambda here is like a for each
    hvec = HashingVectorizer(lowercase=False, analyzer=lambda l: l, n_features=2 ** 12)
    return hvec.fit_transform(data)

