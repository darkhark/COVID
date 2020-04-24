from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation


def selected_topics(model, vectorizer, top_n=3):
    current_words = []
    keywords = []

    for idx, topic in enumerate(model.components_):
        words = [(vectorizer.get_feature_names()[i], topic[i]) for i in topic.argsort()[:-top_n - 1:-1]]
        for word in words:
            if word[0] not in current_words:
                keywords.append(word)
                current_words.append(word[0])

    keywords.sort(key=lambda x: x[1])
    keywords.reverse()
    return_values = []
    for ii in keywords:
        return_values.append(ii[0])
    return return_values


def getLDAModelsAndKeywords(covidDF):
    vectorizers = []

    for i in range(0, 17):
        # Creating a vectorizer
        vectorizers.append(CountVectorizer(min_df=5, max_df=0.9, stop_words='english', lowercase=True,
                                           token_pattern='[a-zA-Z\-][a-zA-Z\-]{2,}'))

    vectorized_data = []

    for current_cluster, cvec in enumerate(vectorizers):
        try:
            vectorized_data.append(cvec.fit_transform(covidDF.loc[covidDF['y'] == current_cluster, 'body_text']))
        except Exception as e:
            print("Not enough instances in cluster: " + str(current_cluster))
            vectorized_data.append(None)

    NUM_TOPICS_PER_CLUSTER = 20

    lda_models = []
    for i in range(0, 20):
        # Latent Dirichlet Allocation Model
        lda = LatentDirichletAllocation(n_components=NUM_TOPICS_PER_CLUSTER, max_iter=10, learning_method='online',
                                        verbose=False, random_state=42)
        lda_models.append(lda)

    clusters_lda_data = []

    for current_cluster, lda in enumerate(lda_models):

        if vectorized_data[current_cluster] is not None:
            clusters_lda_data.append((lda.fit_transform(vectorized_data[current_cluster])))

    all_keywords = []
    for current_vectorizer, lda in enumerate(lda_models):

        if vectorized_data[current_vectorizer] is not None:
            all_keywords.append(selected_topics(lda, vectorizers[current_vectorizer]))

    return vectorized_data, all_keywords
