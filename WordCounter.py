def addAbstractAndBodyWordCountColumn(df_covid):
    """
    Counts every word in the abstract and body and appends it to the row
    that the count belongs to.

    :param df_covid: Cleansed dataframe to avoid counting
    :return: A dataframe with the word counts appended to the end.
    """
    df_covid['abstract_word_count'] = df_covid['abstract'].apply(lambda x: len(x.strip().split()))
    df_covid['body_word_count'] = df_covid['body_text'].apply(lambda x: len(x.strip().split()))
    return df_covid

