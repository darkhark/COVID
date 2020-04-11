"""
Tutorial found here: https://programminghistorian.org/en/lessons/keywords-in-context-using-n-grams
"""


def splitWordsIntoColumns(dataframe):
    """
    Breaks up the body text so that each word becomes it's own column.
    :param dataframe: The cleansed dataframe

    :return: A two dimensional array where each row is a body and each column is a different word of the body.
    """

    text = dataframe.drop(["paper_id", "abstract", "abstract_word_count", "body_word_count", "authors", "title",
                           "journal", "abstract_summary"], axis=1)
    words = []
    for i in range(0, len(text)):
        words.append(str(text.iloc[i]['body_text']).split(" "))
    # for i in range(0, 5):
    #     print(words[i][:10])
    return words


def reduceToNGrams(wordsArray, n):
    """
    Reduces the given text to the ngrams specified. The ngrams are the number of words combined
    into one. For example, if the ngram is 3 for this sentence, "hello world i feel alive", the
    first two ngrams will be helloworldi, worldifeel.

    This application of n-grams is known as keywords in context (often abbreviated as KWIC), which
    helps match words based on the surrounding words.

    :param wordsArray: A two dimensional array where the rows are different bodies of text and columns
    are the different words.
    :param n: The number of neighboring words to be used for context.
    :return: An array where each row is the ngrams for a body and each column is a different ngram.
    """
    allNGrams = []
    for word in wordsArray:
        ngram = []
        for i in range(len(word) - n + 1):
            ngram.append("".join(word[i:i + n]))
        allNGrams.append(ngram)
    for i in range(0, 5):
        print(allNGrams[i][:10])
    return allNGrams


def runNGrams(df, n):
    return reduceToNGrams(splitWordsIntoColumns(df), n)
