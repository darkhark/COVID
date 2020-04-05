import re


def dropNulls(df_covid):
    """
    Removes any rows that contain a null value. We print out the abstract description here because we know there are
    null value in there. We made the abstract a None type whenever no abstract was present in a JSON file.

    Should be completed before running other methods to reduce the number of rows examined.

    :param df_covid: Dataframe of covid data.
    :return: Data frame without any null values.
    """
    df_covid.dropna(inplace=True)
    return df_covid


def removeDuplicates(df_covid):
    """
    Removes any duplicate journals in case the journal was submitted to multiple companies.

    :param df_covid: All the data loaded in.
    :return: Dataframe without duplicate bodies or abstracts.
    """
    df_covid.drop_duplicates(['abstract', 'body_text'], inplace=True)
    return df_covid


def removePunctuation(df_covid):
    """
    Removes punctuation from strings to help better match words, otherwise "covid" will not match with "covid." and
    could lead to inaccurate groupings.

    :param df_covid: All the data loaded in.
    :return: A dataframe with all the punctuation removed from the body and abstract.
    """
    df_covid['body_text'] = df_covid['body_text'].apply(lambda x: re.sub('[^a-zA-z0-9\s]', '', x))
    df_covid['abstract'] = df_covid['abstract'].apply(lambda x: re.sub('[^a-zA-z0-9\s]', '', x))
    return df_covid


def convertDataToLowercase(df_covid):
    """
    Converts all text in the body and abstract to lower case. This will prevent "Covid" and "covid" from being detected
    as different strings.

    :param df_covid: All of the data loaded in.
    :return: A dataframe where all text has been converted to lowercase.
    """
    df_covid['body_text'] = df_covid['body_text'].apply(lambda x: toLowercase(x))
    df_covid['abstract'] = df_covid['abstract'].apply(lambda x: toLowercase(x))
    return df_covid


def toLowercase(input_str):
    """
    Converts a given string to all lowercase

    :param input_str: String to be converted to lowercase
    :return: The string with no uppercase
    """
    input_str = input_str.lower()
    return input_str


def runDataCleanser(df_covid):
    """
    Cleans the data removing duplicates, nulls, and punctuations.
    It then converts all strings to lowercase.

    :param df_covid: All the data.
    :return: A cleaner dataframe.
    """
    df = removeDuplicates(df_covid)
    df = removePunctuation(df)
    return convertDataToLowercase(df)
