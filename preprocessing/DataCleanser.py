from nltk.corpus import stopwords
from tqdm import tqdm
from langdetect import detect
from langdetect import DetectorFactory
from pprint import pprint

#nltk.download('stopwords')  //Uncomment if need to download

import re
import numpy as np


def handleEmptyData(df_covid):
    """
    Removes any rows that contain a null value. We currently create a string 'no abstract provided' if the abstract
    is not present, so this method is to account for potential mishandled abstract entries.  For all other columns
    we convert empty strings to NaN, but only drop rows that do not have either a body or author

    Should be completed before running other methods to reduce the number of rows examined.

    :param df_covid: Dataframe of covid data.
    :return: Data frame without any null values.
    """
    i=0
    for col in df_covid.columns:
        if col == 'abstract':
            df_covid['abstract'].replace('', 'abstract missing', inplace=True)
            i+=1
        else:
            df_covid[col].replace('', np.nan, inplace=True)
            i+=1
    df_covid.dropna(subset=['body_text', 'authors'], inplace=True)
    print("\nDrop Nulls\n")
    print("Empty Data handled:", i)
    print(df_covid["body_text"])
    return df_covid


def removeDuplicates(df_covid):
    """
    Removes any duplicate journals in case the journal was submitted to multiple companies.

    :param df_covid: All the data loaded in.
    :return: Dataframe without duplicate bodies or abstracts.
    """
    df_covid.drop_duplicates(['abstract', 'body_text'], inplace=True)
    print("\nDrop Duplicates\n")
    print(df_covid["body_text"])
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
    print("\nDrop Punctuation\n")
    print(df_covid["body_text"])
    return df_covid


def removeStoppingWords(df_covid):
    """
    Removes stopping words from strings
    
    :param df_covid: All the data loaded in
    :return: A dataframe with all stopping words removed from body and abstract
    """
    stop = stopwords.words('english')
    pat = r'\b(?:{})\b'.format('|'.join(stop))
    df_covid['body_text'] = df_covid['body_text'].apply(lambda x: re.sub(pat, '', x))
    df_covid['abstract'] = df_covid['abstract'].apply(lambda x: re.sub(pat, '', x))
    df_covid['body_text'] = df_covid['body_text'].apply(lambda x: replaceExcessiveSpacing(x))
    df_covid['abstract'] = df_covid['abstract'].apply(lambda x: replaceExcessiveSpacing(x))
    print('\nRemove Stopping Words\n')
    print(df_covid['body_text'])
    return df_covid


def replaceExcessiveSpacing(fullText):
    while "  " in fullText:
        fullText = fullText.replace("  ", " ")
    return fullText


def convertDataToLowercase(df_covid):
    """
    Converts all text in the body and abstract to lower case. This will prevent "Covid" and "covid" from being detected
    as different strings.

    :param df_covid: All of the data loaded in.
    :return: A dataframe where all text has been converted to lowercase.
    """
    df_covid['body_text'] = df_covid['body_text'].apply(lambda x: toLowercase(x))
    df_covid['abstract'] = df_covid['abstract'].apply(lambda x: toLowercase(x))
    print("\nReplace case\n")
    print(df_covid["body_text"])
    return df_covid


def toLowercase(input_str):
    """
    Converts a given string to all lowercase

    :param input_str: String to be converted to lowercase
    :return: The string with no uppercase
    """
    input_str = input_str.lower()
    return input_str

def languageDetection(df_covid):
    """
    Detects the language of the research articles and currently separates
    English-language articles to be used in further analysis

    :param df_covid: dataframe containing all entries
    :return: dataframe cleansed of non-english articles
    """
    DetectorFactory.seed = 0
    languages = []

    # go through each text
    for i in tqdm(range(0,len(df_covid))):
    # split by space into list, take the first x intex, join with space
        text = df_covid.iloc[i]['body_text'].split(" ")
        lang = "en"
        
        try:
            if len(text) > 50:
                lang = detect(" ".join(text[:50]))
                
            elif len(text) > 0:
                lang = detect(" ".join(text[:len(text)]))
                 # ught... beginning of the document was not in a good format
        except Exception as e:
            all_words = set(text)
            try:
                lang = detect(" ".join(all_words))
                # what!! :( let's see if we can find any text in abstract...
            except Exception as e:
                try:
                # let's try to label it through the abstract then
                    lang = detect(df_covid.iloc[i]['abstract_summary'])
                except Exception as e:
                    lang = "unknown"
                    pass
    
        # get the language    
        languages.append(lang)
        
    languages_dict = {}
    for lang in set(languages):
        languages_dict[lang] = languages.count(lang)
    
    print("Total: {}\n".format(len(languages)))
    pprint(languages_dict)
    df_covid['languages'] = languages
    df_covid=df_covid[df_covid['languages'] == 'en']
    df_covid.info()
    
    return df_covid

def runDataCleanser(df_covid, saveToCSV=False):
    """
    Cleans the data removing duplicates, nulls, and punctuations.
    It then converts all strings to lowercase.

    :param saveToCSV: If set to true, saved a csv of the cleansed dataframe.
    :param df_covid: All the data.
    :return: A cleaner dataframe.
    """
    df = languageDetection(df_covid)
    df = handleEmptyData(df)
    df = removeDuplicates(df)
    df = removePunctuation(df)
    df = convertDataToLowercase(df)
    df = removeStoppingWords(df)
    if saveToCSV:
        df.to_csv(path_or_buf="cleanedData/cleanedData.csv", index=False)
        print("Saved CSV to cleanedData/otherCleanedData.csv")
    return df
