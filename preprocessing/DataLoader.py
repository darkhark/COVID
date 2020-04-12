import pandas as pd
import glob
import json


class FileReader:
    def __init__(self, file_path):
        with open(file_path) as file:
            content = json.load(file)
            self.paper_id = content['paper_id']
            self.abstract = []
            self.body_text = []
            if "abstract" in content:
                for entry in content['abstract']:
                    self.abstract.append(entry['text'])
            else:
                content['abstract'] = ''
            for entry in content['body_text']:
                self.body_text.append(entry['text'])
            self.abstract = '\n'.join(self.abstract)
            self.body_text = '\n'.join(self.body_text)

    def __repr__(self):
        return f'{self.paper_id}: {self.abstract[:200]}... {self.body_text[:200]}...'


def getAllData():
    """
    Loads all the metadata from metadata.csv.

    :return: A list of all the json files and a dataframe with all the metadata.
    """
    root_path = "data"
    metadata_path = f'{root_path}/metadata.csv'
    meta_df = pd.read_csv(metadata_path, dtype=object)

    all_json = glob.glob(f'{root_path}/**/*.json', recursive=True)
    print("Number of JSON files", len(all_json))
    return all_json, meta_df


def get_breaks(content, length):
    data = ""
    words = content.split(' ')
    total_chars = 0

    # add break every length characters
    for i in range(len(words)):
        total_chars += len(words[i])
        if total_chars > length:
            data = data + "<br>" + words[i]
            total_chars = 0
        else:
            data = data + " " + words[i]
    return data


def getDataFrame(all_json, meta_df):
    """
    :param all_json: A list of all the JSON files files
    :param meta_df: Dataframe of all the metadata from the JSON files.
    :return: Data frame of all the covid data.
    """
    dict_ = {'paper_id': [], 'abstract': [], 'body_text': [], 'authors': [], 'title': [], 'journal': [],
             'abstract_summary': []}
    for idx, entry in enumerate(all_json):
        if idx % (len(all_json) // 10) == 0:
            print(f'Processing index: {idx} of {len(all_json)}')
        content = FileReader(entry)

        # get metadata information
        meta_data = meta_df.loc[meta_df['sha'] == content.paper_id]
        # no metadata, skip this paper
        if len(meta_data) == 0:
            continue

        dict_['paper_id'].append(str(content.paper_id).strip())
        dict_['abstract'].append(str(content.abstract).strip())
        dict_['body_text'].append(str(content.body_text).strip())

        # also create a column for the summary of abstract to be used in a plot
        if len(content.abstract) == 0:
            # no abstract provided
            dict_['abstract_summary'].append("Not provided.")
        elif len(content.abstract.split(' ')) > 100:
            # abstract provided is too long for plot, take first 300 words append with ...
            info = content.abstract.split(' ')[:100]
            summary = get_breaks(' '.join(info), 40)
            dict_['abstract_summary'].append(summary + "...")
        else:
            # abstract is short enough
            summary = get_breaks(content.abstract, 40)
            dict_['abstract_summary'].append(summary)

        # get metadata information
        meta_data = meta_df.loc[meta_df['sha'] == content.paper_id]

        try:
            # if more than one author
            authors = meta_data['authors'].values[0].split(';')
            if len(authors) > 2:
                # more than 2 authors, may be problem when plotting, so take first 2 append with ...
                dict_['authors'].append(str(". ".join(authors[:2]) + "...").strip())
            else:
                # authors will fit in plot
                dict_['authors'].append(str(". ".join(authors)).strip())
        except Exception as e:
            # if only one author - or Null value
            dict_['authors'].append(str(meta_data['authors'].values[0]).strip())

        # add the title information, add breaks when needed
        try:
            title = get_breaks(meta_data['title'].values[0], 40)
            dict_['title'].append(str(title).strip())
        # if title was not provided
        except Exception as e:
            dict_['title'].append(str(meta_data['title'].values[0]).strip())

        # add the journal information
        dict_['journal'].append(str(meta_data['journal'].values[0]).strip())

    df_covid = pd.DataFrame(dict_, columns=['paper_id', 'abstract', 'body_text', 'authors', 'title', 'journal',
                                            'abstract_summary'])
    return df_covid


def runDataLoader():
    all_json, metadata = getAllData()
    return getDataFrame(all_json, metadata)


def runQuickLoader(numFilesToLoad):
    all_json, metadata = getAllData()
    return getDataFrame(all_json[:numFilesToLoad], metadata)
