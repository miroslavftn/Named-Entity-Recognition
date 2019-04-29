import os
import urllib
import zipfile

from nltk.tag import StanfordNERTagger
from nltk.tokenize import word_tokenize
from orderedset import OrderedSet

from configs import BASE_DIR
from pre_trained.base_ner import BaseNER


class StanfordNER(BaseNER):
    def __init__(self):
        super().__init__()

        self.stanford_data_url = "https://nlp.stanford.edu/software/stanford-ner-2018-10-16.zip"
        self.model_name = 'english.muc.7class.distsim.crf.ser.gz'
        self.jar_name = 'stanford-ner.jar'
        self.filename = urllib.parse.urlparse(self.stanford_data_url).path.rsplit("/")[-1]

        stanford_data_dir = os.path.join(BASE_DIR, 'entities/stanford_ner/stanford_data')
        filepath = os.path.join(stanford_data_dir, self.filename)
        if not os.path.exists(stanford_data_dir):
            print("Can't find data for Stanford NER on path '{}'".format(stanford_data_dir))
            os.mkdir(stanford_data_dir)
            download(self.stanford_data_url, filepath)
            with zipfile.ZipFile(filepath, "r") as zip_ref:
                zip_ref.extractall(stanford_data_dir)
        else:
            with zipfile.ZipFile(filepath, "r") as zip_ref:
                zip_ref.extractall(stanford_data_dir)

        extracted_files_path = os.path.splitext(self.filename)[0]
        classification_model_path = os.path.join(stanford_data_dir, extracted_files_path, 'classifiers',
                                                 self.model_name)
        tagger_jar_path = os.path.join(stanford_data_dir, extracted_files_path, self.jar_name)
        self.st = StanfordNERTagger(classification_model_path, tagger_jar_path, encoding='utf-8')

    def __repr__(self):
        return "Stanford"

    def entities(self, text, types=[]):
        tokenized_text = word_tokenize(text)
        entities = self.st.tag(tokenized_text)

        if types:
            entities = OrderedSet([ent for ent in entities if ent[1] in types])

        return list(entities)


def download(url, filename):
    print("Downloading from '{}' to '{}'".format(url, filename))
    urllib.request.urlretrieve(url, filename)
    print("Downloading finished!")


if __name__ == '__main__':
    ner_model = StanfordNER()
    res = ner_model.entities("Did Uriah honestly think he could beat The Legend of Zelda in under three hours?")

    print(res)
