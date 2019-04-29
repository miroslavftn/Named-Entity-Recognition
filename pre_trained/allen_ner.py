import os
import urllib.request

from allennlp.predictors.predictor import Predictor
from orderedset import OrderedSet

from configs import BASE_DIR
from pre_trained.base_ner import BaseNER


class AllenNER(BaseNER):
    def __init__(self, load_local=False):
        super().__init__()
        self.load_local = load_local
        self.model_url = "https://s3-us-west-2.amazonaws.com/allennlp/models/ner-model-2018.12.18.tar.gz"
        self.filename = urllib.parse.urlparse(self.model_url).path.rsplit("/")[-1]

        allen_data_dir = os.path.join(BASE_DIR, 'entities/allen_ner/allen_data')
        filepath = os.path.join(allen_data_dir, self.filename)
        if not os.path.exists(allen_data_dir) and self.load_local:
            print("Can't find model on path '{}'".format(allen_data_dir))
            os.mkdir(allen_data_dir)
            download(self.model_url, filepath)

        if self.load_local:
            self.predictor = Predictor.from_path(filepath)
        else:
            self.predictor = Predictor.from_path(self.model_url)

    def __repr__(self):
        return "Allen"

    def entities(self, text, types=[]):
        predictions = self.predictor.predict(sentence=text)
        entities = list(zip(predictions.get("words"), predictions.get('tags')))

        if types:
            entities = OrderedSet([ent for ent in entities if ent[1] in types])

        return list(entities)


def download(url, filename):
    print("Downloading from '{}' to '{}'".format(url, filename))
    urllib.request.urlretrieve(url, filename)
    print("Downloading finished!")


if __name__ == "__main__":
    ner = AllenNER()
    res = ner.entities("Did Uriah honestly think he could beat The Legend of Zelda in under three hours?")

    print(res)
