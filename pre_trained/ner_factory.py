from pre_trained.allen_ner import AllenNER
from pre_trained.pavlov_ner import PavlovNER
from pre_trained.spacy_ner import SpacyNER
# from pre_trained.polyglot_ner import PolyglotNER
from pre_trained.stanford_ner import StanfordNER


class NERFactory:
    """Factory class that instantiate NER"""

    @staticmethod
    def from_name(ner_name):
        if ner_name == "pavlov":
            return PavlovNER()
        elif ner_name == "allen":
            return AllenNER()
        elif ner_name == "spacy":
            return SpacyNER()
        # elif ner_name == "polyglot":
        #     return PolyglotNER()
        elif ner_name == "stanford":
            return StanfordNER()
        else:
            raise ValueError("Couldn't instantiate NER. Given `{}`, but supported values are "
                             "`pavlov`, `allen`, `spacy`, `polyglot`".format(ner_name))
