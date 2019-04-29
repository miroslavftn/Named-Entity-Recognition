import spacy
from pre_trained import BaseNER
from orderedset import OrderedSet


class SpacyNER(BaseNER):
    def __init__(self):
        super().__init__()
        self.nlp = spacy.load("en_core_web_sm")

    def __repr__(self):
        return "Spacy"

    def entities(self, text, types=[]):
        doc = self.nlp(text)

        entities = [(token.text, token.ent_type_) for token in doc]

        if types:
            entities = OrderedSet([ent for ent in entities if ent[1] in types])

        return list(entities)


if __name__ == "__main__":
    ner_model = SpacyNER()
    res = ner_model.entities('Bob Ross lived in Florida.')

    print(res)


