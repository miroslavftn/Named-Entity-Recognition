from deeppavlov import configs, build_model
from orderedset import OrderedSet

from pre_trained.base_ner import BaseNER


class PavlovNER(BaseNER):
    def __init__(self):
        super().__init__()
        self.ner = build_model(configs.ner.ner_ontonotes, download=False)

    def __repr__(self):
        return "Pavlov"

    def entities(self, text, types=[]):
        if isinstance(text, str):
            text = [text]

        texts, tags = self.ner(text)

        texts = texts[0]
        tags = tags[0]
        assert len(texts) == len(tags), "Mismatch between text and entities"

        entities = [(texts[i], tags[i]) for i in range(len(texts))]

        if types:
            entities = OrderedSet([ent for ent in entities if ent[1] in types])

        return list(entities)


if __name__ == "__main__":
    ner_model = PavlovNER()
    res = ner_model.entities('Bob Ross lived in Florida.')

    print(res)
