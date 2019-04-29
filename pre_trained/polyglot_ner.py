from orderedset import OrderedSet
from polyglot.text import Text

from pre_trained.base_ner import BaseNER


class PolyglotNER(BaseNER):
    def __init__(self):
        super().__init__()

    def __repr__(self):
        return "Polyglot"

    def entities(self, text, types=[]):
        processed_text = Text(text)
        entities = [(' '.join(ent), ent.tag) for ent in processed_text.entities]

        if types:
            entities = OrderedSet([ent for ent in entities if ent[1] in types])

        return list(entities)


if __name__ == "__main__":
    ner_model = PolyglotNER()
    res = ner_model.entities('Bob Ross lived in Florida.')

    print(res)
