import logging

# from polyglot.downloader import downloader
import nltk

from pre_trained.ner_factory import NERFactory

logging.getLogger().setLevel(logging.ERROR)

# downloader.download("embeddings2.en")
# downloader.download("ner2.en")

nltk.download('popular')

entity_mapping = {'PERSON': ['PERSON', 'B-PER', 'I-PER', 'L-PER', 'U-PER', 'B-PERSON', 'I-PERSON'],
                  'ORGANIZATION': ['ORG', 'B-ORG', 'I-ORG', 'L-ORG', 'U-ORG', 'ORGANIZATION'],
                  'MONEY': ['MONEY', 'B-MONEY', 'I-MONEY']}


def find_entities(name='pavlov', text='', entity_types=[]):
    ner = NERFactory.from_name(name)
    entities = ner.entities(text, entity_types)

    print("{} results:\n{}".format(ner, entities))


if __name__ == '__main__':
    text = "Did Uriah honestly think he could beat The Legend of Zelda in under three hours?"
    name = 'spacy'
    types = ['PERSON', 'ORGANIZATION']
    if len(types) == 1:
        entity_types = entity_mapping[types[0]]
    else:
        entity_types = []
        for t in types:
            entity_types.extend(entity_mapping[t])

    find_entities(name, text, entity_types)
