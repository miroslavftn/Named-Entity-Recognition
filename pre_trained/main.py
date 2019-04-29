import logging
from polyglot.downloader import downloader
import nltk
from entities import NERFactory

logging.getLogger().setLevel(logging.ERROR)

downloader.download("embeddings2.en")
downloader.download("ner2.en")

nltk.download('popular')


entity_mapping = {'PERSON': ['PERSON', 'B-PER', 'I-PER', 'L-PER', 'U-PER', 'B-PERSON', 'I-PERSON'],
                  'ORGANIZATION': ['ORG', 'B-ORG', 'I-ORG', 'L-ORG', 'U-ORG', 'ORGANIZATION'],
                  'MONEY': ['MONEY', 'B-MONEY', 'I-MONEY']}


def find_entities():
    ner = NERFactory.from_name(args.name)
    entities = ner.entities(args.text, entity_types)

    print("{} results:\n{}".format(ner, entities))


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--text", type=str, required=True)
    parser.add_argument("-n", "--name", type=str, choices=["pavlov", "allen", "spacy", "polyglot", "stanford"],
                        required=True)
    parser.add_argument("-type", "--type", default=[], nargs='*')

    args = parser.parse_args()

    types = args.type
    if len(types) == 1:
        entity_types = entity_mapping[types[0]]
    else:
        entity_types = []
        for t in types:
            entity_types.extend(entity_mapping[t])

    find_entities()
