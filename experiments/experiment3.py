from typing import List

import numpy as np

from data_processing.sentence_getter import SentenceGetter
from data_processing.transformers import transform
from models.keras_lstm import BiLSTMNER


def bilstm_ner_model_experiment(sentences: List, words: List, tags: List, n_words: int, n_tags: int, max_len: int):
    word2idx = {w: i for i, w in enumerate(words)}
    tag2idx = {t: i for i, t in enumerate(tags)}
    X, y = transform(word2idx=word2idx, tag2idx=tag2idx, sentences=sentences, n_words=n_words, n_tags=n_tags,
                     max_len=max_len)
    bilstm = BiLSTMNER(n_words=n_words, max_len=max_len, n_tags=n_tags)
    bilstm.train(X, y)

    i = 2318
    print(sentences[i])
    prediction = bilstm.predict(np.array([X[i]]))
    prediction = np.argmax(prediction, axis=-1)
    true = np.argmax(y[i], -1)
    print("{:15}||{:5}||{}".format("Word", "True", "Pred"))
    print(30 * "=")
    for w, t, pred in zip(X[i], true, prediction[0]):
        if w != n_words - 1:
            print("{:15}: {:5} {}".format(words[w], tags[t], tags[pred]))


if __name__ == '__main__':
    MAX_LEN = 50
    getter = SentenceGetter(file_path='../data/ner_dataset.csv')
    data = getter.data
    sentences = getter.sentences

    words = list(set(data["Word"].values))
    words.append("ENDPAD")
    n_words = len(words)
    tags = list(set(data["Tag"].values))
    n_tags = len(tags)

    bilstm_ner_model_experiment(sentences, words, tags, n_words, n_tags, MAX_LEN)
