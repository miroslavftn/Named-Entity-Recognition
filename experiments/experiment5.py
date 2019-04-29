from typing import List

import numpy as np

from data_processing.sentence_getter import SentenceGetter
from data_processing.transformers import transform_char
from models.keras_lstm_char import BiLSTMChar

MAX_LEN = 75
MAX_CHAR_LEN = 10


def bilstm_char_model_experiment(sentences: List, words: List, chars: List, tags: List, n_words: int, n_tags: int,
                                 n_chars: int, max_len: int, max_len_char: int):
    word2idx = {w: i + 2 for i, w in enumerate(words)}
    word2idx["UNK"] = 1
    word2idx["PAD"] = 0
    tag2idx = {t: i + 1 for i, t in enumerate(tags)}
    tag2idx["PAD"] = 0
    char2idx = {c: i + 2 for i, c in enumerate(chars)}
    char2idx["UNK"] = 1
    char2idx["PAD"] = 0
    X, X_char, y = transform_char(word2idx=word2idx,
                                  tag2idx=tag2idx,
                                  char2idx=char2idx,
                                  sentences=sentences,
                                  max_len=max_len,
                                  max_len_char=max_len_char)
    bilstm = BiLSTMChar(n_words=n_words,
                        max_len=max_len,
                        n_chars=n_chars,
                        n_tags=n_tags,
                        max_len_char=max_len_char,
                        nbepochs=5)
    bilstm.train(X, X_char, y)

    i = 2318
    print(sentences[i])
    prediction = bilstm.predict(np.array(X[i]), X_char[i])
    prediction = np.argmax(prediction, axis=-1)
    true = np.argmax(y[i], -1)
    print("{:15}||{:5}||{}".format("Word", "True", "Pred"))
    print(30 * "=")
    for w, t, pred in zip(X[i], true, prediction[0]):
        if w != 0:
            print("{:15}: {:5} {}".format(words[w - 1], tags[t], tags[pred]))


if __name__ == '__main__':
    getter = SentenceGetter(file_path='../data/ner_dataset.csv')
    data = getter.data
    sentences = getter.sentences

    words = list(set(data["Word"].values))
    chars = set([w_i for w in words for w_i in w])
    n_chars = len(chars)
    n_words = len(words)
    tags = list(set(data["Tag"].values))
    n_tags = len(tags)

    bilstm_char_model_experiment(sentences=sentences,
                                 words=words,
                                 chars=chars,
                                 tags=tags,
                                 n_words=n_words,
                                 n_chars=n_chars,
                                 n_tags=n_tags,
                                 max_len=MAX_LEN,
                                 max_len_char=MAX_CHAR_LEN)
