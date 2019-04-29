import numpy as np
from sklearn.model_selection import train_test_split

from data_processing.sentence_getter import SentenceGetter
from data_processing.transformers import transform_elmo
from models.elmo_ner import ELMoNER

if __name__ == '__main__':
    getter = SentenceGetter(file_path='../data/ner_dataset.csv')
    data = getter.data
    sentences = getter.sentences

    words = list(set(data["Word"].values))
    words.append("ENDPAD")
    n_words = len(words)

    tags = list(set(data["Tag"].values))
    n_tags = len(tags)

    MAX_LEN = 64
    batch_size = 32
    X, y = transform_elmo(sentences=sentences, tags=tags, max_len=MAX_LEN)

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.1, random_state=42)
    X_tr, X_val = X_tr[:1213 * batch_size], X_tr[-135 * batch_size:]
    y_tr, y_val = y_tr[:1213 * batch_size], y_tr[-135 * batch_size:]
    y_tr = y_tr.reshape(y_tr.shape[0], y_tr.shape[1], 1)
    y_val = y_val.reshape(y_val.shape[0], y_val.shape[1], 1)
    model = ELMoNER(n_words=n_words, n_tags=n_tags, max_len=MAX_LEN, nbepochs=1)
    model.train(X_tr, y_tr, X_val, y_val)

    i = 19
    p = model.predict(np.array(X_te[i:i + batch_size]))[0]
    p = np.argmax(p, axis=-1)
    print("{:15} {:5}: ({})".format("Word", "Pred", "True"))
    print("=" * 30)
    for w, true, pred in zip(X_te[i], y_te[i], p):
        if w != "__PAD__":
            print("{:15}:{:5} ({})".format(w, tags[pred], tags[true]))