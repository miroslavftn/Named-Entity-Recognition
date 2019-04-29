import numpy as np
from keras.layers import Bidirectional, concatenate, SpatialDropout1D
from keras.layers import LSTM, Embedding, Dense, TimeDistributed
from keras.models import Model, Input
from sklearn.model_selection import train_test_split


class BiLSTMChar(object):
    def __init__(self, n_words: int, n_chars: int, max_len: int, max_len_char: int, n_tags: int,
                 batch_size: int = 64, nbepochs: int = 10):
        self.model = None
        self.n_words = n_words
        self.n_chars = n_chars
        self.n_tags = n_tags
        self.max_len = max_len
        self.max_len_char = max_len_char
        self.nbepochs = nbepochs
        self.batch_size = batch_size
        self.build_model()

    def build_model(self):
        word_in = Input(shape=(self.max_len,))
        emb_word = Embedding(input_dim=self.n_words + 2, output_dim=20,
                             input_length=self.max_len, mask_zero=True)(word_in)

        # input and embeddings for characters
        char_in = Input(shape=(self.max_len, self.max_len_char,))
        emb_char = TimeDistributed(Embedding(input_dim=self.n_chars + 2, output_dim=10,
                                             input_length=self.max_len_char, mask_zero=True))(char_in)
        # character LSTM to get word encodings by characters
        char_enc = TimeDistributed(LSTM(units=20, return_sequences=False,
                                        recurrent_dropout=0.5))(emb_char)

        # main LSTM
        x = concatenate([emb_word, char_enc])
        x = SpatialDropout1D(0.3)(x)
        main_lstm = Bidirectional(LSTM(units=50, return_sequences=True,
                                       recurrent_dropout=0.6))(x)
        out = TimeDistributed(Dense(self.n_tags + 1, activation="softmax"))(main_lstm)

        self.model = Model([word_in, char_in], out)
        self.model.summary()

    def train(self, X_word, X_char, y):
        X_word_tr, X_word_te, y_tr, y_te = train_test_split(X_word, y, test_size=0.1, random_state=2018)
        X_char_tr, X_word_te, _, _ = train_test_split(X_char, y, test_size=0.1, random_state=2018)
        self.model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["acc"])
        history = self.model.fit([X_word_tr,
                                  np.array(X_char_tr).reshape((len(X_char_tr), self.max_len, self.max_len_char))],
                                 np.array(y_tr).reshape(len(y_tr), self.max_len, 1),
                                 batch_size=self.batch_size, epochs=self.nbepochs, validation_split=0.1, verbose=1)
        return history

    def predict(self, X_word, X_char):
        return self.model.predict([X_word,
                                   np.array(X_char).reshape((len(X_char),
                                                             self.max_len, self.max_len_char))])
