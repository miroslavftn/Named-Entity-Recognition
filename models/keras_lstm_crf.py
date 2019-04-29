import numpy as np
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional
from keras.models import Model, Input
from keras_contrib.layers import CRF
from keras_contrib.losses import crf_loss
from keras_contrib.metrics import crf_viterbi_accuracy
from sklearn.model_selection import train_test_split


class BiLSTMCRF(object):
    def __init__(self, n_words: int, max_len: int, n_tags: int, batch_size: int = 512, nbepochs: int = 5):
        self.model = None
        self.n_words = n_words
        self.n_tags = n_tags
        self.max_len = max_len
        self.nbepochs = nbepochs
        self.batch_size = batch_size
        self.build_model()

    def build_model(self):
        input = Input(shape=(self.max_len,))
        model = Embedding(input_dim=self.n_words, output_dim=50,
                          input_length=self.max_len, mask_zero=True)(input)  # 50-dim embedding
        model = Dropout(0.1)(model)
        model = Bidirectional(LSTM(units=100, return_sequences=True,
                                   recurrent_dropout=0.1))(model)
        model = TimeDistributed(Dense(50, activation="relu"))(model)  # a dense layer as suggested by neuralNer
        crf = CRF(self.n_tags)  # CRF layer
        out = crf(model)  # output
        model = Model(input, out)
        self.model = model

    def train(self, X, y):
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.1)
        self.model.compile(optimizer="rmsprop", loss=crf_loss, metrics=[crf_viterbi_accuracy])
        history = self.model.fit(np.array(X_tr), np.array(y_tr), batch_size=self.batch_size, epochs=self.nbepochs,
                                 validation_data=(np.array(X_te), np.array(y_te)), verbose=1)
        return history

    def predict(self, x):
        return self.model.predict(x)
