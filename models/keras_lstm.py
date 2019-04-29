import numpy as np
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional
from keras.models import Model, Input
from sklearn.model_selection import train_test_split


class BiLSTMNER(object):
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
        model = Embedding(input_dim=self.n_words, output_dim=50, input_length=self.max_len)(input)
        model = Dropout(0.1)(model)
        model = Bidirectional(LSTM(units=100, return_sequences=True, recurrent_dropout=0.1))(model)
        out = TimeDistributed(Dense(self.n_tags, activation="softmax"))(model)  # softmax output layer
        model = Model(input, out)
        self.model = model

    def train(self, X, y):
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.1)
        self.model.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"])
        history = self.model.fit(np.array(X_tr), np.array(y_tr), batch_size=self.batch_size, epochs=self.nbepochs,
                                 validation_data=(np.array(X_te), np.array(y_te)), verbose=1)
        return history

    def predict(self, x):
        return self.model.predict(x)
