import keras.backend.tensorflow_backend as K
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from keras.layers import LSTM, Dense, TimeDistributed, Lambda, Bidirectional
from keras.layers.merge import add
from keras.models import Model, Input


class ELMoNER(object):

    def __init__(self, n_words: int, max_len: int, n_tags: int, batch_size: int = 32, nbepochs: int = 5):
        self.model = None
        self.n_words = n_words
        self.n_tags = n_tags
        self.max_len = max_len
        self.nbepochs = nbepochs
        self.batch_size = batch_size
        self.build_model()

    def ElmoEmbedding(self, x):
        sess = tf.Session()
        K.set_session(sess)

        elmo_model = hub.Module("https://tfhub.dev/google/elmo/2", trainable=True)
        sess.run(tf.global_variables_initializer())
        sess.run(tf.tables_initializer())

        return elmo_model(inputs={
            "tokens": tf.squeeze(tf.cast(x, tf.string)),
            "sequence_len": tf.constant(self.batch_size * [self.max_len])
        },
            signature="tokens",
            as_dict=True)["elmo"]

    def build_model(self):
        input_text = Input(shape=(self.max_len,), dtype=tf.string)
        embedding = Lambda(self.ElmoEmbedding, output_shape=(self.max_len, 1024))(input_text)
        # embedding = ElmoEmbeddingLayer() (input_text)
        x = Bidirectional(LSTM(units=512, return_sequences=True,
                               recurrent_dropout=0.2, dropout=0.2))(embedding)
        x_rnn = Bidirectional(LSTM(units=512, return_sequences=True,
                                   recurrent_dropout=0.2, dropout=0.2))(x)
        x = add([x, x_rnn])  # residual connection to the first biLSTM
        out = TimeDistributed(Dense(self.n_tags, activation="softmax"))(x)
        self.model = Model(input_text, out)

    def train(self, X_tr, y_tr, X_val, y_val):
        self.model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        history = self.model.fit(np.array(X_tr), y_tr, validation_data=(np.array(X_val), y_val),
                                 batch_size=self.batch_size, epochs=5, verbose=1)
        return history

    def predict(self, x):
        return self.model.predict(x)
