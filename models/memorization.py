from sklearn.base import BaseEstimator, TransformerMixin


class MemoryTagger(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.tags = []
        self.memory = {}

    def fit(self, X, y):
        """
        Expects a list of words as X and a list of tags as y.
        """
        voc = {}
        self.tags = []
        for x, t in zip(X, y):
            if t not in self.tags:
                self.tags.append(t)
            if x in voc:
                if t in voc[x]:
                    voc[x][t] += 1
                else:
                    voc[x][t] = 1
            else:
                voc[x] = {t: 1}
        self.memory = {}
        for k, d in voc.items():
            self.memory[k] = max(d, key=d.get)

    def predict(self, X, y=None):
        """
        Predict the the tag from memory. If word is unknown, predict 'O'.
        """
        return [self.memory.get(x, 'O') for x in X]
