import pandas as pd


class SentenceGetter(object):
    """
    retrieve sentences with their labels grouped by sentence number and aggregated
    """

    def __init__(self, file_path):
        self.data = pd.read_csv(file_path, encoding="latin1").fillna(method="ffill")
        self.empty = False
        agg_func = lambda s: [(w, p, t) for w, p, t in zip(s["Word"].values.tolist(),
                                                           s["POS"].values.tolist(),
                                                           s["Tag"].values.tolist())]
        self.grouped = self.data.groupby("Sentence #").apply(agg_func)
        self.sentences = [s for s in self.grouped]
