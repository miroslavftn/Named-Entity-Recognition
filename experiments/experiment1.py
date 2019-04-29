from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import Pipeline

from data_processing.sentence_getter import SentenceGetter
from models.feature_transformer import FeatureTransformer
from models.memorization import MemoryTagger


def full_classification_report(getter: SentenceGetter):
    print('Memory tagger full classification ...')
    data = getter.data
    words = data["Word"].values.tolist()
    tags = data["Tag"].values.tolist()
    pred = cross_val_predict(estimator=MemoryTagger(), X=words, y=tags, cv=5)
    report = classification_report(y_pred=pred, y_true=tags)
    print(report)


def full_classification_rf(getter: SentenceGetter):
    print('Memory tagger with Random Forest ...')
    data = getter.data
    tags = data["Tag"].values.tolist()
    pred = cross_val_predict(Pipeline([("feature_map", FeatureTransformer()),
                                       ("clf", RandomForestClassifier(n_estimators=5, n_jobs=1))]),
                             X=data, y=tags, cv=3)
    report = classification_report(y_pred=pred, y_true=tags)
    print(report)


if __name__ == '__main__':
    getter = SentenceGetter(file_path='../data/ner_dataset.csv')

    full_classification_report(getter=getter)
    full_classification_rf(getter=getter)



