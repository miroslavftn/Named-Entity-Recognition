from sklearn.model_selection import cross_val_predict
from sklearn_crfsuite import CRF
from sklearn_crfsuite.metrics import flat_classification_report

from data_processing.sentence_getter import SentenceGetter
from data_processing.transformers import sent2features, sent2labels


def main():
    print('CRF model ...')
    crf = CRF(algorithm='lbfgs',
              c1=0.1,
              c2=0.1,
              max_iterations=100,
              all_possible_transitions=False)
    getter = SentenceGetter(file_path='../data/ner_dataset.csv')
    sentences = getter.sentences

    X = [sent2features(s) for s in sentences]
    y = [sent2labels(s) for s in sentences]
    pred = cross_val_predict(estimator=crf, X=X, y=y, cv=3)
    report = flat_classification_report(y_pred=pred, y_true=y)
    print(report)


if __name__ == '__main__':
    main()
