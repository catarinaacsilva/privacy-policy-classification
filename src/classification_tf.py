# coding: utf-8

__author__ = 'Catarina Silva'
__version__ = '0.1'
__email__ = 'c.alexandracorreia@ua.pt'
__status__ = 'Development'


import re
import json
import nltk
import tqdm
import joblib
import pathlib
import logging
import argparse

from sklearn.model_selection import train_test_split
from sklearn.metrics import matthews_corrcoef, accuracy_score, f1_score

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def main(args):
    # load the cleaned corpus file
    with pathlib.Path(args.a).open(encoding="UTF-8") as source:
        corpus = json.load(source)
    
    # get X an y from json
    corpus_array = corpus['corpus']
    y = [element['class'] for element in corpus_array]
    X = [''.join(element['tokens']) for element in corpus_array]
    
    # split the dataset into training and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
    logger.info(f'Training Data : {len(X_train)}')
    logger.info(f'Testing Data  : {len(X_test)}')

    # train Bag of Words model
    cv = CountVectorizer(ngram_range = (1, 2), max_features=args.vs)
    X_train_cv = cv.fit_transform(X_train)
    logger.info(f'Training Data CV : {X_train_cv.shape}')

    # transform X_test using CV
    X_test_cv = cv.transform(X_test)

    clfs = [
        ('LR', LogisticRegression(random_state=42, multi_class='auto', max_iter=600)),
        ('KNN', KNeighborsClassifier(n_neighbors=1)),
        ('NB', MultinomialNB()),
        ('RFC', RandomForestClassifier(random_state=42)),
        ('MLP', MLPClassifier(random_state=42, learning_rate='adaptive', max_iter=1000))
    ]

    with joblib.parallel_backend('loky', n_jobs=-1):
        for label, clf in clfs:
            # train the model
            clf.fit(X_train_cv, y_train)

            # generate predictions
            predictions = clf.predict(X_test_cv)

            # compute the performance metrics
            mcc = matthews_corrcoef(y_test, predictions)
            acc = accuracy_score(y_test, predictions)
            f1 = f1_score(y_test, predictions, average='weighted')
            logger.info(f'{label:3} {acc:.2f} {f1:.2f} {mcc:.2f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create corpus from annotations')
    parser.add_argument('-a', type=str, help='corpus annotations path', default='corpus/corpus_annotations.json')
    parser.add_argument('--vs', type=int, help='word vector size', default=4096)
    args = parser.parse_args()
    
    main(args)