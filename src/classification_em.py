# coding: utf-8

__author__ = 'Catarina Silva'
__version__ = '0.1'
__email__ = 'c.alexandracorreia@ua.pt'
__status__ = 'Development'


import os
import re
import enum
import json
import nltk
import tqdm
import joblib
import pathlib
import logging
import argparse
import numpy as np

import gensim
from gensim.models import Word2Vec
from gensim.models.fasttext import FastText

from sklearn.model_selection import train_test_split
from sklearn.metrics import matthews_corrcoef, accuracy_score, f1_score

from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


logging.getLogger('gensim').setLevel(logging.ERROR)
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


# Enumerate that allows the used to select
# the correct embedding
class WordEmbeddings(enum.Enum):
    word2vec = 'word2vec'
    fasttext = 'fasttext'

    def __str__(self):
        return self.value


def div_norm(x):
   norm_value = np.linalg.norm(x)
   if norm_value > 0:
       return x * ( 1.0 / norm_value)
   else:
       return x


def word_vector_to_sentence_vector(sentence:list, model):
    vectors = []
    # for all the tokens in the setence
    for token in sentence:
        if token in model:
            vectors.append(model[token])
    # add the EOS token
    if '\n' in model:
        vectors.append(model['\n'])
    # normalize all the vectors
    vectors = [div_norm(x) for x in vectors]
    return np.mean(vectors, axis=0)


def main(args):
    # load the cleaned corpus file
    with pathlib.Path(args.a).open(encoding="UTF-8") as source:
        corpus = json.load(source)
    
    # load the corpus_policies file
    with pathlib.Path(args.p).open(encoding="UTF-8") as source:
        corpus_policies = json.load(source)
    
    # get targets and tokens from json
    corpus_array = corpus['corpus']
    targets = [element['class'] for element in corpus_array]
    tokens =  [element['tokens'] for element in corpus_array]

    # get the tokens from the corpus policies
    tokens_policies = corpus_policies['corpus']

    if args.we is WordEmbeddings.fasttext:
        text_model = FastText(vector_size=args.vs, window=args.ws,
        min_count=args.mc, workers=os.cpu_count(), seed=42)
    else:
        text_model = Word2Vec(vector_size=args.vs, window=args.ws,
        min_count=args.mc, workers=os.cpu_count(), seed=42)
    
    # train the language model on the Privacy Policies documents
    logger.info('Train the language model on the Privacy Policies')
    text_model.build_vocab(tokens_policies)
    text_model.train(tokens_policies, total_examples=text_model.corpus_count, epochs=100)

    # update model with tokens from annotation
    logger.info('Update the language model using the annotations')
    text_model.build_vocab(tokens, update=True)
    text_model.train(tokens, total_examples=len(tokens), epochs=10)

    # get setence embeddings
    logger.info('Compute the sentence embedding from the word embedding')
    X = np.array([word_vector_to_sentence_vector(sentence, text_model.wv) for sentence in tqdm.tqdm(tokens)])
    X_train, X_test, y_train, y_test = train_test_split(X, targets, stratify=targets, test_size=0.2, random_state=42)

    logger.info(f'Training Data : {len(X_train)}')
    logger.info(f'Testing Data  : {len(X_test)}')

    # define the list of classifiers
    clfs = [
        ('LR', LogisticRegression(random_state=42, multi_class='auto', max_iter=600)),
        ('KNN', KNeighborsClassifier(n_neighbors=1)),
        ('NB', GaussianNB()),
        ('RFC', RandomForestClassifier(random_state=42)),
        ('MLP', MLPClassifier(random_state=42, learning_rate='adaptive', max_iter=1000))
    ]

    # whenever possible used joblib to speed-up the training
    with joblib.parallel_backend('loky', n_jobs=-1):
        for label, clf in clfs:
            # train the model
            clf.fit(X_train, y_train)

            # generate predictions
            predictions = clf.predict(X_test)

            # compute the performance metrics
            mcc = matthews_corrcoef(y_test, predictions)
            acc = accuracy_score(y_test, predictions)
            f1 = f1_score(y_test, predictions, average='weighted')
            logger.info(f'{label:3} {acc:.2f} {f1:.2f} {mcc:.2f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create corpus from annotations')
    parser.add_argument('-a', type=str, help='corpus annotations path', default='corpus/corpus_annotations.json')
    parser.add_argument('-p', type=str, help='corpus policies path', default='corpus/corpus_policies.json')
    parser.add_argument('--we', type=WordEmbeddings, choices=list(WordEmbeddings), default='fasttext')
    parser.add_argument('--vs', type=int, help='word vector size', default=256)
    parser.add_argument('--mc', type=int, help='minimum count', default=3)
    parser.add_argument('--ws', type=int, help='window size', default=7)
    
    args = parser.parse_args()
    
    main(args)