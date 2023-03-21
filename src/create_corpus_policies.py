# coding: utf-8

__author__ = 'Catarina Silva'
__version__ = '0.1'
__email__ = 'c.alexandracorreia@ua.pt'
__status__ = 'Development'


import re
import bs4
import json
import nltk
import tqdm
import joblib
import pathlib
import logging
import argparse


logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def setup_nltk():
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('wordnet', quiet=True)


def text2tokens(raw_text, lemmatizer, min_word:int=3):
    # array to store the document tokens
    document_tokens = []

    # apply the setence tokenizer (divides a document into sentences)
    sentences = nltk.tokenize.sent_tokenize(raw_text)
    for s in sentences:
        # normalize the text
        text = re.sub('[^a-zA-Z]', ' ', s)
        text = text.lower()
        # apply the tokenizer (divides a sentence into tokens)
        tokens = nltk.tokenize.word_tokenize(text)
        # remove stop words
        tokens = [word for word in tokens if word not in nltk.corpus.stopwords.words('english')]
        # remove small words
        tokens = [t for t in tokens if len(t) >= min_word]
        # applying lemmanaziation to the selected tokens
        tokens = [lemmatizer.lemmatize(word) for word in tokens]
        document_tokens.append(tokens)
    return document_tokens


def main(args):
    # setup nltk (download auxiliary files)
    setup_nltk()
    
    # open input path and list all the files
    input_path = pathlib.Path(args.i)
    files = [item for item in input_path.iterdir() if item.is_file()]
    
    # create array to store the raw policies
    document_tokens = []
    lemmatizer = nltk.stem.WordNetLemmatizer()

    # for each file in the input folder
    for f in tqdm.tqdm(files):
        with f.open(encoding="UTF-8") as source:
            soup = bs4.BeautifulSoup(source, "html.parser")
            raw = soup.get_text()
            #raw_policies.append(raw)
            tokens = text2tokens(raw, lemmatizer, args.mw)
            document_tokens.extend(tokens)
    
    # write the corpus file (json)
    with pathlib.Path(args.o).open('w', encoding='UTF-8') as target: 
        json.dump({'corpus':document_tokens}, target)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create corpus from policies')
    parser.add_argument('-i', type=str, help='input path', default='sanitized_policies')
    parser.add_argument('-o', type=str, help='output path', default='corpus/corpus_policies.json')
    parser.add_argument('--mw', type=int, help='minimum word size', default=3)
    args = parser.parse_args()
    
    main(args)