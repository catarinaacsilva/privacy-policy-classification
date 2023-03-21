# coding: utf-8

__author__ = 'Catarina Silva'
__version__ = '0.1'
__email__ = 'c.alexandracorreia@ua.pt'
__status__ = 'Development'


import re
import json
import nltk
import tqdm
import pathlib
import logging
import argparse

import polars as pl


logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def setup_nltk():
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('wordnet', quiet=True)


def text2tokens(raw_text, lemmatizer, min_word:int=3):
    # normalize the text
    text = re.sub('[^a-zA-Z]', ' ', raw_text)
    text = text.lower()
    # apply the tokenizer (divides a sentence into tokens)
    tokens = nltk.tokenize.word_tokenize(text)
    # remove stop words
    tokens = [word for word in tokens if word not in nltk.corpus.stopwords.words('english')]
    # remove small words
    tokens = [t for t in tokens if len(t) >= min_word]
    # applying lemmanaziation to the selected tokens
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    return tokens


def main(args):
    # setup nltk (download auxiliary files)
    setup_nltk()

    # open input path and list all the files
    input_path = pathlib.Path(args.i)
    files = [item for item in input_path.iterdir() if item.is_file()]
    
    # create one array for the corpus (json)
    array_json = []
    lemmatizer = nltk.stem.WordNetLemmatizer()

    # for each file in the input folder
    for f in tqdm.tqdm(files):
        df = pl.read_csv(f, has_header=False)
        df = df[['column_6', 'column_7']]
        df = df.rename({'column_6':'category', 'column_7':'text'})

        for row in df.rows():
            category  = row[0]
            # skip the Other category
            if category != 'Other':
                # load the json from the second column
                text = json.loads(row[1])
                
                # each key in the json is a target
                for target in text:
                    selected = text[target]

                    # the selected text can be in a key named text or selectedText
                    if 'text' in selected:
                        selected_text = selected['text']
                    elif 'selectedText' in selected:
                        selected_text = selected['selectedText']
                    
                    # discard null and Not selected pieces of data
                    if selected_text not in ['null', 'Not selected']:
                        tokens = text2tokens(selected_text, lemmatizer, args.mw)
                        if len(tokens) >= args.mt:
                            array_json.append({'class':category, 'tokens':tokens})

    
    # write the corpus file (json)
    with pathlib.Path(args.o).open('w', encoding='UTF-8') as target: 
        json.dump({'corpus':array_json}, target)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create corpus from annotations')
    parser.add_argument('-i', type=str, help='input path', default='annotations')
    parser.add_argument('-o', type=str, help='output path', default='corpus/corpus_annotations.json')
    parser.add_argument('--mw', type=int, help='minimum word size', default=3)
    parser.add_argument('--mt', type=int, help='minimum number of tokens', default=3)
    args = parser.parse_args()
    
    main(args)