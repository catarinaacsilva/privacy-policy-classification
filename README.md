# Privacy Policy Classification

The code in this repository is intended to be used as a simple 
proof-of-concept, showing that NLP techniques can be used to
identify relevant sentences within privacy policy documents.

This work relies on the [OPP-115 Corpus (ACL 2016)](https://cmu.flintbox.com/technologies/9900a353-1bac-4d65-b197-d785cdda85bc)
which can be found [here](https://usableprivacy.org/data).
Due to the license of the dataset, we do not re-distribute it here.
However, we have prepared a bash script that setups the annotations folder
for the corpus generation.

## Usage

The following sections explain how to run this proof-of-concept.

### Setup

Given the license of the dataset, we do not redistribute it here.
To set up the environment run the following commands:

```bash
./setup.sh
```

After we need to create a virtual environment to run the proof-of-concept.

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Generate the corpus

From the OPP-115 dataset, we generate two corpus 
1. One using the sanitized policies (only used to train the word embeddings)
2. The other one is created using the annotations (used to train the classification models)

The first corpus contains the tokens per sentence in all the privacy policies.
The second one contains specific sentences that were selected and annotated as relevant parts of the privacy policy.
This is the target of the classification methods in this work.

To generate the corpus simply run the following python script (within the virtual environment created previously):

```bash
python -m src.create_corpus_policies
python -m src.create_corpus_annotations
```

### Train models using term frequency

To train the models using only term frequency, run the following script (after the generation of the corpus datasets):

```bash
python -m src.classification_tf
```

### Train models using sentence embedding

To train the models using sentence embedding, run the following script (after the generation of the corpus datasets):

```bash
python -m src.classification_em
```

The sentence embedding is computed and the normalized average of the word embedding (either computed with fasttext, or word2vec).
This approach provides better results overall.

### Results

The models were trained on 80% of the annotations, excluding the **Other** category, since it englobes such broad topics that are difficult to predict (as per the original paper on the dataset).

The first script uses the term frequency to train a model.
The obtained results are aligned with the previous publications, meaning that we are reaching around 0.6 on F1-Score.

|Model| Accuracy | F1-Score| MCC |
|-----|      ---:|     ---:| ---:|
|LR   |0.68      | 0.63    | 0.43|
|KNN  |0.57      | 0.56    | 0.38|
|NB   |0.68      | 0.63    | 0.42|
|RFC  |0.68      | 0.63    | 0.43|
|MLP  |0.68      | 0.63    | 0.43|

The main contribution of this work is the usage of language models to improve these results.
The following tables show the results for the classification using Word2Vec and FastText language modelling.

|Model| Accuracy | F1-Score| MCC |
|-----|      ---:|     ---:| ---:|
|LR   |0.70      | 0.69    | 0.48|
|KNN  |0.86      | 0.87    | 0.77|
|NB   |0.55      | 0.58    | 0.35|
|RFC  |0.87      | 0.86    | 0.77|
|MLP  |0.86      | 0.86    | 0.77|

|Model| Accuracy | F1-Score| MCC |
|-----|      ---:|     ---:| ---:|
|LR   |0.70      | 0.69    | 0.48|
|KNN  |0.86      | 0.87    | 0.78|
|NB   |0.56      | 0.59    | 0.36|
|RFC  |0.87      | 0.86    | 0.77|
|MLP  |0.86      | 0.86    | 0.77|

As we can see in the results, there is a considerable improvement when using a language model to interpret the sentences in a privacy policy.
The limitation of the previous term-based classification is relying specifically on the existence of specific terms. While a language model computes fixed-size vectors that model the meaning of the sentence.

## Authors

* **Catarina Silva** - [catarinaacsilva](https://github.com/catarinaacsilva)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details