from utils import read_functions
import spacy
from tqdm import tqdm


def tokenize(text_input: str, nlp_fr):
    document = nlp_fr(text_input)
    return document


def sum_words(doc):
    return len(doc)


def process(doc):
    """
    Remove stop word and create lemma of word
    :param doc: list of token in text
    :return: list of word lemma
    """
    filtered_sent = []
    for word in doc:
        if not word.is_stop:
            filtered_sent.append(word.lemma_)
    return filtered_sent



# use your path
path = r'./data'
frame = read_functions.read_csv(path)
print(frame)

nlp = spacy.load("fr_core_news_sm")
tqdm.pandas()
print("Tokenize")
frame['tokenized'] = frame['text'].progress_apply(tokenize, args=(nlp,))
print("Tokenized\nSum words")
frame['sum'] = frame['tokenized'].progress_apply(sum_words)
print("Tokenized\nSum words")

print("Number of token:", frame['sum'].sum())

