from utils import read_functions
import spacy
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer


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


def dummy_f(doc):
    return doc


def ngram_vectorize(train_texts):
    """Vectorizes texts as n-gram vectors.

    1 text = 1 tf-idf vector the length of vocabulary of unigrams + bigrams.

    # Arguments
        train_texts: list, training text strings.
        train_labels: np.ndarray, training labels.
        val_texts: list, validation text strings.

    # Returns
        x_train, x_val: vectorized training and validation texts
    """
    # Create keyword arguments to pass to the 'tf-idf' vectorizer.
    kwargs = {
        'ngram_range': (1, 2),  # Use 1-grams + 2-grams.
        'dtype': 'int32',
        'strip_accents': 'unicode',
        'decode_error': 'replace',
        'analyzer': 'word',  # Split text into word tokens.
        'min_df': 0,
        'tokenizer': process,
        'preprocessor': dummy_f,
        'token_pattern': None
    }
    vectorizer = TfidfVectorizer(**kwargs)

    # Learn vocabulary from training texts and vectorize training texts.
    x_train = vectorizer.fit_transform(train_texts)
    # print(vectorizer.vocabulary_)

    return x_train
    # Vectorize validation texts.
    # x_val = vectorizer.transform(val_texts)

    # # Limit on the number of features. We use the top 20K features.
    # TOP_K = 20000
    #
    # # Select top 'k' of the vectorized features.
    # selector = SelectKBest(f_classif, k=min(TOP_K, x_train.shape[1]))
    # selector.fit(x_train, train_labels)
    # x_train = selector.transform(x_train).astype('float32')
    # x_val = selector.transform(x_val).astype('float32')
    # return x_train, x_val, selector, vectorizer

# use your path
path = r'./data_test'
frame = read_functions.read_csv(path)
print(frame)

nlp = spacy.load("fr_core_news_sm")
tqdm.pandas()
print("Tokenize")
frame['tokenized'] = frame['text'].progress_apply(tokenize, args=(nlp,))
print("Tokenized\nSum words")
frame['sum'] = frame['tokenized'].progress_apply(sum_words)
print("Tokenized\nSum words")
# frame['processed'] = frame['tokenized'].progress_apply(process)
#
# doc = frame['tokenized'].iloc[0]
# for token in doc:
#     print(token.text, " is punct ", token.is_punct, " POS ", token.pos_,
#           " lemma nb ", token.lemma, " lemma ", token.lemma_, " is stop ", token.is_stop)

print("Number of token:", frame['sum'].sum())

tfidf_features = ngram_vectorize(frame['tokenized'].tolist())
print(tfidf_features.shape)

