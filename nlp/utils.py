from sklearn.feature_extraction.text import TfidfVectorizer
# import pprint
import re
import tensorflow_datasets as tfds


def remove_accents(string):
    if type(string) is not str:
        string = str(string, encoding='utf-8')

    string = re.sub(u"[àáâãäå]", 'a', string)
    string = re.sub(u"[èéêë]", 'e', string)
    string = re.sub(u"[ìíîï]", 'i', string)
    string = re.sub(u"[òóôõö]", 'o', string)
    string = re.sub(u"[ùúûü]", 'u', string)
    string = re.sub(u"[ýÿ]", 'y', string)
    string = re.sub(u"[ç]", 'c', string)

    return string


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


def preprocess_token(token) -> str:
    return remove_accents(token.lemma_.lower().strip())


def token_filter(corpus: list) -> list:
    list_out = dict()
    for text in corpus:
        for word in text:
            if not word.is_stop and word.is_alpha:
                word_text = preprocess_token(word)
                if word_text in list_out:
                    list_out[word_text] = list_out[word_text] + 1
                else:
                    # print("w:" + word.lemma_)
                    list_out[word_text] = 1
    return list_out


def token_to_str(words: list) -> str:
    str_out = ""
    for word in words:
        if not word.is_stop and word.is_alpha:
            # print("w:" + word.lemma_)
            str_out += " " + preprocess_token(word)
    return str_out


def dummy_f(doc):
    return doc


# # TODO maybe some possible optimization here
# def one_hot_encode(train_texts):
#     own_embedding_vocab_size = 500
#     encoded_docs_oe = [one_hot(d, own_embedding_vocab_size) for d in train_texts]
#     # print(encoded_docs_oe)
#     maxlen = 500
#     padded_docs_oe = pad_sequences(encoded_docs_oe, maxlen=maxlen, padding='post')
#     print(padded_docs_oe)
#     return padded_docs_oe


def ngram_vectorize(train_texts):
    """Vectorizes texts as n-gram vectors.

    1 text = 1 tf-idf vector the length of vocabulary of unigrams + bigrams.

    # Arguments
        train_texts: list, training text strings.
    # Returns
        x_train, vectorized training
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
    return x_train


def build_encoder(train_examples: list, vocab_size: int = 2000) -> tfds.features.text.TokenTextEncoder:
    unique_tokens = token_filter(train_examples)
    unique_tokens_sorted = {k: v for k, v in sorted(unique_tokens.items(), key=lambda item: item[1], reverse=True)}
    # for el in unique_tokens:
    #    print("TEXT: ", el)
    print("There are ", len(unique_tokens_sorted), " unique words but we keep the ", vocab_size, " most common")
    vocab_tokens = list(unique_tokens_sorted)[:vocab_size]
    print(len(vocab_tokens))
    # print(len(vocab_tokens), " and ", vocab_tokens)
    return tfds.features.text.TokenTextEncoder(vocab_tokens)
