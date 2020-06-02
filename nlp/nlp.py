from utils import read_functions
import spacy
import io
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow.keras.utils

import tensorflow_datasets as tfds
import tensorflow as tf

import time
import numpy as np
import matplotlib.pyplot as plt

import pprint

import re
from sklearn.model_selection import train_test_split

from tensorflow import keras
from tensorflow.keras import layers

import tensorflow_datasets as tfds
tfds.disable_progress_bar()


# %% codecell
# Load functions
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
    list_out = set()
    for text in corpus:
        for word in text:
            if not word.is_stop and word.is_alpha:
                # print("w:" + word.lemma_)
                list_out.add(preprocess_token(word))
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


def one_hot_encode(train_texts):
    own_embedding_vocab_size = 500
    encoded_docs_oe = [one_hot(d, own_embedding_vocab_size) for d in train_texts]
    # print(encoded_docs_oe)
    maxlen = 500
    padded_docs_oe = pad_sequences(encoded_docs_oe, maxlen=maxlen, padding='post')
    print(padded_docs_oe)
    return padded_docs_oe


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


# %% codecell
# Load Data
# use your path
path = r'data_test'
frame = read_functions.read_csv(path)
# print(frame)

nlp = spacy.load("fr_core_news_sm")
tqdm.pandas()
print("Tokenize")
frame['tokenized'] = frame['text'].progress_apply(tokenize, args=(nlp,))
print("Tokenized\nSum words")
frame['sum'] = frame['tokenized'].progress_apply(sum_words)
print("Tokenized\ntoekn to str")
frame['token_str'] = frame['tokenized'].progress_apply(token_to_str)

# print(frame['token_str'][0])

# %% codecell
# NLP

#
doc = frame['tokenized'].iloc[0]
len(doc.vocab)

# Creating vocab from all words
train_examples = frame['tokenized'].tolist()
unique_tokens = token_filter(train_examples)
for el in unique_tokens:
    print("TEXT: ", el)

vocab = {k: v for v, k in enumerate(unique_tokens)}
vocab_decode = {v: k for k, v in vocab.items()}
pprint.pprint(vocab)

print(vocab_decode[224])
print(len(vocab))

encoder = tfds.features.text.TokenTextEncoder(unique_tokens)

test = "front être test"
encoded_example = encoder.encode(test)
# print(encoded_example)

decoded_example = encoder.decode(encoded_example)
# print(decoded_example)


# %% codecell
# Build dataset
def encode(text: str, encoder: tfds.features.text.TokenTextEncoder, pad: bool = True) -> list:
    encoded_text = encoder.encode(text)
    return encoded_text


frame['encoded'] = frame['token_str'].progress_apply(encode, args=(encoder,))

print(frame['encoded'])


def decode(text_encoded: list, encoder: tfds.features.text.TokenTextEncoder) -> str:
    decoded_text = encoder.decode(text_encoded)
    return decoded_text


print(decode([0, 0, 0, 0, 0, 0], encoder))

frame['decoded'] = frame['encoded'].progress_apply(decode, args=(encoder,))

print(frame['decoded'])

maxlen = 500
dataset_list = pad_sequences(frame['encoded'].tolist(), maxlen=maxlen, padding='post')

dataset = tf.data.Dataset.from_tensor_slices(dataset_list)
print(dataset)
for elem in dataset:
    print(elem.numpy())


# %% codecel
# NLP
# BUFFER_SIZE = 50000
# BATCH_SIZE = 64
# TAKE_SIZE = int(len(dataset_list) * 2 / 3)
#
# train_data = dataset.skip(TAKE_SIZE).shuffle(BUFFER_SIZE)
# train_data = train_data.padded_batch(BATCH_SIZE)
#
# test_data = dataset.take(TAKE_SIZE)
# test_data = test_data.padded_batch(BATCH_SIZE)
#
# sample_text = next(iter(train_data))
# sample_text[0]
#
# sample_text2 = next(iter(test_data))
# sample_text2[0]

# %% codecell
# Let's do CBOW
# https://petamind.com/word2vec-with-tensorflow-2-0-a-simple-cbow-implementation/
# First we create the feeatures / labels
# TODO Do not make association between words separated by a point.


def features_cbow(text: list):
    data = []
    WINDOW_SIZE = 2
    for word_index, word in enumerate(text):
        for nb_word in text[max(word_index - WINDOW_SIZE, 0): min(word_index + WINDOW_SIZE, len(text)) + 1]:
            if nb_word != word:
                data.append([word, nb_word])
    return data


frame['features_cbow'] = frame['encoded'].progress_apply(features_cbow)

print(frame['encoded'][0])
print(frame['features_cbow'][0])


dataset_cbow = frame['features_cbow'].tolist()
concat_list = [j for i in dataset_cbow for j in i]

print(np.asarray(concat_list).shape)
print(concat_list[0])


# function to convert numbers to one hot vectors
def to_one_hot(data_point_index, vocab_size):
    temp = np.zeros(vocab_size)
    temp[data_point_index] = 1
    return temp


def to_one_hot_all_features(features: list, vocab_size: int):
    # input word
    x_train = []
    # output word
    y_train = []
    for data_word in features:
        x_train.append(to_one_hot(features[0], vocab_size))
        y_train.append(to_one_hot(features[1], vocab_size))
    # convert them to numpy arrays
    x_train = np.asarray(x_train, dtype='float32')
    y_train = np.asarray(y_train, dtype='float32')
    return x_train, y_train

print("Starting the conversion")
concat_list_one_hot_x, concat_list_one_hot_y = to_one_hot_all_features(concat_list, encoder.vocab_size)
print(concat_list_one_hot_x[1])

data_train, data_test, labels_train, labels_test = train_test_split(concat_list_one_hot_x,
                                                                    concat_list_one_hot_y,
                                                                    test_size=0.20,
                                                                    random_state=42)

print("Done converting to one-hot")

# %% codecell

embedding_dim = 16

model = keras.Sequential([
  layers.Embedding(input_dim=encoder.vocab_size, output_dim=embedding_dim, input_length=encoder.vocab_size),
  # layers.GlobalAveragePooling1D(),
  layers.Flatten(),
  # layers.Dense(100),
  # layers.GlobalAveragePooling1D(),
  layers.Dense(encoder.vocab_size, activation='softmax')
  ])

model.summary()

learning_rate = 1e-4

METRICS = [
          tf.keras.metrics.TruePositives(name='tp'),
          tf.keras.metrics.FalsePositives(name='fp'),
          tf.keras.metrics.TrueNegatives(name='tn'),
          tf.keras.metrics.FalseNegatives(name='fn'),
          tf.keras.metrics.BinaryAccuracy(name='accuracy'),
          tf.keras.metrics.Precision(name='precision'),
          tf.keras.metrics.Recall(name='recall'),
          tf.keras.metrics.AUC(name='auc'),
          ]

optimizer = tf.keras.optimizers.Adam(lr=learning_rate)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=METRICS)

# model.compile(optimizer='adam',
#               loss=tf.keras.losses.BinaryCrossentropy(),
#               metrics=['accuracy'])

print(data_train.shape)
print(labels_train.shape)

history = model.fit(
  x=data_train,
  y=labels_train,
  batch_size=52,
  verbose=1,
  validation_data=(data_test, labels_test),
  epochs=10)


# %% codecell
# Retrieve embedding

e = model.layers[0]
weights = e.get_weights()[0]
print(weights.shape) # shape: (vocab_size, embedding_dim)


out_v = io.open('saved_files/vecs.tsv', 'w', encoding='utf-8')
out_m = io.open('saved_files/meta.tsv', 'w', encoding='utf-8')

# encoder.save_to_file("tokens")

for num, word in enumerate(encoder.tokens):
    # skip 0, it's padding
    vec = weights[num + 1]
    out_m.write(word + "\n")
    out_v.write('\t'.join([str(x) for x in vec]) + "\n")
out_v.close()
out_m.close()
