from utils import read_functions
import spacy
import io
from tqdm import tqdm
# from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow.keras.utils

import tensorflow_datasets as tfds
import tensorflow as tf

# import time
import numpy as np
import matplotlib.pyplot as plt

# import pprint

from sklearn.model_selection import train_test_split

from tensorflow import keras
from tensorflow.keras import layers

from nlp import generator
from nlp import utils

# import tensorflow_datasets as tfds
tfds.disable_progress_bar()



# %% codecell
# Load Data
# use your path
path = r'data_test'
frame = read_functions.read_csv(path)
# print(frame)

nlp = spacy.load("fr_core_news_sm")
tqdm.pandas()
print("Tokenize")
frame['tokenized'] = frame['text'].progress_apply(utils.tokenize, args=(nlp,))
print("Tokenized\nSum words")
frame['sum'] = frame['tokenized'].progress_apply(utils.sum_words)
print("Tokenized\ntoekn to str")
frame['token_str'] = frame['tokenized'].progress_apply(utils.token_to_str)

# print(frame['token_str'][0])

# %% codecell
# NLP

#
doc = frame['tokenized'].iloc[0]
len(doc.vocab)

# Creating vocab from all words
train_examples = frame['tokenized'].tolist()
unique_tokens = utils.token_filter(train_examples)
unique_tokens_sorted = {k: v for k, v in sorted(unique_tokens.items(), key=lambda item: item[1], reverse=True)}
# for el in unique_tokens:
#    print("TEXT: ", el)
vocab_size = 2000
print("There are ", len(unique_tokens_sorted), " unique words but we keep the ", vocab_size, " most common")
vocab_tokens = list(unique_tokens_sorted)[:vocab_size]
print(len(vocab_tokens))
# print(len(vocab_tokens), " and ", vocab_tokens)
encoder = tfds.features.text.TokenTextEncoder(vocab_tokens)

print(unique_tokens_sorted.keys())


# %% codecell
# Build dataset
def encode(text: str, encoder: tfds.features.text.TokenTextEncoder, pad: bool = True) -> list:
    encoded_text = encoder.encode(text)
    return encoded_text


frame['encoded'] = frame['token_str'].progress_apply(encode, args=(encoder,))

# print(frame['encoded'])


def decode(text_encoded: list, encoder: tfds.features.text.TokenTextEncoder) -> str:
    decoded_text = encoder.decode(text_encoded)
    return decoded_text


# print(decode([568, 569, 1865, 0, 0, 0], encoder))
frame['decoded'] = frame['encoded'].progress_apply(decode, args=(encoder,))

# print(frame['decoded'])


# %% codecell

def stats_encoding(text):
    unk = 0
    known = 0
    for el in text:
        # print(el)
        if el == 2001:
            unk += 1
        else:
            known += 1
    return unk, known


stats = frame['encoded'].progress_apply(stats_encoding).tolist()

known = sum([k[1] for k in stats])
unk = sum([k[0] for k in stats])
print("The dataset has ", known, " known tokens and ", unk, " unknown tokens")


# %% codecell
# print( len(max(frame['encoded'].tolist(), key=len)))
# maxlen = len(max(frame['encoded'].tolist(), key=len))
# print("Longest text has ", maxlen, " words")
# dataset_list = pad_sequences(frame['encoded'].tolist(), maxlen=maxlen, padding='post')
#
# for el in dataset_list:
#     assert(len(el) == maxlen)

# dataset = tf.data.Dataset.from_tensor_slices(dataset_list)

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


print("Starting the split")
# concat_list_one_hot_x, concat_list_one_hot_y = to_one_hot_all_features(concat_list, encoder.vocab_size)
# print(concat_list_one_hot_x[1])
x_features = [k[0] for k in concat_list]
x_labels = [k[1] for k in concat_list]

print(x_features[0])
print(x_labels[0])

print("Train, val")

data_train, data_test, labels_train, labels_test = train_test_split(x_features,
                                                                    x_labels,
                                                                    test_size=0.20,
                                                                    random_state=42)

print("Done converting splitting the dataset")

# %% codecell


class one_hot_batch_generator(tensorflow.keras.utils.Sequence):
    def __init__(self, features, labels, batch_size, vocab_size):
        self.features, self.labels = np.array(features), np.array(labels)
        print(type(self.features))
        self.batch_size = batch_size
        self.vocab_size = vocab_size

    def __len__(self):
        return int(np.ceil(len(self.features) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.features[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.labels[idx * self.batch_size:(idx + 1) * self.batch_size]

        batch_x_one_hot = tf.one_hot(batch_x.astype(np.int32), depth=self.vocab_size)
        batch_y_one_hot = tf.one_hot(batch_y.astype(np.int32), depth=self.vocab_size)

        return batch_x_one_hot, batch_y_one_hot


batch_size = 52
training_batch_generator = one_hot_batch_generator(data_train, labels_train, batch_size, vocab_size + 2)
print("Number of batches", training_batch_generator.__len__())

batch_train, batch_label = training_batch_generator.__getitem__(10)

batch_train_index = [np.where(r==1)[0][0] for r in batch_train]
batch_label_index = [np.where(r==1)[0][0] for r in batch_label]
print(decode(batch_train_index, encoder))
print(decode(batch_label_index, encoder))

# %% codecell
# NLP DEEP LEARNING
embedding_dim = 16

print(encoder.vocab_size)

model = keras.Sequential([
  layers.Embedding(input_dim=encoder.vocab_size, output_dim=embedding_dim, input_length=encoder.vocab_size),
  # layers.GlobalAveragePooling1D(),
  layers.Flatten(),
  # layers.Dense(100),
  # layers.GlobalAveragePooling1D(),
  layers.Dense(encoder.vocab_size),
  layers.Softmax(),
  ])

model.summary()

learning_rate = 1e-4

METRICS = [
          # tf.keras.metrics.TruePositives(name='tp'),
          # tf.keras.metrics.FalsePositives(name='fp'),
          # tf.keras.metrics.TrueNegatives(name='tn'),
          # tf.keras.metrics.FalseNegatives(name='fn'),
          tf.keras.metrics.CategoricalAccuracy(),
          # tf.keras.metrics.Precision(name='precision'),
          # tf.keras.metrics.Recall(name='recall'),
          # tf.keras.metrics.AUC(name='auc'),
          ]

optimizer = tf.keras.optimizers.Adam(lr=learning_rate)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=METRICS)

# model.compile(optimizer='adam',
#               loss=tf.keras.losses.BinaryCrossentropy(),
#               metrics=['accuracy'])

# %%codcell

# print(data_train.shape)
# print(labels_train.shape)

history = model.fit(
  x=training_batch_generator,
  # batch_size=batch_size,
  verbose=1,
  validation_data=training_batch_generator,
  epochs=2)

# %% codecell
# SOME TESTS
batch_size = 1
training_batch_generator_1 = one_hot_batch_generator(data_train, labels_train, batch_size, vocab_size + 2)
print("Number of batches", training_batch_generator_1.__len__())

batch_train, batch_label = training_batch_generator_1.__getitem__(1)
print(batch_train)


batch_x_one_hot = tf.one_hot(np.array(data_train[0]).astype(np.int32), depth=vocab_size+2)
batch_y_one_hot = tf.one_hot(np.array(labels_train[0]).astype(np.int32), depth=vocab_size+2)

print(batch_x_one_hot.shape)

model.predict(x=np.array( [batch_x_one_hot,] ))

# %% codecell
# Retrieve embedding

e = model.layers[0]
weights = e.get_weights()[0]
print(weights.shape) # shape: (vocab_size, embedding_dim)


out_v = io.open('saved_files/vecs_g.tsv', 'w', encoding='utf-8')
out_m = io.open('saved_files/meta_g.tsv', 'w', encoding='utf-8')

# encoder.save_to_file("tokens")

for num, word in enumerate(encoder.tokens):
    # skip 0, it's padding
    vec = weights[num + 1]
    out_m.write(word + "\n")
    out_v.write('\t'.join([str(x) for x in vec]) + "\n")
out_v.close()
out_m.close()


# %% codecell

history_dict = history.history

acc = history_dict['categorical_accuracy']
# val_acc = history_dict['val_accuracy']
loss = history_dict['loss']
# val_loss = history_dict['val_loss']

epochs = range(1, len(acc) + 1)

plt.figure(figsize=(12,9))
plt.plot(epochs, loss, 'bo', label='Training loss')
# plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.figure(figsize=(12,9))
plt.plot(epochs, acc, 'bo', label='Training acc')
# plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.ylim((0,1))
plt.show()
