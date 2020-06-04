# from utils import read_functions
# import spacy
import io
# from tqdm import tqdm
import pandas as pd
# from tensorflow.keras.preprocessing.text import one_hot
# from tensorflow.keras.preprocessing.sequence import pad_sequences
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
from nlp import preprocess
from nlp import cbow
# import tensorflow_datasets as tfds
tfds.disable_progress_bar()


# %% codecell
# Load Data
# use your path
path = r'data'
vocab_size = 2000
frame = preprocess.load_and_tokenize(path)

# %% codecell
# NLP
#
# Creating vocab from all words
train_examples = frame['tokenized'].tolist()
encoder = utils.build_encoder(train_examples, vocab_size)
# print(unique_tokens_sorted.keys())


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


def encoding_stat_dataset(dataset: pd.DataFrame, column_name: str = 'token_str') -> None:
    stats = dataset['encoded'].progress_apply(stats_encoding).tolist()
    known = sum([k[1] for k in stats])
    unk = sum([k[0] for k in stats])
    print("The dataset has ", known, " known tokens and ", unk, " unknown tokens")


frame['encoded'] = frame['token_str'].progress_apply(preprocess.encode, args=(encoder,))
encoding_stat_dataset(frame)


# %% codecell
# Let's do CBOW
# https://petamind.com/word2vec-with-tensorflow-2-0-a-simple-cbow-implementation/
# First we create the feeatures / labels
# TODO Do not make association between words separated by a point.

frame['features_cbow'] = frame['encoded'].progress_apply(cbow.features_cbow)

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


batch_size = 52
training_batch_generator = generator.one_hot_batch_generator(data_train, labels_train, batch_size, vocab_size + 2)
test_batch_generator = generator.one_hot_batch_generator(data_test, labels_test, batch_size, vocab_size + 2)
print("Number of batches", training_batch_generator.__len__())

batch_train, batch_label = training_batch_generator.__getitem__(10)

batch_train_index = [np.where(r == 1)[0][0] for r in batch_train]
batch_label_index = [np.where(r == 1)[0][0] for r in batch_label]
print(preprocess.decode(batch_train_index, encoder))
print(preprocess.decode(batch_label_index, encoder))

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

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    verbose=1,
    patience=10,
    mode='max',
    restore_best_weights=True)


history = model.fit(
  x=training_batch_generator,
  # batch_size=batch_size,
  verbose=1,
  validation_data=test_batch_generator,
  epochs=50,
  callbacks=[early_stopping])


# %% codecell
# SOME TESTS
batch_size = 1
training_batch_generator_1 = generator.one_hot_batch_generator(data_train, labels_train, batch_size, vocab_size + 2)
print("Number of batches", training_batch_generator_1.__len__())

batch_train, batch_label = training_batch_generator_1.__getitem__(1)
print(batch_train)


batch_x_one_hot = tf.one_hot(np.array(data_train[0]).astype(np.int32), depth=vocab_size+2)
batch_y_one_hot = tf.one_hot(np.array(labels_train[0]).astype(np.int32), depth=vocab_size+2)

print(batch_x_one_hot.shape)

model.predict(x=np.array([batch_x_one_hot, ]))

# %% codecell
# Retrieve embedding

e = model.layers[0]
weights = e.get_weights()[0]
print(weights.shape)  # shape: (vocab_size, embedding_dim)


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

plt.figure(figsize=(12, 9))
plt.plot(epochs, loss, 'bo', label='Training loss')
# plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.figure(figsize=(12, 9))
plt.plot(epochs, acc, 'bo', label='Training acc')
# plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.ylim((0, 1))
plt.show()
