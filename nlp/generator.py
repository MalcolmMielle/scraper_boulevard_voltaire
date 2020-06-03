import numpy as np
import tensorflow.keras.utils
import tensorflow as tf


class one_hot_batch_generator(tensorflow.keras.utils.Sequence):

    def __init__(self, features, labels, batch_size, vocab_size):
        self.features, self.labels = features, labels
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
