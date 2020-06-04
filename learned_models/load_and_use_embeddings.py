import csv
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


# %% codecell
def find_most_simialar(vec, vocab_dict):
    index = None
    cos = None
    for key in vocab_dict:
        cos_tmp = cosine_similarity(vec, vocab_dict[key].reshape(1, 16))
        if cos is None or (1 - cos_tmp < 1 - cos):
            cos = cos_tmp
            index = key
    return index


def make_vocab():
    list_arrays = list()
    with open('learned_models/vecs_embeddings.tsv') as tsvfile:
        reader = csv.reader(tsvfile, delimiter='\t')
        for row in reader:
            # print(row)
            # print(np.array(row).astype(np.float))
            list_arrays.append(np.array(row).astype(np.float))

    list_labels = list()
    with open('learned_models/meta_embeddings.tsv') as tsvfile:
        reader = csv.reader(tsvfile, delimiter='\t')
        for row in reader:
            # print(row[0])
            list_labels.append(row[0])

    # Create dict
    vocab_dict = dict()
    for i in range(len(list_arrays)):
        vocab_dict[list_labels[i]] = list_arrays[i]
    return vocab_dict


# %% codecell
vocab_dict = make_vocab()
word_vec = vocab_dict["france"].reshape(1, 16)
word2_vec = vocab_dict["terre"].reshape(1, 16)
word_vec_4 = word_vec + word2_vec
word4 = find_most_simialar(word_vec_4, vocab_dict)
print(word4)
