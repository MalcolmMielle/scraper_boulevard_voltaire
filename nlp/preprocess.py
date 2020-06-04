from utils import read_functions
import spacy
from tqdm import tqdm
import tensorflow_datasets as tfds
from nlp import utils

# import tensorflow_datasets as tfds
tfds.disable_progress_bar()


def load_and_tokenize(path: str):
    """
    Load data in pandas frame, tokenize it and return
    """
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
    return frame


def encode(text: str, encoder: tfds.features.text.TokenTextEncoder) -> list:
    """
    Encode a txt str. Need for pandas apply
    """
    encoded_text = encoder.encode(text)
    return encoded_text


def decode(text_encoded: list, encoder: tfds.features.text.TokenTextEncoder) -> str:
    """
    Decode a index list. Need for pandas apply
    """
    decoded_text = encoder.decode(text_encoded)
    return decoded_text
