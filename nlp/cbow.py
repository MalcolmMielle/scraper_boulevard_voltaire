

def features_cbow(text: list):
    data = []
    WINDOW_SIZE = 2
    for word_index, word in enumerate(text):
        for nb_word in text[max(word_index - WINDOW_SIZE, 0): min(word_index + WINDOW_SIZE, len(text)) + 1]:
            if nb_word != word:
                data.append([word, nb_word])
    return data
