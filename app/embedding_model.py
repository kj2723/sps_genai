import spacy

nlp = spacy.load("en_core_web_lg")


def calculate_embedding(input_word):
    word = nlp(input_word)
    return word.vector


def calculate_similarity(word1, word2):
    return nlp(word1).similarity(nlp(word2))