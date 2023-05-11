from nltk.stem.porter import PorterStemmer
import nltk
import numpy as np
# nltk.download('punkt')
stemmer = PorterStemmer()


def tokenize(sentence):
    return nltk.word_tokenize(sentence)


def stem(ward):
    return stemmer.stem(ward.lower())


def bag_of_ward(tokenized_sentence, all_ward):

    tokenized_sentence = [stem(w) for w in tokenized_sentence]

    bag = np.zeros(len(all_ward), dtype=np.float32)

    for idx, w in enumerate(all_ward):
        if w in tokenized_sentence:
            bag[idx] = 1.0

    return bag
