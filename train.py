import string
import re
import json
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pickle
import json
import numpy as np
from nltk_utils import tokenize, stem, bag_of_ward
from tensorflow.keras.preprocessing.text import Tokenizer

###################################################################################
with open('dataset.json') as f:
    intents = json.load(f)
###################################################################################
all_word = []
tags = []
xy = []

Diagnosis1 = {
    "depreesion": 9,
    "anxiety disorder": 7,
    "addictive": 5,
    "Schizophrenia": 8,
    "Postpartum": 7,
    "ADHD": 5,
    "PTSD": 7,
    "education": 6,
    }
Diagnosis = {
    "depreesion": 0,
    "anxiety disorder": 0,
    "addictive": 0,
    "Schizophrenia": 0,
    "Postpartum": 0,
    "ADHD": 0,
    "PTSD": 0,
    "education": 0,
    }
    
for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        w = tokenize(pattern)

        all_word.extend(w)
        xy.append((w, tag))

ignor_word = ['?', '.', '!', ',']
all_word = [stem(w) for w in all_word if w not in ignor_word]

all_word = sorted(set(all_word))
tags = sorted(set(tags))

vocab_size = len(all_word)
print(vocab_size)


Xtrain=[]
ytrain=[]
for (pattern_sentence, tag) in xy:
    bag = bag_of_ward(pattern_sentence, all_word)
    Xtrain.append(bag)

    label = tags.index(tag)
    ytrain.append(label)

Xtrain = np.array(Xtrain)
ytrain = np.array(ytrain)
ytrain = pd.get_dummies(ytrain)
ytrain = np.array(ytrain)
######################################################################################
max_length = Xtrain.shape[1]
max_classes = ytrain.shape[1]
max_length
######################################################################################
visited = []


def set_state(tag_init):
    tag = tag_init[:len(tag_init) - 3]
    signal = tag_init[-1]

    if tag_init in visited:
        return
    else:
        visited.append(tag_init)
        # print("msaaaaaaaa",tag,signal)
        if Diagnosis.get(tag, -1) + 1 and signal == 'Y':
            Diagnosis[tag] = Diagnosis.get(tag, 0) + 1


state = []


def calc_state():
    Diagnosis_copy = Diagnosis
    keys = list(Diagnosis_copy.keys())
    # print(keys)
    total = sum(Diagnosis_copy.values())
    # print(total)

    for i in keys:
        state.append([i, round(100 * Diagnosis_copy[i] / total)])
    state.sort(key=lambda i: i[1], reverse=True)
    visited = []
    Diagnosis_copy = Diagnosis_copy.fromkeys(Diagnosis_copy, 0)


def show_state():
    for i in state:
        if i[1]:
            print(f"{bot_name}: {i[0]} related symptoms {str(i[1]) + '%'}")


