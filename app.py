from flask import Flask, request
import random
import numpy as np
import pandas as pd
import pickle
import json
import nltk
from keras.models import load_model
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import pandas as pd
import string
import re
import json
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pickle
import json
from nltk_utils import tokenize, stem, bag_of_ward
from train import all_word,tags,xy,set_state


model = load_model("model.h5")
intents = json.loads(open("dataset.json").read())

app = Flask(__name__)


@app.route("/hello")
def hello():
    return "hello"


@app.route("/chat", methods=["GET", "POST"])
def chatbot_response():
    flag = 0
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

    msg = str(request.args['msg'])

    real_sentence = msg
    msg = tokenize(msg)

    X = bag_of_ward(msg, all_word)
    X = np.array(X)

    output = model.predict(np.array([X]))[0]
    predicted = np.argmax(output)
    tag = tags[predicted.item()]
    if tag[len(tag) - 4:] == '_key':
        Diagnosis[tag[:len(tag) - 4]] = 1

    probs = np.exp(output) / (np.exp(output).sum())
    prob = probs[predicted.item()]
    if prob.item() > 0:

        if Diagnosis.get(tag[:len(tag) - 3], -1) + 1 and Diagnosis[tag[:len(tag) - 3]] == 0:
            Diagnosis[tag[:len(tag) - 3]] = 1

            if tag[:len(tag) - 3] == 'addictive' and Diagnosis['depreesion'] != 0:
                tag = tag
            else:
                tag = tag[:len(tag) - 3] + '_key'

        if Diagnosis.get(tag[:len(tag) - 3], -1) + 1 and tag[-2] == 'E':
            flag = 1
        for intent in intents["intents"]:
            if tag == intent["tag"]:
                set_state(tag)
                res = np.random.choice(intent['responses'])

                if flag == 1:
                    calc_state()
                    flag = 0
    else:
        res = "I do not understand..."

    dict1 = {}
    dict1["cnt"] = res

    return dict1
