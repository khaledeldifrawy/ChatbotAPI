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

model = load_model("final.h5")
intents = json.loads(open("new-dataset.json").read())


app = Flask(__name__)


@app.route("/hello")
def hello():
    return "hello"


@app.route("/chat", methods=["GET", "POST"])          
def chatbot_response():
    steps = 0
    depreesion = ["1. Practice Self-Care: Taking care of your physical and emotional needs is important for managing depression. This can include getting enough sleep, eating a balanced diet, exercising regularly, and engaging in activities that bring you joy.","2. Seek professional help: Depression is a serious condition that may require medical attention. It is important to speak with a mental health professional who can provide guidance and support in managing your symptoms.","3. Follow your treatment plan: If you are prescribed a medication, be sure to take it as directed by your doctor. Attend all therapy sessions and actively participate in your treatment plan.","4. Challenging negative thoughts: Depression can often involve negative thoughts and self-criticism. Practice challenging these thoughts by looking for evidence that supports or contradicts them, and rephrasing them in a more positive and realistic light.","5. Stay connected: Social support is important for managing depression. Make an effort to stay in touch with friends and loved ones, even if it feels challenging.","6. Practice relaxation techniques: Relaxation techniques such as deep breathing, meditation or yoga can help manage stress and reduce symptoms of depression.","7. Remember that recovery from depression is a process and may take time. Be patient with yourself and don't hesitate to ask for help when you need it."]
    def steps_print (Diagnosy):
        if Diagnosy == 'depreesion':
            for i in depreesion :
                print(i)
    msg = str(request.args['msg'])
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
            steps=1
            
        for intent in intents["intents"]:
            if tag == intent["tag"]:
                set_state(tag)
                res = np.random.choice(intent['responses'])
                if steps == 1 :
                        steps_print(tag[:len(tag)-3])
                        steps=0
                if flag == 1:
                    calc_state()
                    flag = 0
    else:
        res = "I do not understand..."

    dict1 = {}
    dict1["cnt"] = res

    return dict1

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000,debug=True)
