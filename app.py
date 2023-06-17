from flask import Flask, request
import numpy as np
import pickle
import json
from keras.models import load_model
from nltk_utils import tokenize, stem, bag_of_ward
from train import all_word,tags,xy,set_state

app = Flask(__name__)

model = load_model("grad.h5")
intents = json.loads(open("new-dataset.json").read())
 

@app.route("/chat", methods=["GET", "POST"])
def chatbot_response():
    flag = 0
    diagnosis = {
        "depression": 0,
        "anxiety disorder": 0,
        "addictive": 0,
        "Schizophrenia": 0,
        "Postpartum": 0,
        "ADHD": 0,
        "PTSD": 0,
        "education": 0,
    }

    msg = str(request.args.get('msg', ''))
    tokens = tokenize(msg)
    X = bag_of_ward(tokens, all_word)
    X = np.array(X)

    output = model.predict(np.array([X]))[0]
    predicted = np.argmax(output)
    tag = tags[predicted]

    if tag[-4:] == '_key':
        diagnosis[tag[:-4]] = 1

    prob = output[predicted]
    if prob > 0:
        if diagnosis.get(tag[:-3], -1) + 1 and diagnosis[tag[:-3]] == 0:
            diagnosis[tag[:-3]] = 1
            if tag[:-3] == 'addictive' and diagnosis['depression'] != 0:
                tag = tag
            else:
                tag = tag[:-3] + '_key'

        if diagnosis.get(tag[:-3], -1) + 1 and tag[-2] == 'E':
            flag = 1

        for intent in intents["intents"]:
            if tag == intent["tag"]:
                res = np.random.choice(intent['responses'])
             
        
    else:
        res = "Please start with a meaningful word."

    dict1 = {"cnt": res}
    return dict1

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000,debug=True)
