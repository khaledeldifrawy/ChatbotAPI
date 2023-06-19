from flask import Flask, request
import numpy as np
import pickle
import tensorflow as tf
import json
import joblib
from keras.models import load_model
from nltk_utils import tokenize, stem, bag_of_ward
from train import all_word,tags,xy,set_state



app = Flask(__name__)


tokenizer_q1 = joblib.load("tokenizer_q.pkl")
tokenizer_a1 = joblib.load("tokenizer_a.pkl")

num_layers = 4
d_model = 1024
dff = 512
num_heads = 8
input_vocab_size = tokenizer_q.vocab_size + 2
target_vocab_size = tokenizer_a.vocab_size + 2
dropout_rate = 0.1
load_transformer = Transformer(num_layers, d_model, num_heads, dff,
                          input_vocab_size, target_vocab_size, 
                          pe_input=input_vocab_size, 
                          pe_target=target_vocab_size,
                          rate=dropout_rate)
load_transformer.load_weights('weights.data-00000-of-00001')


model = load_model("model.h5")
intents = json.loads(open("new-dataset.json").read())
 
MAX_LENGTH = 70

def evaluate(inp_sentence, model,  tokenizer_q, tokenizer_a):
    start_token = [tokenizer_q.vocab_size]
    end_token = [tokenizer_q.vocab_size + 1]

    # All questions has the start and end token
    inp_sentence = start_token + tokenizer_q.encode(inp_sentence) + end_token
    encoder_input = tf.expand_dims(inp_sentence, 0)

    # 'answers' start token : 27358
    decoder_input = [tokenizer_a.vocab_size]
    decoder_input = tf.expand_dims(decoder_input, 0)

    for i in range(MAX_LENGTH):
        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(encoder_input, decoder_input)

        # predictions.shape == (batch_size, seq_len, vocab_size)
        predictions, attention_weights = model(encoder_input, 
                                                     decoder_input,
                                                     False,
                                                     enc_padding_mask,
                                                     combined_mask,
                                                     dec_padding_mask)

        # select the last word from the seq_len dimension
        predictions = predictions[: ,-1:, :]  # (batch_size, 1, vocab_size)

        predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

        # return the result if the predicted_id is equal to the end token
        if predicted_id == tokenizer_a.vocab_size+1:
#             print(f"=============\nGot end token\n=============")
            return tf.squeeze(decoder_input, axis=0), attention_weights

        # concatentate the predicted_id to the output which is given to the decoder
        # as its input.
        decoder_input = tf.concat([decoder_input, predicted_id], axis=-1)

    return tf.squeeze(decoder_input, axis=0), attention_weights
 
def reply(sentence, transformer,  tokenizer_q, tokenizer_a, plot=''):
    result, attention_weights = evaluate(sentence, transformer,  tokenizer_q, tokenizer_a)
#     print("Attention_Blocks:", list(attention_weights.keys()))
    predicted_sentence = tokenizer_a.decode([i for i in result 
                                            if i < tokenizer_a.vocab_size])  

#     print('Input: {}'.format(sentence))
#     print('Predicted translation: {}'.format(predicted_sentence))
    if plot:
        plot_attention_weights(attention_weights,tokenizer_q, tokenizer_a, sentence, result, plot)
    return sentence, predicted_sentence
 

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
    if generative == 0:
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
            _,respon=reply(msg, load_transformer,  tokenizer_q, tokenizer_a, "decoder_layer2_block2")
            res = respon
    else:
        _,respon=reply(msg, load_transformer,  tokenizer_q, tokenizer_a, "decoder_layer2_block2")
        res = respon

    dict1 = {"cnt": res}
    return dict1

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000,debug=True)
