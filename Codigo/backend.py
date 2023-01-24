#Se importaran las librerias para crear la IA y el modelo de la red neuronal
import json
import nltk
import tensorflow
import tflearn
import random
import numpy as np
import pickle
#Se importan las librerias para el preprocesamiento de los datos
from nltk.stem.lancaster import LancasterStemmer
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()


with open('intents.json') as file:
    data = json.load(file)

data

try: 
    with open('data.pickle', 'rb') as f:
        words, labels, training, output = pickle.load(f)
except:

    #Se crea una lista para las palabras
    words = []
    labels = []
    docs_x = []
    docs_y = []


    for intents in data['intents']:
        for patterns in intents['patterns']:

            wrds = nltk.word_tokenize(patterns)
            words.extend(wrds)
            docs_x.append(wrds)
            docs_y.append(intents["tag"])

            if intents['tag'] not in labels:
                labels.append(intents['tag'])


    words = [stemmer.stem(w.lower()) for w in words if w != "?"]


    words = sorted(list(set(words)))
    words = [stemmer.stem(w.lower()) for w in words if w != "?"]
    words = sorted(list(set(words)))
    labels = sorted(labels)

    training = []
    output = []


    out_empty = [0 for _ in range(len(labels))]

    for x, doc in enumerate(docs_x):
        bag = []

        wrds = [stemmer.stem(w.lower()) for w in doc]

        for w in words:
            if w in wrds:

                bag.append(1)
            else:

                bag.append(0)

            output_row = out_empty[:]
            output_row[labels.index(docs_y[x])] = 1

            training.append(bag)
            output.append(output_row)

    #convertir lista a arreglos
    training = np.array(training)
    output = np.array(output)

with open ("data.pickle", "wb") as f:
        pickle.dump((words, labels, training, output), f)

tensorflow.compat.v1.reset_default_graph()

#Crear modelo de red neuronal
net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)
model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
model.save("model.tflearn")

def bag_of_words(s, words):
    
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1

    return np.array(bag)
