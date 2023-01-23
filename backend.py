#Se importaran las librerias para crear la IA y el modelo de la red neuronal

import numpy as np
import nltk
import tensorflow
import tflearn
import random

#Se importan las librerias para el preprocesamiento de los datos
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

#Se crea una lista para las palabras
words = []
labels = []
docs_x = []
docs_y = []

#Se importan los datos de los archivos json
import json
with open("banco.json") as file:
    data = json.load(file)


#Se recorre el archivo json para obtener las palabras y las etiquetas
for banco in data["banco"]:
    for pattern in banco["patterns"]:
        wrds = nltk.word_tokenize(pattern)
        words.extend(wrds)
        docs_x.append(wrds)
        docs_y.append(banco["tag"])

    if banco["tag"] not in labels:
        labels.append(banco["tag"])

#Se limpian las palabras y se ordenan

words = [stemmer.stem(w.lower()) for w in words if w != "?"]
words = sorted(list(set(words)))
labels = sorted(labels)

#Se crea el modelo de la red neuronal y se entrena
training = []
output = []
out_empty = [0 for _ in range(len(labels))]

for x, doc in enumerate(docs_x):
    bag=[]
    wrds = [stemmer.stem(w) for w in doc]

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

#Crear modelo de red neuronal
net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 10)
net = tflearn.fully_connected(net, 10)
net = tflearn.fully_connected(net, 10)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

#Entrenar modelo+
model = tflearn.DNN(net)

try:
    model.load("model.tflearn")
except:                         
    model.fit(training, output, n_epoch = 5000, batch_size = 10, show_metric = True)
    model.save("model.tflearn")

def bankWords(s, words):
    bag = [0 for _ in range(len(words))]
    
    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]
    
    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
                
    return np.array(bag)
    




