import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import pickle
import numpy as np
import tkinter
from tkinter import *

from keras.models import load_model
model = load_model('chatbot_model.h5')
import json
import random
intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl','rb'))
classes = pickle.load(open('classes.pkl','rb'))


def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bolsa(sentence, words, show_details=True):
    sentence_words = clean_up_sentence(sentence)
    bag = [0]*len(words)
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)
    return(np.array(bag))

def predict_class(sentence, model):
    p = bolsa(sentence, words,show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_MINIMO = 0.25
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_MINIMO]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def darRespuesta(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if(i['tag']== tag):
            result = random.choice(i['responses'])
            break
    return result

def chatbot_response(msg):
    ints = predict_class(msg, model)
    res = darRespuesta(ints, intents)
    return res


def send():
    msg = EntryBox.get("1.0",'end-1c').strip()
    EntryBox.delete("0.0",END)

    if msg != '':
        ChatLog.config(state=NORMAL)
        ChatLog.insert(END, "Tú: " + msg + '\n\n')
        ChatLog.config(foreground="#442265", font=("Verdana", 12 ))

        res = chatbot_response(msg)
        ChatLog.insert(END, "Bot: " + res + '\n\n')

        ChatLog.config(state=DISABLED)
        ChatLog.yview(END)


base = Tk()
base.title("ChatBot I.A. 2021B")
base.geometry("600x600")
base.resizable(width=FALSE, height=FALSE)

ChatLog = Text(base, bd=0, bg="#fdf964", height="30", width="40", font="Arial",)

ChatLog.config(state=DISABLED)

scrollbar = Scrollbar(base, command=ChatLog.yview, cursor="heart")
ChatLog['yscrollcommand'] = scrollbar.set

EntryBox = Text(base, bd=0, bg="#eeeeee",width="65", height="25", font="Arial")

SendButton = Button(base, font=("Arial",12,'bold'), text="Enviar", width="12", height=5,
                    bd=0, bg="#acf4e1", activebackground="#3c9d9b",fg='#000000',
                    command= send )

scrollbar.place(x=576,y=12, height=518)
ChatLog.place(x=6,y=6, height=570, width=570)
EntryBox.place(x=6, y=522, height=60, width=565)
SendButton.place(x=440, y=522, height=60)

base.mainloop()
