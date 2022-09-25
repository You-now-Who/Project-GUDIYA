import re
import speech_recognition as sr
import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import pickle
import numpy as np
from keras.models import load_model
model = load_model('chatbot_model.h5')
import json
import random

# from translate import Translator

# translator_bot = Translator(from_lang = "en", to_lang="mr")
# translator_human = Translator(from_lang = "mr", to_lang="en")

import pyttsx3

from huggingface_hub import list_models

from transformers import MarianMTModel, MarianTokenizer
from typing import Sequence

class Translator:
    def __init__(self, source_lang: str, dest_lang: str) -> None:
        self.model_name = f'Helsinki-NLP/opus-mt-{source_lang}-{dest_lang}'
        self.model = MarianMTModel.from_pretrained(self.model_name)
        self.tokenizer = MarianTokenizer.from_pretrained(self.model_name)
        
    def translate(self, texts: Sequence[str]) -> Sequence[str]:
        tokens = self.tokenizer(list(texts), return_tensors="pt", padding=True)
        translate_tokens = self.model.generate(**tokens)
        return [self.tokenizer.decode(t, skip_special_tokens=True) for t in translate_tokens]
        
global translator_human
global translator_bot

translator_human = Translator('hi', 'en')
translator_bot = Translator('en', 'hi')

global curr_lang
curr_lang = 'hindi'

engine = pyttsx3.init()

voices = engine.getProperty('voices')
for voice in voices:
    print(f"Voice: {voice.name}")
engine.setProperty('voice', voices[1].id)

#""" RATE"""
rate = engine.getProperty('rate')   # getting details of current speaking rate
print (rate)                        #printing current voice rate
engine.setProperty('rate', 175)     # setting up new voice rate

intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl','rb'))
classes = pickle.load(open('classes.pkl','rb'))

languages = {
    'marathi': 'mr',
    'hindi': 'hi',
    'english': 'en'
}

recognize_lang = {
        'marathi': 'mr-IN',
        'hindi': 'hi-IN',
        'english': 'en-US'
    }

def woman_in_stem_response():

    women_list = [
        "Katherine Johnson was a NASA space scientist. Born in ninteen eighteen, she graduated from university at the age of 18. She was awarded the prestigious Presidential Medal of freedom in two thousand fifteen for her contirbutions to physics and mathematics.",
        "Radia Perlman an early computer scientist and student of MIT. She was a pioneer of the internet. She developed the algorithm behind the Spanning Tree Protocol (STP), an innovation that made today’s Internet possible. She also invented TRILL to correct limitations of STP as well.",
        "Dr. Priya Abraham, an alumna of Christian Medical College in Vellore, India and the former head of the department of Clinic Virology, was appointed in November 2019 as the Director of the National Institute of Virology (NIV) in Pune, India—not knowing she would be confronted with a global pandemic merely two months later. With her team at NIV, Dr. Priya Abraham has been instrumental in isolating and sequencing strains of the SARS-CoV-2, the virus responsible for the COVID-19 pandemic."
        
    ]

    return random.choice(women_list)


# CHANGE THE LANGUAGE
def lang_change():

    global curr_lang
    global translator_human
    global translator_bot

    bot_response = "What language do you want to change to? Marathi, Hindi or English?"
            
    print("Bot:", bot_response)

    if curr_lang != 'english':
        bot_response = translator_bot.translate([bot_response])[0]
        print("Bot (translated):", bot_response)
    
    engine.say(bot_response)
    engine.runAndWait()
    
    user_input = mic_input()
    user_input = user_input.lower()
    
    if user_input in languages:
        
        curr_lang = user_input

        if curr_lang != 'english':
            translator_human = Translator(languages[curr_lang] , 'en')
            translator_bot = Translator('en', languages[curr_lang])
            voices = engine.getProperty('voices')
            for voice in voices:
                print(f"Voice: {voice.name}")
            engine.setProperty('voice', voices[1].id)
        
        else:
            voices = engine.getProperty('voices')
            for voice in voices:
                print(f"Voice: {voice.name}")
            engine.setProperty('voice', voices[2].id)
    
    else:
        return("Sorry, I don't know that language yet")

    return (f"Language changed to {curr_lang}")





# PREPROCESSING DATA
def clean_up_sentence(sentence):
    # tokenize the pattern - split words into array
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word - create short form for word
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words
# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence



def bow(sentence, words, show_details=True):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words - matrix of N words, vocabulary matrix
    bag = [0]*len(words) 
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s: 
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)
    return(np.array(bag))



# GETTING PREDICTIONS
def predict_class(sentence, model):
    # filter out predictions below a threshold
    p = bow(sentence, words,show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list



# GETTING RESPONSES
def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if(i['tag']== tag):
            result = random.choice(i['responses'])
            break
    return result



# MAIN FUNCTION THAT WORKS THE CODE OF THE RESPONSES
#I know I should have used a class or at least a switch for this but I was in a hurry

def check_response(sentence):
    if sentence == "CODE_WOMAN_IN_STEM":
        return (woman_in_stem_response())
    
    elif sentence == "CODE_CHANGE_LANG":
        return (lang_change())

    else:
        return sentence



def chatbot_response(text):
    ints = predict_class(text, model)
    res = getResponse(ints, intents)

    return check_response(res)





 # TAKES THE MIC INPUT 

def mic_input():
    global curr_lang
    with mic as source:
        r.adjust_for_ambient_noise(source)
        print("\nListening...")
        audio = r.listen(source)

    user_input = r.recognize_google(audio, language = recognize_lang[curr_lang])
    print("Thinking...")

    print("\nYou:", user_input)

    if curr_lang != 'english':
        user_input = translator_human.translate([user_input])[0]
        print("You (translated):", user_input)

    # user_input = translator_human.translate([user_input])[0]
    # print("\nYou (translated):", user_input)

    return user_input


# MAIN FUNCTION THAT RUNS THE CHATBOT

if __name__ == "__main__":
    r = sr.Recognizer()
    mic = sr.Microphone()

    # translator_human = Translator('hi', 'en')
    # translator_bot = Translator('en', 'hi')

    while True:

        try:
            user_input = mic_input()

            # user_input = input("\nYou: ")
            
            bot_response = chatbot_response(user_input)
            
            print("Bot:", bot_response)

            if curr_lang != 'english':
                bot_response = translator_bot.translate([bot_response])[0]
                print("Bot (translated):", bot_response)

            # bot_response = translator_bot.translate([bot_response])[0]
            # print("Bot (translated):", bot_response)
            
            engine.say(bot_response)
            engine.runAndWait()
        
        except:
            print("No input recieved! Stopping recording")
            input()
        
        