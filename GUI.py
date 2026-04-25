from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
from tkinter import filedialog
import matplotlib.pyplot as plt
import numpy as np
from tkinter.filedialog import askopenfilename
import pandas as pd
import seaborn as sns
import os
import pickle
import nltk
import random
from sklearn.feature_extraction.text import TfidfVectorizer
import string
import warnings
from nltk.corpus import stopwords
import Firefly
from Firefly import *
import SwarmPackagePy
from SwarmPackagePy import testFunctions as tf
from rouge_score import rouge_scorer


global filename, main, tf1
global rougeScores
global X, Y
global dataset, articles, summaries, tfidf, tfidf_results

stopwords =  stopwords.words('english')

HANDICAP = 0.95

def remove_punctuation_marks(text) :
    punctuation_marks = dict((ord(punctuation_mark), None) for punctuation_mark in string.punctuation)
    return text.translate(punctuation_marks)

def get_lemmatized_tokens(text) :
    normalized_tokens = nltk.word_tokenize(remove_punctuation_marks(text.lower()))
    return [nltk.stem.WordNetLemmatizer().lemmatize(normalized_token) for normalized_token in normalized_tokens]

def uploadDataset(): #function to upload dataset
    global filename, dataset, labels, X, Y
    filename = filedialog.askopenfilename(initialdir="Dataset")
    pathlabel.config(text=filename)
    text.delete('1.0', END)
    dataset = pd.read_csv(filename, nrows=100, usecols=['article','highlights'])
    text.insert(END,str(dataset))

    temp = dataset['article']
    for i in range(0, 10):
        print(temp[i])
        print()

def get_average(values) :
    greater_than_zero_count = total = 0
    for value in values :
        if value != 0 :
            greater_than_zero_count += 1
            total += value 
    return total / greater_than_zero_count

def get_threshold(tfidf_results) :
    i = total = 0
    while i < (tfidf_results.shape[0]) :
        total += get_average(tfidf_results[i, :])
        i += 1
    return total / tfidf_results.shape[0]

def get_summary(documents, tfidf_results, tfidf) :
    summary = ""
    i = 0
    while i < (tfidf_results.shape[0]) :
        test = tfidf.transform([documents[i]]).toarray()
        ff = Firefly(test, tf.easom_function, np.min(test), np.max(test), test.shape[1], 1, 1, 1, 1, 0.1, 0, 0.1)
        test = ff.get_agents()
        test = np.asarray(test)
        test = test[0].ravel()
        print(test.shape)
        if (get_average(test)) >= get_threshold(tfidf_results) * HANDICAP :
            summary += ' ' + documents[i]
        i += 1        
    return summary    

def preprocessDataset():
    global dataset, article, summaries
    text.delete('1.0', END)
    article = nltk.sent_tokenize(dataset['article'][0])
    summaries = dataset['highlights'][0]
    text.insert(END,"Extracted Sentences from Dataset\n\n")
    text.insert(END,article)

def runVector():
    global article, summaries, tfidf, tfidf_results
    text.delete('1.0', END)
    #create vector from training dataset
    tfidf = TfidfVectorizer(tokenizer = get_lemmatized_tokens, stop_words = stopwords)
    tfidf_results = tfidf.fit_transform(article).toarray()
    text.insert(END,"Dataset Converted Numeric Vector\n\n")
    text.insert(END,str(tfidf_results))

def runFirefly():
    text.delete('1.0', END)
    global article, summaries, tfidf, tfidf_results, rougeScores
    rougeScores = []
    #tfidf vector will be input to firefly for optimization
    ff = Firefly(tfidf_results, tf.easom_function, np.min(tfidf_results), np.max(tfidf_results), tfidf_results.shape[1], 1, 1, 1, 1, 0.1, 0, 0.1)
    #select best features from FF firefly object
    tfidf_results = ff.get_agents()
    tfidf_results = np.asarray(tfidf_results)
    tfidf_results = tfidf_results[0]
    #use firefly optimize features to predict summary
    predict = get_summary(article, tfidf_results, tfidf)
    #calculate rouge score on firefly predicted summary and original summary
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    scores = scorer.score(summaries, predict)
    rouge1 = np.amax(scores.get('rouge1'))
    rouge2 = np.amax(scores.get('rougeL'))
    rougeScores.append(rouge2)
    rougeScores.append(rouge1)
    text.insert(END,"Summarization Scores\n\n")
    text.insert(END,"Rouge Score without Firefly : "+str(rouge2)+"\n")
    text.insert(END,"Rouge Score with Firefly Optimization: "+str(rouge1)+"\n")

def predict():
    text.delete('1.0', END)
    global article, summaries, tfidf, tfidf_results, tf1
    data = tf1.get()
    tf1.delete(0, END)
    article = nltk.sent_tokenize(data)
    tfidf = TfidfVectorizer(tokenizer = get_lemmatized_tokens, stop_words = stopwords)
    tfidf_results = tfidf.fit_transform(article).toarray()
    ff = Firefly(tfidf_results, tf.easom_function, np.min(tfidf_results), np.max(tfidf_results), tfidf_results.shape[1], 1, 1, 1, 1, 0.1, 0, 0.1)
    tfidf_results = ff.get_agents()
    tfidf_results = np.asarray(tfidf_results)
    tfidf_results = tfidf_results[0]
    predict = get_summary(article, tfidf_results, tfidf)
    text.insert(END,"Input Test = "+data+"\n\n")
    text.insert(END,"Predicted Summary = "+predict)
    

def close():
    main.destroy()

def graph():
    global rougeScores
    labels = ['Non Firefly', 'Firefly Optimization']
    height = rougeScores
    bars = labels
    y_pos = np.arange(len(bars))
    plt.figure(figsize = (4, 3)) 
    plt.bar(y_pos, height)
    plt.xticks(y_pos, bars)
    plt.xlabel("Algorithm Names")
    plt.ylabel("Rouge Score")
    plt.title("Rouge Score Comparison Between Non-Firefly & Firefly Optimization")
    plt.xticks()
    #plt.tight_layout()
    plt.show()

    
def gui():
    global text, pathlabel, main, tf1
    main = tkinter.Tk()
    main.title("Text Summarization using Firefly Algorithm") #designing main screen
    main.geometry("1300x1200")

    font = ('times', 16, 'bold')
    title = Label(main, text='Text Summarization using Firefly Algorithm')
    title.config(bg='brown', fg='white')  
    title.config(font=font)           
    title.config(height=3, width=120)       
    title.place(x=0,y=5)

    font1 = ('times', 13, 'bold')
    uploadButton = Button(main, text="Upload CNN-Daily Mail Dataset", command=uploadDataset)
    uploadButton.place(x=50,y=100)
    uploadButton.config(font=font1)  

    pathlabel = Label(main)
    pathlabel.config(bg='brown', fg='white')  
    pathlabel.config(font=font1)           
    pathlabel.place(x=360,y=100)

    preprocessButton = Button(main, text="Preprocess Dataset", command=preprocessDataset)
    preprocessButton.place(x=50,y=150)
    preprocessButton.config(font=font1) 

    vectorButton = Button(main, text="Convert Text to Vector", command=runVector)
    vectorButton.place(x=330,y=150)
    vectorButton.config(font=font1) 

    fireflyButton = Button(main, text="Train Summarization using Firefly", command=runFirefly)
    fireflyButton.place(x=650,y=150)
    fireflyButton.config(font=font1) 

    graphButton = Button(main, text="Comparison Graph", command=graph)
    graphButton.place(x=50,y=200)
    graphButton.config(font=font1)

    predictButton = Button(main, text="Summarize from Test Data", command=predict)
    predictButton.place(x=330,y=200)
    predictButton.config(font=font1)

    exitButton = Button(main, text="Exit", command=close)
    exitButton.place(x=650,y=200)
    exitButton.config(font=font1)

    l1 = Label(main, text='Input Text:')
    l1.config(font=font)
    l1.place(x=50,y=250)

    tf1 = Entry(main,width=70)
    tf1.config(font=font)
    tf1.place(x=160,y=250)


    font1 = ('times', 12, 'bold')
    text=Text(main,height=19,width=150)
    scroll=Scrollbar(text)
    text.configure(yscrollcommand=scroll.set)
    text.place(x=10,y=300)
    text.config(font=font1)
    main.config(bg='brown')
    main.mainloop()

if __name__ == "__main__":
    gui()
