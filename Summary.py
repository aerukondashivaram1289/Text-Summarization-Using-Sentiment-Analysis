import os
from tkinter import *
import tkinter
from tkinter import filedialog
from tkinter.filedialog import askopenfilename
import os
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from string import punctuation
import re
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


main = tkinter.Tk()
main.title("Text Summarization using Sentiment Analysis")
main.geometry("1300x900")

global model, tokenizer, device, process_text

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
global sentiment_model

def cleanText(doc):
    tokens = doc.split()
    table = str.maketrans('', '', punctuation)
    tokens = [w.translate(table) for w in tokens]
    tokens = [w for w in tokens if not w in stop_words]
    tokens = [word for word in tokens if len(word) > 1]
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    tokens = ' '.join(tokens)
    tokens = re.sub(r'\d+', '', tokens)
    return tokens

def loadModel():
    global model, tokenizer, device, sentiment_model
    sentiment_model = SentimentIntensityAnalyzer()
    model = T5ForConditionalGeneration.from_pretrained('t5-small')
    tokenizer = T5Tokenizer.from_pretrained('t5-small',model_max_length=512)
    device = torch.device('cpu')
    pathlabel.config(text="Sentiment Summary Model Loaded")
    text.delete('1.0', END)
    text.insert(END,"Model loaded\n")

def preprocessText():
    global process_text
    text_data = inputtext.get(1.0, "end-1c")
    print(text_data)
    process_text = cleanText(text_data)
    text.delete('1.0', END)
    text.insert(END,process_text)
    process_text = text_data

def sentimentSummary():
    global model, tokenizer, device, process_text, sentiment_model
    preprocessedText = process_text.strip().replace('\n','')
    process_text = 'summarize: ' + process_text
    tokenizedText = tokenizer.encode(process_text, return_tensors='pt', max_length=512, truncation=True).to(device)
    summaryIds = model.generate(tokenizedText, min_length=30, max_length=120)
    summary = tokenizer.decode(summaryIds[0], skip_special_tokens=True)
    sentiment_dict = sentiment_model.polarity_scores(summary)
    compound = sentiment_dict['compound']
    result = ''
    if compound >= 0.05 : 
        result = 'Positive'   
    elif compound <= - 0.05 : 
        result = 'Negative'   
    else : 
        result = 'Neutral'    
    text.delete('1.0', END)
    text.insert(END,"Extracted Summary : "+summary+"\n\n")
    text.insert(END,"Predicted Sentiments : "+str(result))

def clearText():
    text.delete('1.0', END)
    inputtext.delete('1.0', END)

font = ('times', 16, 'bold')
title = Label(main, text='Text Summarization using Sentiment Analysis',anchor=W, justify=LEFT)
title.config(bg='black', fg='white')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)


font1 = ('times', 13, 'bold')

loadButton = Button(main, text="Generate & Load Summary Sentiment Model", command=loadModel)
loadButton.place(x=50,y=100)
loadButton.config(font=font1)  

pathlabel = Label(main)
pathlabel.config(bg='DarkOrange1', fg='white')  
pathlabel.config(font=font1)           
pathlabel.place(x=450,y=100)

inputlabel = Label(main, text='Input Your Text Below')
inputlabel.config(bg='DarkOrange1', fg='white')  
inputlabel.config(font=font1)           
inputlabel.place(x=420,y=150)

inputtext = Text(main,height=10,width=120)
scroll=Scrollbar(inputtext)
inputtext.configure(yscrollcommand=scroll.set)
inputtext.place(x=10,y=200)
inputtext.config(font=font1)


preprocessButton = Button(main, text="Preprocess Text", command=preprocessText)
preprocessButton.place(x=50,y=400)
preprocessButton.config(font=font1)

annotationButton = Button(main, text="Generate Summary using Sentiment", command=sentimentSummary)
annotationButton.place(x=230,y=400)
annotationButton.config(font=font1)

clearButton = Button(main, text="Clear Text", command=clearText)
clearButton.place(x=520,y=400)
clearButton.config(font=font1)

text=Text(main,height=10,width=120)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=10,y=450)
text.config(font=font1)

main.config(bg='chocolate1')
main.mainloop()

