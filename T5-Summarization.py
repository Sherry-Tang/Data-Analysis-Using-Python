# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 12:12:56 2020

@author: sherry
"""
#Load Packages
from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance
import numpy as np
import networkx as nx
import nltk
import seaborn as sns
import matplotlib.pyplot as plt
from transformers import T5Tokenizer, T5Config, T5ForConditionalGeneration
from newspaper import Article
# Traditional Summarization with textrank ########################
# Teaditianal Approach generate summarization by generating similarity matrix and 
# ranking all sentences by texkrank algorithm, and pick top sentences
#%%
#Approach1: Newspaper
# Step1 - Get the aritle
url='https://www.cnbc.com/2020/12/11/hyundai-motor-to-buy-robot-maker-boston-dynamics-from-softbank.html'
article=Article(url)
# Step2 - Do some NLP
article.download()
article.parse()
article.nlp()
article.authors
# Step3 - Get the publish date
print("Publish Date:",article.publish_date)
print('--------------------------------------')
# Step4 - Get the top image
article.top_image
# Step5 - Get the article text
article.text
#Step6 - Get a summary of the article
print(article.summary)
#%%
#Approach2: NLTK
# Define a function to load file
def read_article(file_name):
    file = open(file_name, "r",encoding='UTF-8')
    filedata = file.readlines()
    article = filedata[0].split(". ")
    sentences = []
    for sentence in article:
        sentences.append(sentence.replace("[^a-zA-Z]", " ").split(" "))
    sentences.pop()    
    return sentences
# Define a function to calculate similarity between each sentences
def sentence_similarity(sent1, sent2, stopwords=None):
    if stopwords is None:
        stopwords = []
    sent1 = [w.lower() for w in sent1]
    sent2 = [w.lower() for w in sent2] 
    all_words = list(set(sent1 + sent2)) 
    vector1 = [0] * len(all_words)
    vector2 = [0] * len(all_words)
    for w in sent1:
        if w in stopwords:
            continue
        vector1[all_words.index(w)] += 1
    for w in sent2:
        if w in stopwords:
            continue
        vector2[all_words.index(w)] += 1
    return 1 - cosine_distance(vector1, vector2)
# Define a function to create similarity matrix
def build_similarity_matrix(sentences, stop_words):
    similarity_matrix = np.zeros((len(sentences), len(sentences))) 
    for idx1 in range(len(sentences)):
        for idx2 in range(len(sentences)):
            if idx1 == idx2: 
                continue 
            similarity_matrix[idx1][idx2] = sentence_similarity(sentences[idx1], sentences[idx2], stop_words)
    return similarity_matrix
def generate_summary(file_name, top_n=5):
    stop_words = stopwords.words('english')
    summarize_text = []
    # Step 1 - Read text anc split it
    sentences =  read_article(file_name)
    # Step 2 - Generate Similary Martix across sentences
    sentence_similarity_martix = build_similarity_matrix(sentences, stop_words)
    # Step 3 - Rank sentences in similarity martix
    sentence_similarity_graph = nx.from_numpy_array(sentence_similarity_martix)
    scores = nx.pagerank(sentence_similarity_graph)
    # Step 4 - Sort the rank and pick top sentences
    ranked_sentence = sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse=True)        
    for i in range(top_n):
      summarize_text.append(" ".join(ranked_sentence[i][1]))
    # Step 5 - Output the summarize text
    print("Summarize Text: \n", ". ".join(summarize_text))    
a=generate_summary('news.txt', top_n=2)
#%%
# Deep Learning Approach with T5 ########################
#Deep learning model generate summary based on the semantic understanding of
#original documents, T5 was trained on C4 dataset-a comprehensive and large 
# dataset developed by google, and it has both encoder and decoder blocks which
# guarantees its outstanding performance.
from transformers import T5Tokenizer, T5Config, T5ForConditionalGeneration
# Step1 - load the local txt file, you can also integrate different python packages such as “Fransc” to transform other source of docuent into text file.
file_name="news.txt"
file = open(file_name, "r",encoding='UTF-8')
filedata = file.readlines() 
preprocess_text = filedata[0].strip().replace("\n","")
original_text=preprocess_text.copy()
# Step2 - Initiate the model, we used t5-large here. There are more advanced T5 versions such as t5-3b and t5-11b which will give you  better performance,
#  but they will also take a lot of RAM, so make sure that you have enough space, and it will also take a longer time.
my_model = T5ForConditionalGeneration.from_pretrained('t5-large') 
# Step3 -Load tokenizer
tokenizer = T5Tokenizer.from_pretrained('t5-large')               
# Step4 - define the instruction by adding the prefix"Summarize" 
text = "Summarize" + original_text                                # s
# You can replace "Summarize" by "translate English to other languages:"  to translate the summary into different language
#Step5 - Encode text 
tokenized_text = tokenizer.encode(text, return_tensors="pt")
#Step6 - Generate summary
summary_ids = my_model.generate(
            tokenized_text,
            max_length=300,
            min_length=100,
            num_beams=3,
            repetition_penalty=2.5, 
            length_penalty=2, 
            early_stopping=True
        )
#Step7 - Decoder the summary and print it out 
output = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

print ("\n\nSummarized text: \n",output)

# T5 is powerful,outsource and simple to use, it is a good tool to improve your daily work efficiency!








