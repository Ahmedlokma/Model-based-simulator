import csv
from mimetypes import init 
import pandas as pd 
import json 
import random 
import nltk 
from nltk.translate.bleu_score import sentence_bleu  
from nltk.tokenize.treebank import TreebankWordDetokenizer
import time 
from csv import reader

class my_metrics :
  def __init__(self):
   pass

  def intial(self):

   f = False
   with open('/Users/ahmedlokma/Desktop/user-simulator-master/sequicity_user/results/wed22.csv', 'r') as read_obj:

    l = []
    csv_reader = reader(read_obj)

    for row in csv_reader:

        if(row[0] == 'START_CSV_SECTION') :
           f = True 
           print(row)
        #    break 
        elif (f == False) :
            c = '4'
        else :
         l.append(row)

   with open('/Users/ahmedlokma/Desktop/user-simulator-master/sequicity_user/results/wed22.csv', 'w') as f: 
     write = csv.writer(f) 
     write.writerows(l)        
   v = self.match_metric_rate()
   vv = self.bleu_metric_rate()
        

  def match_metric_rate(self):

   match,total = 0,1e-8
   col_list = ["response","generated_response"]
   df = pd.read_csv("/Users/ahmedlokma/Desktop/user-simulator-master/sequicity_user/results/wed22.csv", usecols=col_list)

   for i in range(len(df["response"])):
    question = df['response'][i].lower()
    question2 = df['generated_response'][i].lower()
    nltk_tokens = nltk.word_tokenize(question)
    nltk_tokens2 = nltk.word_tokenize(question2)
    z = TreebankWordDetokenizer().detokenize(nltk_tokens)
    z2 = TreebankWordDetokenizer().detokenize(nltk_tokens2)
    if(z == z2):
          match+=1
    total += 1
   x =  print(match /total) 
   return x 



  def bleu_metric_rate(self):
    fac = 1.05
    x = 0
    col_list = ["response","generated_response"]
    df = pd.read_csv("/Users/ahmedlokma/Desktop/user-simulator-master/sequicity_user/results/wed22.csv", usecols=col_list)
    for i in range(len(df["response"])):
     question = df['response'][i].lower()
     question2 = df['generated_response'][i].lower()
     nltk_tokens = nltk.word_tokenize(question)
     nltk_tokens2 = nltk.word_tokenize(question2)
     reference = [
        nltk_tokens
      ]
     candidate = nltk_tokens2
    #  print("------------------------------")
    #  print(nltk_tokens)
    #  print(nltk_tokens2)
    #  print("------------------------------")
     x += sentence_bleu(reference, candidate)

    print((x/len(df['response']))/fac)
    vv = (x/len(df['response'])/fac)
    return vv 
    
user = my_metrics()  
user.intial()
        

