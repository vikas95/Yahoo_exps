import heapq
import math
import ast
import numpy as np
from collections import Counter
from nltk.tokenize import RegexpTokenizer  ### for nltk word tokenization
tokenizer = RegexpTokenizer(r'\w+')
from nltk.stem.wordnet import WordNetLemmatizer
lmtzr = WordNetLemmatizer()

import nltk
from nltk.corpus import stopwords
stop_words=stopwords.words('english')
stop_words=[lmtzr.lemmatize(w1) for w1 in stop_words]
stop_words=list(set(stop_words))
# stop_words=[]
print(stop_words)
print(len(stop_words))


import xml.etree.ElementTree as ET
tree = ET.parse('corr_dev.xml')
root = tree.getroot()

def Len_algo(curr_ques, Cand_ans):
    prediction=[]
    for cand1 in Cand_ans:
        prediction.append(len(cand1))

    prediction=np.asarray(prediction)
    pred_val=np.argmax(prediction)
    return pred_val


Question_set=[]
Candidate_answers=[]
Cand_ans=[]
Correct_ans=[]
Predicted_ans=[]
Positive_term_threshold=4
Negative_term_threshold=2
for question in root:
    curr_ques=question.find('text').text
    curr_ques=tokenizer.tokenize(curr_ques.lower())
    curr_ques = [lmtzr.lemmatize(w1) for w1 in curr_ques]
    curr_ques = [w for w in curr_ques if not w in stop_words]

    Question_set.append(curr_ques)
    count=0
    for answer in question.find("answers").findall("answer"):
        curr_cand=answer.find("text").text
        curr_cand = tokenizer.tokenize(curr_cand.lower())

        curr_cand = [lmtzr.lemmatize(w1) for w1 in curr_cand]
        curr_cand = [w for w in curr_cand if not w in stop_words]

        Cand_ans.append(curr_cand)
        if answer.find("gold").text=="true":
           Correct_ans.append(count)
        count+=1

    Candidate_answers.append(Cand_ans)

    Predictions=Len_algo(curr_ques, Cand_ans) # , Positive_term_threshold, Negative_term_threshold)
    Predicted_ans.append(Predictions)

    Cand_ans=[]
    count=0





accuracy=0
if len(Predicted_ans)!=len(Correct_ans):
   print("there is a big error somewhere, find it: ")

else:
   for ind1, pred1 in enumerate(Predicted_ans):
       if pred1==Correct_ans[ind1]:
          accuracy+=1

print("accuracy or P@1 is: ",accuracy/float(len(Predicted_ans)))
print(Correct_ans)

print(len(Correct_ans))