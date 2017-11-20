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


"""
Vocab_Wemb={}
Voc_file=open("Vocab_Wemb.txt","r")
for Voc_line in Voc_file:
    Voc_line=Voc_line.strip()
    Words=Voc_line.split()
    Emb=ast.literal_eval[Words[1:]]

    Vocab_Wemb.update({Words[0]:Emb})

print ("len of Vocab embeddings is: ", len(Vocab_Wemb))
"""

#becky_emb=open("ss_qz_04.dim50vecs.txt","r", encoding='utf-8')
embeddings_index = {}
# glove_emb = open('glove.6B.100d.txt','r', encoding='utf-8')
f = open('glove.840B.300d.txt','r', encoding='utf-8')
# f = open("GW_vectors.txt", 'r', encoding='utf-8')  ## gives a lot lesser performance.

#f = open('ss_qz_04.dim50vecs.txt')
for line in f:
    values = line.split()
    word = values[0]
    try:
       coefs = np.asarray(values[1:], dtype='float32')
       emb_size=coefs.shape[0]
    except ValueError:
       print (values[0])
       continue
    embeddings_index[word] = coefs
print("Word2vc matrix len is : ",len(embeddings_index))
print("Embedding size is: ", emb_size)



IDF_file=open("IDF_file_dev.txt","r")
for line in IDF_file:
    IDF=ast.literal_eval(line)
    break

# print ("questions length is: ", len(Question_set))
# print ("Candidate answers set length is: ", len(Candidate_answers[2]))
print ("len of IDF is: ",len(IDF))



def Ques_Emb(ques1, IDF, embeddings_index):
    Ques_Matrix = np.empty((0, emb_size), float)
    IDF_Mat = np.empty((0, 1), float)  ##### IDF is of size = 1 coz its a value
    for q_term in ques1:
        if q_term in embeddings_index.keys():
           Ques_Matrix = np.append(Ques_Matrix, np.array([embeddings_index[q_term]]), axis=0)
           IDF_Mat = np.append(IDF_Mat, np.array([[IDF[q_term]]]), axis=0)

    return Ques_Matrix, IDF_Mat



def Word2Vec_score(curr_ques, Cand_ans, IDF, Word_Embs):
    max_score=0
    min_score=0
    Cand_ans_score=[]

    threshold_vals = len(curr_ques)  ## math.ceil     math.ceil(0.75 * float()
    # print("threshold value is: ",threshold_vals)
    Ques_Matrix, Ques_IDF = Ques_Emb(curr_ques, IDF, Word_Embs)



    for cand_a1 in Cand_ans:
        Cand_Matrix, Cand_IDF = Ques_Emb(cand_a1, IDF, Word_Embs)
        Cand_Matrix = Cand_Matrix.transpose()
        if Cand_Matrix.size == 0 or Ques_Matrix.size == 0:
            pass
        else:

            Score = np.matmul(Ques_Matrix, Cand_Matrix)
            max_indices = np.argmax(Score, axis=1)
            min_indices = np.argmin(Score, axis=1)

            max_score = np.amax(Score, axis=1)
            max_score = np.multiply(np.transpose(Ques_IDF), max_score)
            max_score1 = np.asarray(max_score).flatten()
            max_score1 = heapq.nlargest(threshold_vals, max_score1)
            max_score = (sum(max_score1))

            min_score = np.amin(Score, axis=1)
            min_score = np.multiply(np.transpose(Ques_IDF), min_score)
            min_score = np.asarray(min_score).flatten()
            min_score = heapq.nsmallest(threshold_vals, min_score)  ## threshold=2
            min_score = (sum(min_score))  # .item(0)
            total_score = max_score # (min_score)
            total_score = total_score ## / len(curr_ques)
            Cand_ans_score.append(total_score)
    # print("Can scores are: ", Cand_ans_score)
    Cand_ans_score=np.asarray(Cand_ans_score)
    predicted_val=np.argmax(Cand_ans_score)

    return predicted_val




import xml.etree.ElementTree as ET
tree = ET.parse('cqa_questions_yadeep_min4_causal.cqa.dev.xml')
root = tree.getroot()



Question_set=[]
Candidate_answers=[]
Cand_ans=[]
Correct_ans=[]
Predicted_ans=[]

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

    Predictions=Word2Vec_score(curr_ques, Cand_ans, IDF, embeddings_index)
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
"""
print (len(Question_set))
print (len(Candidate_answers))
print (len(Correct_ans))

"""