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
# f = open('glove.840B.300d.txt','r', encoding='utf-8')
# f = open("GW_vectors.txt", 'r', encoding='utf-8')  ## gives a lot lesser performance.
f=open("deps.contexts","r", encoding='utf-8')
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



IDF_file=open("IDF_file.txt","r")
for line in IDF_file:
    IDF=ast.literal_eval(line)
    break

# print ("questions length is: ", len(Question_set))
# print ("Candidate answers set length is: ", len(Candidate_answers[2]))
print ("len of IDF is: ",len(IDF))

def Word2Vec_score(curr_ques, Cand_ans, IDF):

    Doc_Score=[0]


    max_score=0
    min_score=0
    #Ques_score=[]
    Justification_set=[]
    Document_score=[[0] for i in range(len(Question))]
    Justification_ind = [[0] for i in range(len(Question))]
    #SCORES=[]


    for Jind, Justifications in enumerate(Corpus):

        threshold_vals=1
        if Jind%1000==0:
           print (Jind)
           # print(threshold_vals)
        Justification_set = []
        Justifications = Justifications.strip()
        cols = Justifications.split("\t")  ## cols[0] has the question number, cols[1]  has the candidate option number for that specific question.
        Feature_col = cols[6].split(";;")
        # print (len(Feature_col))
        if len(Feature_col) >= Justification_threshold:
            for ind1 in range(Justification_threshold):  #### we take only top 10 justifications.
                ##["AggregatedJustification"]["text"]
                dict1 = ast.literal_eval(Feature_col[ind1])
                Justification_set.append((dict1["AggregatedJustification"]["text"]).lower())

        for just_ind, just1 in enumerate(Justification_set):
            Doc_set = tokenizer.tokenize(just1)
            # Doc_set=list(set(Doc_set))
            # Doc_set = [lmtzr.lemmatize(w1) for w1 in Doc_set]
            Doc_set = [w for w in Doc_set if not w in stop_words]

            Doc_Matrix = np.empty((0, emb_size), float)  ####################### DIMENSION of EMBEDDING
            Doc_len=0
            for key in Doc_set:
                if key in embeddings_index.keys():
                   Doc_Matrix=np.append(Doc_Matrix, np.array([embeddings_index[key]]), axis=0)
                   Doc_len+=1
            if Doc_Matrix.size==0:
               pass
            else:

                Doc_IDF_Mat = np.empty((0, 1), float)
                Doc_IDF_Mat_min = np.empty((0, 1), float)

                Doc_Matrix=Doc_Matrix.transpose()
                #print(Doc_Matrix.shape)
                ques1=Question[Jind]
                threshold_vals =  ques1.shape[0] ## math.ceil     math.ceil(0.75 * float()

                Score=np.matmul(ques1,Doc_Matrix)
                max_indices = np.argmax(Score, axis=1)
                min_indices = np.argmin(Score, axis=1)
                max_list=[]
                for mind1 in max_indices:
                    if Doc_set[mind1] in IDF.keys():
                        Doc_IDF_Mat = np.append(Doc_IDF_Mat, np.array([[IDF[Doc_set[mind1]]]]), axis=0)
                        max_list.append(Doc_set[mind1])
                    else:
                        Doc_IDF_Mat = np.append(Doc_IDF_Mat, np.array([[5.379046132954042]]), axis=0)

                #if Jind<8:
                   #print (max_list)
                   #print (ques1.shape)
                for mind1 in min_indices:
                    if Doc_set[mind1] in IDF.keys():
                        Doc_IDF_Mat_min = np.append(Doc_IDF_Mat_min, np.array([[IDF[Doc_set[mind1]]]]), axis=0)
                    else:
                        Doc_IDF_Mat_min= np.append(Doc_IDF_Mat_min, np.array([[5.379046132954042]]), axis=0)


                max_score=np.amax(Score,axis=1)
                #print("after taking max elements only ",max_score.shape)

                #print(ques1.shape," ",max_score.shape)
                #print(IDF_Mat[Jind])
                #print(max_score)
                max_score=np.multiply(np.transpose(IDF_Mat[Jind]),max_score)
                #print("After multiplying query term IDF ",max_score.shape)
                #print(max_score)
                #max_score = np.multiply(Doc_IDF_Mat, max_score) ### Becky suggestion which is not working
                max_score1 = np.asarray(max_score).flatten()
                # max_score=np.sort(max_score)  ### not required as heapq takes care of that...

                max_score1 = heapq.nlargest(threshold_vals,max_score1) ## threshold=2
                #print(ques1.shape," "," ", len(max_score))

                max_score=(sum(max_score1))#.item(0)

                #print (max_score)
                min_score=np.amin(Score,axis=1)
                min_score = np.multiply(np.transpose(IDF_Mat[Jind]), min_score)
                # min_score = np.multiply(Doc_IDF_Mat_min, min_score)  ### Becky suggestion which is not working
                min_score = np.asarray(min_score).flatten()
                min_score = heapq.nsmallest(threshold_vals,min_score)  ## threshold=2
                min_score=(sum(min_score))#.item(0)
                total_score=max_score + (min_score)
                total_score=total_score/float(ques1.shape[0])
                Document_score[Jind].append(total_score)
                Justification_ind[Jind].append(just_ind)


    return Document_score, Justification_ind



import xml.etree.ElementTree as ET
tree = ET.parse('cqa_questions_yadeep_min4_causal.cqa.train.xml')
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

    Predictions=Word2Vec_score(curr_ques, Cand_ans, IDF)
    Predicted_ans.append(Predictions)

    Cand_ans=[]
    count=0


"""
print (len(Question_set))
print (len(Candidate_answers))
print (len(Correct_ans))

"""