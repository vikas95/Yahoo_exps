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
Vocab_file=open("Vocab.txt","r")
for line1 in Vocab_file:
    All_words=ast.literal_eval(line1)
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
"""
term_absent=0
for term1 in All_words:
    if term1 in embeddings_index.keys():
       pass
    else:
       term_absent+=1
print("number of terms not present in W2V is:  ", term_absent)
mat1=np.matrix([[4,5],[6,7]])
mat2=np.matrix([[1,5],[2,7],[3,6]])
mat2=mat2.transpose()
c=np.matmul(mat1,mat2)
d=np.amax(c,axis=1)
e=np.amin(c,axis=1)
idf=numpy.matrix([[2],[3]])
g=np.multiply(idf,c)  ####### elementwise matrix multiplication.


taking top n elements from a column of a matrix:
d=np.amax(c,axis=1)
g=heapq.nlargest(d,threshold)
Score=sum(g)

"""



file2=open("IDF.txt","r")
for line2 in file2:
    IDF=ast.literal_eval(line2)



def Word2Vec_score(Question, IDF_Mat, Corpus, IDF, Justification_threshold):

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



def Ques_Emb(ques1, IDF):
    Ques_Matrix = np.empty((0, emb_size), float)
    IDF_Mat = np.empty((0, 1), float)  ##### IDF is of size = 1 coz its a value
    for q_term in ques1:
        if q_term in embeddings_index.keys():
           Ques_Matrix = np.append(Ques_Matrix, np.array([embeddings_index[q_term]]), axis=0)
           IDF_Mat = np.append(IDF_Mat, np.array([[IDF[q_term]]]), axis=0)

    return Ques_Matrix, IDF_Mat


Question_file = open('training_set.tsv', 'r')
 #[]
Correct_ans = []#[]
All_questions = []
IDF_Mat=[]


counter=0
file1=open("structured_kerasInput_train_bestIR_08j5.tsv","r")
Final_scores=[]
All_terms=[]
All_Ques_terms=[]


for line1 in Question_file:
    counter+=1
    #print(counter)
    Question = ""
    Option_A = ""  # []  ####### These will contain justification text also and later on, becky features will be added.
    Option_B = ""  # []
    Option_C = ""  # []
    Option_D = ""
    Cand_score = []
    line1 = line1.strip()
    cols = line1.split("\t")
    Correct_ans.append(cols[3])
    A_index = cols[10].index("(A)")
    B_index = cols[10].index("(B)")
    C_index = cols[10].index("(C)")
    D_index = cols[10].index("(D)")

    Question = (cols[10][:A_index - 1])
    Option_A = (cols[10][A_index + 4:B_index - 1])
    Option_B = (cols[10][B_index + 4:C_index - 1])
    Option_C = (cols[10][C_index + 4:D_index - 1])
    Option_D = (cols[10][D_index + 4:])

    Question = tokenizer.tokenize(Question.lower())
    Question=[lmtzr.lemmatize(w1) for w1 in Question]
    Question = [w for w in Question if not w in stop_words]

    Option_A = tokenizer.tokenize(Option_A.lower())
    Option_A = [lmtzr.lemmatize(w1) for w1 in Option_A]
    Option_A = [w for w in Option_A if not w in stop_words]

    Option_B = tokenizer.tokenize(Option_B.lower())
    Option_B = [lmtzr.lemmatize(w1) for w1 in Option_B]
    Option_B = [w for w in Option_B if not w in stop_words]

    Option_C = tokenizer.tokenize(Option_C.lower())
    Option_C = [lmtzr.lemmatize(w1) for w1 in Option_C]
    Option_C = [w for w in Option_C if not w in stop_words]

    Option_D = tokenizer.tokenize(Option_D.lower())
    Option_D = [lmtzr.lemmatize(w1) for w1 in Option_D]
    Option_D = [w for w in Option_D if not w in stop_words]

    All_Ques_terms=Question+Option_A+Option_B+Option_C+Option_D
    All_Ques_terms=(set(All_Ques_terms))
    All_terms.append(All_Ques_terms)
    All_Ques_terms=[]


    Ques1 = Question + Option_A  ###### Question + Candidate answer 1
    Q1_matrix, IDF_Mat1=Ques_Emb(Ques1, IDF)

    Ques2 = Question + Option_B
    Q2_matrix, IDF_Mat2 = Ques_Emb(Ques2, IDF)

    Ques3 = Question + Option_C
    Q3_matrix, IDF_Mat3 = Ques_Emb(Ques3, IDF)

    Ques4 = Question + Option_D
    Q4_matrix, IDF_Mat4 = Ques_Emb(Ques4, IDF)

    if counter<=2:
       print (Ques1)


    All_questions += [Q1_matrix, Q2_matrix, Q3_matrix, Q4_matrix]
    IDF_Mat += [IDF_Mat1, IDF_Mat2, IDF_Mat3, IDF_Mat4]



J_Threshold=5
Score_matrix, Justification_matrix = Word2Vec_score(All_questions, IDF_Mat,  file1, IDF, J_Threshold)

# print(Score_matrix)
out_file=open("Becky_files_W2V_score_"+str(J_Threshold)+"Final_T1"+".txt","w")
out_file.write(str(Score_matrix))


"""
Score_matrix, Justification_matrix = Word2Vec_score(All_questions, IDF_Mat,  file1, 3)
out_file=open("Becky_files_W2V_score_"+str(3)+"Final"+".txt","w")
out_file.write(str(Score_matrix))

"""
