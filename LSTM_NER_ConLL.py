from keras.models import Sequential
import numpy as np
from keras.layers.recurrent import LSTM
from keras.layers.core import TimeDistributedDense, Activation
from keras.preprocessing.sequence import pad_sequences
from keras.layers.embeddings import Embedding
from keras.layers import Merge
from keras.layers import Dropout
from keras.layers.wrappers import Bidirectional
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

import unicodedata

Train_file="ned.FullTrain"
Test_file="ned.testb"

#############################################################################################3
def Max_Val_Cal(Doc1):
    raw = open(Doc1, 'r',encoding="ISO-8859-1").readlines()
    all_x = []
    point = []
    for line in raw:
        stripped_line = line.strip().split(' ')
        point.append(stripped_line)
        if line == '\n':
           all_x.append(point[:-1])
           point = []
    lengths = [len(x) for x in all_x]
    return max(lengths)

Train_Max=Max_Val_Cal(Train_file)
Test_Max=Max_Val_Cal(Test_file)
maxlen=max(Train_Max,Test_Max)
#maxlen=250

#############################################################################################3
### TRAINING DATA   ### TRAINING DATA   ### TRAINING DATA  ### TRAINING DATA

raw = open(Train_file, 'r',encoding="ISO-8859-1").readlines()
all_x = []
point = []
for line in raw:
    stripped_line = line.strip().split(' ')
    point.append(stripped_line)
    if line == '\n':
       all_x.append(point[:-1])
       point = []    
    
X = [[c[0] for c in x] for x in all_x]
y = [[c[2] for c in y] for y in all_x]

all_text = [c for x in X for c in x]
words = list(set(all_text))
word2ind = {word: index+1 for index, word in enumerate(words)}
ind2word = {index+1: word for index, word in enumerate(words)}

labels = list(set([c for x in y for c in x]))    
label2ind = {label: (index) for index, label in enumerate(labels)}
ind2label = {(index): label for index, label in enumerate(labels)}
    

def encode(x, n):
    result = np.zeros(n)
    result[x] = 1
    return result
X_enc = [[word2ind[c] for c in x] for x in X]


max_label = max(label2ind.values())+1

X_enc_f = pad_sequences(X_enc, maxlen=maxlen)

y_enc = [[label2ind[c] for c in ey] for ey in y]
y_enc = [[encode(c, max_label) for c in ey] for ey in y_enc]
y_enc = pad_sequences(y_enc, maxlen=maxlen)

X_train_f=X_enc_f
y_train=y_enc

#############################################################################################3
### TESTING DATA   ### TESTING DATA   ### TESTING DATA  ### TESTING DATA
    
raw = open(Test_file, 'r',encoding="ISO-8859-1").readlines()
all_x = []
point = []
Words_inLines=[]
for line in raw:
    stripped_line = line.strip().split(' ')
    point.append(stripped_line)
    if line == '\n':
       Words_inLines.append(len(point)-1)  ## -1 because we are counting blank line as an element in the list. So for removing it, -1 is required.
       all_x.append(point[:-1])
       point = []    
    
X_t = [[c[0] for c in x] for x in all_x]
y_t = [[c[2] for c in y] for y in all_x]

Testlabels = list(set([c for x in y for c in x]))




################################################
all_text = [c for x in X_t for c in x]
words1 = list(set(all_text))
LastIndex=max(ind2word)
words1_2ind={}
Dict_count=0
for i in range(len(words1)):
    if words1[i] in words:       
       words1_2ind.update({words1[i]: words.index(words1[i])+1})
       Dict_count=Dict_count+1
    else:
       words1_2ind.update({words1[i]: LastIndex+1})
       LastIndex=LastIndex+1  


##################################### Not used anywhere
LastLabelIndex=max(ind2label)
Test_label2ind=[]
for i in range(len(Testlabels)):
    if Testlabels[i] in labels:
       Test_label2ind.append({Testlabels[i]:labels.index(Testlabels[i])})
    else:
       Test_label2ind.append({Testlabels[i]:LastLabelIndex+1})
       LastLabelIndex=LastLabelIndex+1  
###################################################


X_enc_test = [[words1_2ind[c] for c in x] for x in X_t]

y_enc_t = [[label2ind[c] for c in ey] for ey in y_t]
y_enc_t = [[encode(c, max_label) for c in ey] for ey in y_enc_t]
y_enc_t = pad_sequences(y_enc_t, maxlen=maxlen)


X_test_f = pad_sequences(X_enc_test, maxlen=maxlen)

######################################################################################################


max_features = 1+max(max(word2ind.values()),max(words1_2ind.values())) ## doubt - why maxfeatures is taken len of indices

# max_features = len(word2ind)
# out_size = len(label2ind) + 1
embedding_size = 256
embedding_size_POS = 32
embedding_size_case = 128
inputdim=embedding_size+embedding_size_POS  #+embedding_size_case

hidden_size = 64
out_size = max(label2ind.values())+1

model_forward = Sequential()
model_forward.add(Embedding(max_features, embedding_size, input_length=maxlen, mask_zero=True))

model_forward.add(Bidirectional(LSTM(hidden_size, return_sequences=True)))

model_forward.add(Dropout(0.15))
model_forward.add(TimeDistributedDense(out_size))

model_forward.add(Activation('softmax'))

#Final_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['precision', 'recall','fbeta_score','accuracy'])
model_forward.compile(loss='mse', optimizer='rmsprop')
batch_size = 32


# model.fit(X, Y, validation_split=0.33, nb_epoch=150, batch_size=10, validation_split=0.15, callbacks=callbacks_list, verbose=0)

model_forward.fit([X_train_f,POS_enc_train,Case_enc_train], y_train, batch_size=batch_size, nb_epoch=18) #5  #,Case_enc_train

# score1 = model_forward.evaluate(X_test_f, y_test, batch_size=batch_size)

pr = model_forward.predict_classes(X_test_f)

yh = y_enc_t.argmax(2)

thefile = open('PredFile_8tags_18', 'w')

for i in range(len(Words_inLines)):
    pred=pr[i][-Words_inLines[i]:]
    for pred_val in pred:
        thefile.write("%s\n" %ind2label[pred_val])
    thefile.write("\n")    


