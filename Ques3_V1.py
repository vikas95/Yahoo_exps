############# THis is Ques 2 where we have done smoothing

from collections import Counter
import numpy as np
import sys

# file_name=(sys.argv[1])
## file_name=open("PTBSmall/train.tagged")
############################################ TRAINING

file1 = open("PTBSmall/train.tagged","r")   ##### reading the file
unigrams=[]
bigrams=[]
POS=[]
Vocab=[] ## we need len of vocab for add 1 smoothing.
Word_tag_posting={}  ## we need this to determine what all tags does the word have in the history

prev_tag="start" ## st for start
end_tag="end"
word_len=[]
Error_lines=[]
for line1 in file1:
    line1=line1.strip()
    words1=line1.split()
    word_len.append(len(words1))
    if len(words1)>1:
       Vocab.append(words1[0])
       if words1[0] in Word_tag_posting.keys():
          if words1[1] in Word_tag_posting[words1[0]]:  ## this ensures that we are appending unique word in the posting list.
             pass
          else:
             Word_tag_posting[words1[0]].append(words1[1])  ## for the given word, we are appendings its seen tag.
       else:
          Word_tag_posting.update({words1[0]:[words1[1]]})

       unigrams.append(words1[0]+" "+words1[1])
       POS.append(words1[1])
       bigrams.append(prev_tag+" "+words1[1])
       prev_tag=words1[1]
    else:
       bigrams.append(prev_tag+" "+end_tag) ## coz after every line, we have to calculate end tag transition prob.
       prev_tag="start"
       POS.append("start")
       # if len()

Vocab=list(set(Vocab))
K=len(Vocab)
print(K)   ### We need K in smoothing part.
# print(Word_tag_posting)
print("number of distinct words in each line is ",list(set(word_len)))

Pos_count_dict=Counter(POS)
POS_set=list(set(POS))

# print (Pos_dict)

word_tag_count=Counter(unigrams)
print (word_tag_count)

trans_count=Counter(bigrams)

for trans_key in trans_count.keys():
    prev_tag1=trans_key.split()[0]
    trans_count[trans_key]=trans_count[trans_key]/float(Pos_count_dict[prev_tag1])  ## dividing by the count of previous tag

#print(trans_count)
########################### Smoothing of known word is done here.  This is done only for known word tag pair.
for word_key in word_tag_count:
    pos1=word_key.split()[1]
    word_tag_count[word_key]=(word_tag_count[word_key]+1)/float(Pos_count_dict[pos1]+K)  ## dividing by the count of same word tag




############################################ TRAINING complete

############################################ Validation Performance

val_file= open("PTBSmall/small_dev.txt","r")
prev_vtag="start"
Gold_val_tag=[]
Predicted_val_tag=[]
unknown_indices=[]
indices=0
Viterbi_Matrix=[]  ######## to store Viterbi array for each word
Back_pointer=[]   ####### to keep track to transition.
Back_prop_matrix=[]
for vline in val_file:
    # print (indices)
    vline=vline.strip()
    vwords=vline.split()
    if len(vwords)<2: ## new line, hence we should consider end tag here.
        prev_viterbi = Viterbi_Matrix[-1]  ## T-1 viterbi matrix

        possible_tags = POS_set
        prev_viterbi = np.asarray(prev_viterbi)
        viterbi = []
        back_prop1 = 0

        for pt1 in possible_tags:
            multi2 = []
            multi2.append(trans_count[pt1 + " " + "end"])


        multi1 = np.asarray(multi2)
        backprop_pot = np.multiply(prev_viterbi, multi2)
        back_prop1=(np.argmax(backprop_pot))
        # print(Back_prop_matrix)
        # traceback=Back_prop_matrix[back_prop1,:]
        for indexes in Back_prop_matrix:
            Predicted_val_tag.append(indexes[back_prop1])
        Predicted_val_tag.append(back_prop1)


        print ("len of Viterbi matrix, ", len(Viterbi_Matrix))
        Viterbi_Matrix=[]
        Back_prop_matrix = []

        prev_vtag="start"
    else:
        # print ("we should be here")
        Gold_val_tag.append(vwords[1])
        if prev_vtag=="start":  ## this means we are on the first word of the sentence
           Viterbi=[]

           Viterbi_Matrix = []
           Back_prop_matrix=[]

           Back_pointer = []
           possible_tags=POS_set
           multi1=[]
           for pt1 in possible_tags:

               emis_prob = (word_tag_count[vwords[0] + " " + pt1])  # smoothing of word tag count is done in the last segment.
               if emis_prob == 0:  ## when the word is never seen with "pt1" tag
                   emis_prob = 1 / (float(Pos_count_dict[pt1] + K))
               Viterbi.append(trans_count[prev_tag + " " + pt1] * emis_prob)
           # Back_pointer.append("start") ## instead of adding 0 which is the index, I am adding tag  ## wont be used in the accuracy calculation
           Viterbi_Matrix.append(Viterbi)
           prev_vtag="any"
        else:
            prev_viterbi=Viterbi_Matrix[-1] ## T-1 viterbi matrix
            prev_viterbi=np.asarray(prev_viterbi)

            viterbi=[]
            back_prop1 = []
            if vwords[0] in Vocab:

               # possible_tags=Word_tag_posting[vwords[0]]  ### now if we are smoothing, even if word is known, we still need to consider all possibilities for tags
               possible_tags = POS_set

               for pt1 in possible_tags:
                   multi1 = []
                   multi2 = []
                   for ss1 in possible_tags:
                       emis_prob = (word_tag_count[vwords[0] + " " + pt1])  # smoothing of word tag count is done in the last segment.
                       if emis_prob == 0:  ## when the word is never seen with "pt1" tag
                           emis_prob = 1 / (float(Pos_count_dict[pt1] + K))
                       multi1.append(trans_count[ss1 + " " + pt1] * emis_prob)
                       multi2.append(trans_count[ss1 + " " + pt1])
                   multi1=np.asarray(multi1)
                   multi1=np.asarray(multi2)
                   viterbi_pot=np.multiply(prev_viterbi,multi1)
                   backprop_pot=np.multiply(prev_viterbi, multi2)
                   viterbi.append(max(viterbi_pot))
                   back_prop1.append(np.argmax(backprop_pot))
               Viterbi_Matrix.append(viterbi)

               # print ("we are here ")
               Back_prop_matrix.append(back_prop1)
               # print(Back_prop_matrix)

            else:  ######### Unknown words
               prev_viterbi = Viterbi_Matrix[-1]  ## T-1 viterbi matrix
               prev_viterbi = np.asarray(prev_viterbi)

               viterbi = []
               back_prop1 = []

               possible_tags=POS_set
               possible_tag_score=[]
               tag_trans_vec = []
               for pt1 in possible_tags:
                   multi1 = []
                   multi2 = []
                   for ss1 in possible_tags:
                       emis_prob=1/(float(Pos_count_dict[pt1]+K))

                       multi1.append(trans_count[ss1 + " " + pt1] * emis_prob)
                       multi2.append(trans_count[ss1 + " " + pt1])
                   multi1 = np.asarray(multi1)
                   multi1 = np.asarray(multi2)
                   viterbi_pot = np.multiply(prev_viterbi, multi1)
                   backprop_pot = np.multiply(prev_viterbi, multi2)
                   viterbi.append(max(viterbi_pot))
                   back_prop1.append(np.argmax(backprop_pot))
               Viterbi_Matrix.append(viterbi)
               Back_prop_matrix.append(back_prop1)

        indices+=1
    if indices==56600:
       break

print("len of predicted value is: ", len(Predicted_val_tag))
print("len of gold label is: ", len(Gold_val_tag))

accuracy=0
unknown_word_accuracy=0
Predicted_val_tag_1=[]
for pred in Predicted_val_tag:
    Predicted_val_tag_1.append(POS_set[pred])
print (Gold_val_tag)

for ind, pred_val in enumerate(Predicted_val_tag_1):
    if pred_val==Gold_val_tag[ind]:
       accuracy+=1

    if ind in unknown_indices:
        if pred_val == Gold_val_tag[ind]:
            unknown_word_accuracy += 1

print("accuracy is: ",accuracy/float(len(Predicted_val_tag_1)))

print("accuracy for unknown word is: ", unknown_word_accuracy/len(unknown_indices))



############################################ Validation Performance complete