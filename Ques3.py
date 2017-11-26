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

val_file= open("PTBSmall/test.tagged","r")
prev_vtag="start"
Gold_val_tag=[]
Predicted_val_tag=[]
unknown_indices=[]
indices=0
Viterbi_Matrix=[]  ######## to store Viterbi array for each word
Back_pointer=[]   ####### to keep track to transition.
for vline in val_file:
    vline=vline.strip()
    vwords=vline.split()
    if len(vwords)<2: ## new line, hence we should consider end tag here.
        possible_tags = POS_set
        tag_trans_vec = []
        possible_tag_score = []
        for pt1 in possible_tags:
            possible_tag_score.append(trans_count[prev_vtag + " " + pt1] * emis_prob)
            tag_trans_vec.append(trans_count[pt1 + " " + end_tag])

        possible_tag_score = np.asarray(possible_tag_score)
        possible_tag_score = np.reshape(possible_tag_score, (1, len(POS_set)))
        prev_viterbi = np.asarray(prev_viterbi)
        prev_viterbi = np.reshape(prev_viterbi, (len(POS_set), 1))
        tag_trans_vec = np.asarray(tag_trans_vec)
        tag_trans_vec = np.reshape(tag_trans_vec, (1, len(POS_set)))

        Score_mat = np.matmul(prev_viterbi, possible_tag_score)
        New_viterbi_indices = np.argmax(Score_mat, axis=1)
        New_viterbi = []
        for ind in New_viterbi_indices:
            New_viterbi.append(POS_set[ind])
        if len(New_viterbi) != len(prev_viterbi):
            print ("something is wrong, verify in previous steps")
        Viterbi_Matrix.append(New_viterbi)

        Back_pointer_score = np.matmul(prev_viterbi, tag_trans_vec)
        Back_pointer_ind = np.amax(Back_pointer_score, axis=1)  ## % (len(POS_set))
        Back_pointer.append(Back_pointer_ind)

       prev_vtag="start"
    else:
        Gold_val_tag.append(vwords[1])
        if prev_vtag=="start":  ## this means we are on the first word of the sentence
           Viterbi=[]
           Viterbi_Matrix = []
           Predicted_val_tag+=Back_pointer
           Back_pointer = []
           possible_tags=POS_set
           for pt1 in possible_tags:
               emis_prob = (word_tag_count[vwords[0] + " " + pt1])  # smoothing of word tag count is done in the last segment.
               if emis_prob == 0:  ## when the word is never seen with "pt1" tag
                   emis_prob = 1 / (float(Pos_count_dict[pt1] + K))

               Viterbi.append(trans_count[prev_vtag + " " + pt1] * emis_prob)
           Back_pointer.append("start") ## instead of adding 0 which is the index, I am adding tag
           Viterbi_Matrix.append(Viterbi)
        else:
            prev_viterbi=Viterbi_Matrix[-1] ## T-1 viterbi matrix
            if vwords[0] in Vocab:
               # possible_tags=Word_tag_posting[vwords[0]]  ### now if we are smoothing, even if word is known, we still need to consider all possibilities for tags
               possible_tags = POS_set
               tag_trans_vec=[]
               possible_tag_score = []
               for pt1 in possible_tags:
                   emis_prob = (word_tag_count[vwords[0]+" "+pt1]) #smoothing of word tag count is done in the last segment.
                   if emis_prob==0:  ## when the word is never seen with "pt1" tag
                      emis_prob=1/(float(Pos_count_dict[pt1]+K))
                   possible_tag_score.append(trans_count[prev_vtag+" "+pt1] * emis_prob)
                   tag_trans_vec.append(trans_count[prev_vtag+" "+pt1])

               possible_tag_score=np.asarray(possible_tag_score)
               possible_tag_score=np.reshape(possible_tag_score,(1,len(POS_set)))
               prev_viterbi=np.asarray(prev_viterbi)
               prev_viterbi=np.reshape(prev_viterbi,(len(POS_set),1))
               tag_trans_vec=np.asarray(tag_trans_vec)
               tag_trans_vec=np.reshape(tag_trans_vec,(1,len(POS_set)))

               Score_mat=np.matmul(prev_viterbi,possible_tag_score)
               New_viterbi_indices=np.argmax(Score_mat,axis=1)
               New_viterbi=[]
               for ind in New_viterbi_indices:
                   New_viterbi.append(POS_set[ind])
               if len(New_viterbi)!= len(prev_viterbi):
                  print ("something is wrong, verify in previous steps")
               Viterbi_Matrix.append(New_viterbi)

               Back_pointer_score=np.matmul(prev_viterbi, tag_trans_vec)
               Back_pointer_ind = np.amax(Back_pointer_score, axis=1)  ## % (len(POS_set))
               Back_pointer.append(Back_pointer_ind)


               # Predicted_val_tag.append(possible_tags[np.argmax(possible_tag_score)])
               prev_vtag=possible_tags[np.argmax(possible_tag_score)]

            else:  ######### Unknown words
               possible_tags=POS_set
               possible_tag_score=[]
               tag_trans_vec = []
               for pos_lab in possible_tags:
                   emis_prob=1/(float(Pos_count_dict[pos_lab]+K))

                   trans_prob=trans_count[prev_vtag+" "+pos_lab]
                   possible_tag_score.append(emis_prob*trans_prob)
                   tag_trans_vec.append(trans_prob)

               possible_tag_score = np.asarray(possible_tag_score)
               possible_tag_score = np.reshape(possible_tag_score, (1, len(POS_set)))
               prev_viterbi = np.asarray(prev_viterbi)
               prev_viterbi = np.reshape(prev_viterbi, (len(POS_set), 1))
               tag_trans_vec = np.asarray(tag_trans_vec)
               tag_trans_vec = np.reshape(tag_trans_vec, (1, len(POS_set)))

               Score_mat = np.matmul(prev_viterbi, possible_tag_score)
               New_viterbi_indices = np.argmax(Score_mat, axis=1)
               New_viterbi = []
               for ind in New_viterbi_indices:
                   New_viterbi.append(POS_set[ind])
               if len(New_viterbi) != len(prev_viterbi):
                   print ("something is wrong, verify in previous steps")
               Viterbi_Matrix.append(New_viterbi)

               Back_pointer_score = np.matmul(prev_viterbi, tag_trans_vec)
               Back_pointer_ind = np.amax(Back_pointer_score,axis=1) ## % (len(POS_set))
               Back_pointer.append(Back_pointer_ind)

               # Predicted_val_tag.append(possible_tags[np.argmax(possible_tag_score)])
               prev_vtag = possible_tags[np.argmax(possible_tag_score)]
               unknown_indices.append(indices)  ###### if the word is not in the Vocab of training data, then it is a unknown word.

        indices+=1


print("len of predicted value is: ", len(Predicted_val_tag))
print("len of gold label is: ", len(Gold_val_tag))

accuracy=0
unknown_word_accuracy=0
for ind, pred_val in enumerate(Predicted_val_tag):
    if pred_val==Gold_val_tag[ind]:
       accuracy+=1

    if ind in unknown_indices:
        if pred_val == Gold_val_tag[ind]:
            unknown_word_accuracy += 1

print("accuracy is: ",accuracy/ind)

print("accuracy for unknown word is: ", unknown_word_accuracy/len(unknown_indices))



############################################ Validation Performance complete