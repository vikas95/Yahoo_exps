#############

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
word_len=[]
for line1 in file1:
    line1=line1.strip()
    words1=line1.split("\t")
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
       prev_tag="start"
       POS.append("start")

Vocab=list(set(Vocab))
K=len(Vocab)
print(K)   ### We need K in smoothing part.
# print(Word_tag_posting)
print("number of distinct words in each line is ",list(set(word_len)))

Pos_count_dict=Counter(POS)
POS_set=list(set(POS))

# print (Pos_dict)

word_tag_count=Counter(unigrams)
# print (word_tag_count)

trans_count=Counter(bigrams)

for trans_key in trans_count.keys():
    prev_tag1=trans_key.split()[0]
    trans_count[trans_key]=trans_count[trans_key]/float(Pos_count_dict[prev_tag1])  ## dividing by the count of previous tag

#print(trans_count)

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
for vline in val_file:
    vline=vline.strip()
    vwords=vline.split("\t")
    if len(vwords)<2:
       prev_vtag="start"
    else:

        Gold_val_tag.append(vwords[1])

        if vwords[0] in Vocab:
           possible_tags=Word_tag_posting[vwords[0]]
           possible_tag_score=[]
           for pt1 in possible_tags:
               emis_prob = (word_tag_count[vwords[0]+" "+pt1]) #+1) / float((Pos_count_dict[pt1] + K))
               possible_tag_score.append(trans_count[prev_vtag+" "+pt1] * emis_prob)

           possible_tag_score=np.asarray(possible_tag_score)
           Predicted_val_tag.append(possible_tags[np.argmax(possible_tag_score)])
           prev_vtag=possible_tags[np.argmax(possible_tag_score)]

        else:  ######### Unknown words
           possible_tags=POS_set
           possible_tag_score=[]
           for pos_lab in possible_tags:
               emis_prob=1/(float(Pos_count_dict[pos_lab]+K))

               trans_prob=trans_count[prev_vtag+" "+pos_lab]
               possible_tag_score.append(emis_prob*trans_prob)
           if len(unknown_indices)<10:
              #print(possible_tag_score)
              pass
           possible_tag_score=np.asarray(possible_tag_score)
           Predicted_val_tag.append(possible_tags[np.argmax(possible_tag_score)])
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