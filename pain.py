
# import nltk
# from nltk.corpus import wordnet as wn

from collections import OrderedDict
from sense2vec import Sense2Vec
s2v = Sense2Vec().from_disk('/home/maro/s2v_old')

# Distractors from Wordnet
def get_distractors_wordnet(syn,word):
    distractors=[]
    word= word.lower()
    orig_word = word
    if len(word.split())>0:
        word = word.replace(" ","_")
    hypernym = syn.hypernyms()
    if len(hypernym) == 0: 
        return distractors
    for item in hypernym[0].hyponyms():
        name = item.lemmas()[0].name()
        #print ("name ",name, " word",orig_word)
        if name == orig_word:
            continue
        name = name.replace("_"," ")
        name = " ".join(w.capitalize() for w in name.split())
        if name is not None and name not in distractors:
            distractors.append(name)
    return distractors


# original_word = "lion"
# synset_to_use = wn.synsets(original_word,'n')[0]
# distractors_calculated = get_distractors_wordnet(synset_to_use,original_word)

# print ("original word: ",original_word.capitalize())
# print (distractors_calculated)

# original_word = "bat"
# synset_to_use = wn.synsets(original_word,'n')[0]
# distractors_calculated = get_distractors_wordnet(synset_to_use,original_word)

# print ("\noriginal word: ",original_word.capitalize())
# print (distractors_calculated)

# original_word = "green"
# synset_to_use = wn.synsets(original_word,'n')[0]
# distractors_calculated = get_distractors_wordnet(synset_to_use,original_word)

# print ("\noriginal word: ",original_word.capitalize())
# print (distractors_calculated)


def sense2vec_get_words(word,s2v):
    output = []
    word = word.lower()
    word = word.replace(" ", "_")

    sense = s2v.get_best_sense(word)
    most_similar = s2v.most_similar(sense, n=20)

    for each_word in most_similar:
        append_word = each_word[0].split("|")[0].replace("_", " ").lower()
        if append_word.lower() != word:
            output.append(append_word.title())

    out = list(OrderedDict.fromkeys(output))
    return out


word =  "business structure"


distractors = sense2vec_get_words(word,s2v)

print ("Distractors for ",word, " : ")
print (distractors)






# @app.post("/getquestion", response_model= QuestionResponse)
# def getquestion(question: QuestionRequest):
#     context = question.context
#     question_array, gfg = get_question(context,model,tokenizer)
#     distractors = []
#     print(gfg)
#     for word in gfg:
#         argo=wn.synsets(word)
#         for syn in argo:
#              #sublist of distractors for each answer
#             print(syn)
#             distractors_sublist =  sense2vec_get_words(word,s2v)
#             print ("distractors_sublist ",distractors_sublist)
#             distractors.append(distractors_sublist)

#     print (distractors)
#     return QuestionResponse(question=question_array,answer=gfg,distractors_sublist=distractors)
