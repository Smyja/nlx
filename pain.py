import nltk
from nltk.corpus import wordnet as wn

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


original_word = "lion"
synset_to_use = wn.synsets(original_word,'n')[0]
distractors_calculated = get_distractors_wordnet(synset_to_use,original_word)

print ("original word: ",original_word.capitalize())
print (distractors_calculated)

original_word = "bat"
synset_to_use = wn.synsets(original_word,'n')[0]
distractors_calculated = get_distractors_wordnet(synset_to_use,original_word)

print ("\noriginal word: ",original_word.capitalize())
print (distractors_calculated)

original_word = "green"
synset_to_use = wn.synsets(original_word,'n')[0]
distractors_calculated = get_distractors_wordnet(synset_to_use,original_word)

print ("\noriginal word: ",original_word.capitalize())
print (distractors_calculated)
