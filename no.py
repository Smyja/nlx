
from numpy import array
from textblob import TextBlob
def get_question(sentence):
    gfg = TextBlob(sentence)
    gfg = gfg.noun_phrases
    print(len(gfg))
    array=[]
    for i in gfg:
        text = "context: {} answer: {}".format(sentence,i)
        array.append(text)
       
    return print(len(array))
get_question(sentence="Many intellectuals, labour unions, artists, and political parties worldwide have been influenced by Marx's work, with many modifying or adapting his ideas. Marx is typically cited as one of the principal architects of modern social science." )
