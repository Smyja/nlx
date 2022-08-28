from typing import List
from fastT5 import get_onnx_model,get_onnx_runtime_sessions,OnnxT5
from transformers import AutoTokenizer
from pathlib import Path
import os
from fastapi import FastAPI
import nltk
from nltk.corpus import wordnet as wn
from pydantic import BaseModel
from textblob import TextBlob
from collections import OrderedDict
from sense2vec import Sense2Vec
s2v = Sense2Vec().from_disk('/home/maro/s2v_old')



app = FastAPI()

class QuestionRequest(BaseModel):
    context: str
    

class QuestionResponse(BaseModel):
    question: List[str] = []
    answer: List[str] = []
    distractors_sublist: List[List[str]] = [ [] ]



trained_model_path = './t5_squad_v1/'

pretrained_model_name = Path(trained_model_path).stem


encoder_path = os.path.join(trained_model_path,f"{pretrained_model_name}-encoder-quantized.onnx")
decoder_path = os.path.join(trained_model_path,f"{pretrained_model_name}-decoder-quantized.onnx")
init_decoder_path = os.path.join(trained_model_path,f"{pretrained_model_name}-init-decoder-quantized.onnx")

model_paths = encoder_path, decoder_path, init_decoder_path
model_sessions = get_onnx_runtime_sessions(model_paths)
model = OnnxT5(trained_model_path, model_sessions)

tokenizer = AutoTokenizer.from_pretrained(trained_model_path)


def get_question(sentence,mdl,tknizer):
    gfg = TextBlob(sentence)
    gfg = gfg.noun_phrases
    array=[]
    for i in gfg:
        text = "context: {} answer: {}".format(sentence,i)
        array.append(text)
        
    max_len = 256
    question_array =[]
    for text in array:
        encoding = tknizer.encode_plus(text,max_length=max_len, pad_to_max_length=False,truncation=True, return_tensors="pt")
        input_ids, attention_mask = encoding["input_ids"], encoding["attention_mask"]
        outs = mdl.generate(input_ids=input_ids,
                                        attention_mask=attention_mask,
                                        early_stopping=True,
                                        num_beams=5,
                                        num_return_sequences=1,
                                        no_repeat_ngram_size=2,
                                        max_length=128)
        dec = [tknizer.decode(ids,skip_special_tokens=True) for ids in outs]
        Question = dec[0].replace("question:","")
        Question= Question.strip()
        question_array.append(Question)
        print (question_array)
    return question_array, gfg

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


def sense2vec_get_words(word,s2v):
    output = []
    print("word---",word)
    word = word.lower()
    word = word.replace(" ", "_")

    sense = s2v.get_best_sense(word)
    most_similar = s2v.most_similar(sense, n=20)

    # print ("most_similar ",most_similar)

    for each_word in most_similar:
        append_word = each_word[0].split("|")[0].replace("_", " ").lower()
        if append_word.lower() != word:
            output.append(append_word.title())

    out = list(OrderedDict.fromkeys(output))
    return out



@app.get('/')
def index():
    return {'message':'hello world'}


@app.post("/getquestion", response_model= QuestionResponse)
def getquestion(question: QuestionRequest):
    context = question.context
    question_array, gfg = get_question(context,model,tokenizer)
    distractors = []
    print(gfg)
    for word in gfg:
        distractors_sublist =  sense2vec_get_words(word,s2v)
        print ("distractors_sublist ",distractors_sublist)
        distractors.append(distractors_sublist)

    print (distractors)
    return QuestionResponse(question=question_array,answer=gfg,distractors_sublist=distractors)
