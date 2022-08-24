from typing import List
from fastT5 import get_onnx_model,get_onnx_runtime_sessions,OnnxT5
from transformers import AutoTokenizer
from pathlib import Path
import os
from fastapi import FastAPI
from pydantic import BaseModel
from textblob import TextBlob

import nltk
from nltk.corpus import wordnet as wn

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


@app.get('/')
def index():
    return {'message':'hello world'}


@app.post("/getquestion", response_model= QuestionResponse)
def getquestion(question: QuestionRequest):
    context = question.context
    question_array, gfg = get_question(context,model,tokenizer)
    distractors = []
    for word in gfg:
        argo=wn.synsets(word)
        for syn in argo:
             #sublist of distractors for each answer
            distractors_sublist = get_distractors_wordnet(syn,word) 
            distractors.append(distractors_sublist)

    print (distractors)
    return QuestionResponse(question=question_array,answer=gfg,distractors_sublist=distractors)


   
    return QuestionResponse(question=question_array,answer=gfg,distractors_sublist = distractors)
