import re
import numpy as np
import pandas as pd
import pickle
import json
import collections
import math

transcripts = pd.read_csv('transcripts.csv')
talk_information = pd.read_csv('ted_main.csv')

def tokenize(text):
    """Returns a list of words that make up the text.
    
    Note: for simplicity, lowercase everything.
    Requirement: Use Regex to satisfy this function
    
    Params: {text: String}
    Returns: List
    """
    words = re.findall(r"[A-Za-z]+'[a-z]+[[:>:]]|[A-Za-z]+", text.lower())
    return words

def tokenize_transcript(tokenize_method,input_transcript):
    """Returns a list of words contained in an entire transcript.
    Params: {tokenize_method: Function (a -> b),
             input_transcript: Tuple}
    Returns: List
    """
    final_lst = []
    for i in (range(0,len(input_transcript))):
        #print(tokenize_method(input_transcript[i]))
        final_lst = final_lst + list(set(tokenize_method(input_transcript[i])))
    return final_lst

all_words_total = tokenize_transcript(tokenize,talk_information['description'])

description_word_dict = (collections.Counter(all_words_total))
good_types_descriptions = {k:v for (k,v) in description_word_dict.items()}

all_words_total_transcripts = tokenize_transcript(tokenize, transcripts['transcript'])
transcript_word_dict = (collections.Counter(all_words_total_transcripts))
good_types_transcripts = {k:v for (k,v) in transcript_word_dict.items()}

def compute_idf(doc_freq, n_docs, min_df=1, max_df_ratio=0.85):
    """Returns a dictionary of IDFs for each word
    Params: {doc_freq: Dictionary,
             n_docs: Int}
    Returns: Dictionary
    """
    q = {}
    temp = 0
    for term in doc_freq.keys():
        temp = doc_freq[term]
        if temp >= min_df and temp <= n_docs * max_df_ratio:
            q[term] = math.log(n_docs/(1+temp),2)
    return q

description_low_idf = compute_idf(good_types_descriptions,len(good_types_descriptions.keys()),1,0.05)
transcript_low_idf = compute_idf(good_types_transcripts,len(good_types_transcripts.keys()),1,0.05)

def compute_inv(tokenize_method,input_transcript,t_idf):
    q = {}
    for i in (range(0,len(input_transcript))):
        final_lst = tokenize_method(input_transcript[i])
        df_temp = (collections.Counter(final_lst))
        trans_df = {k:v for (k,v) in df_temp.items()}
        temp = {}
        for term in trans_df.keys():
            if term in t_idf.keys():
                if temp.get(term) == None:
                    temp[term] = 1
                else:
                    temp[term] += 1
        for k in temp.keys():
            if q.get(k) == None:
                q[k] = [(i,temp[k])]
            else:
                q[k].append((i,temp[k]))
    return q


description_low_inv = compute_inv(tokenize,talk_information['description'],description_low_idf)
transcript_low_inv = compute_inv(tokenize,transcripts['transcript'],transcript_low_idf)

def compute_doc_norms(index, idf, n_docs):
    d = {}
    for k in index.keys():
        for t in index[k]:
            if idf.get(k) != None:
                if d.get(t[0]) == None:
                    d[t[0]] = (t[1] * idf[k])**2
                else:
                    d[t[0]] += (t[1] * idf[k])**2
    for doc in d.keys():
        d[doc] = math.sqrt(d[doc])
    return d

description_low_norms = compute_doc_norms(description_low_inv, description_low_idf, len(description_low_inv))
transcript_low_norms = compute_doc_norms(transcript_low_inv, transcript_low_idf, len(transcript_low_inv))

f = open("description_low_inv.pkl","wb")
pickle.dump(description_low_inv,f)
f.close()

f = open("description_low_idf.pkl","wb")
pickle.dump(description_low_idf,f)
f.close()

f = open("description_low_norms.pkl","wb")
pickle.dump(description_low_norms,f)
f.close()

f = open("transcript_low_inv.pkl","wb")
pickle.dump(transcript_low_inv,f)
f.close()

f = open("transcript_low_idf.pkl","wb")
pickle.dump(transcript_low_idf,f)
f.close()

f = open("transcript_low_norms.pkl","wb")
pickle.dump(transcript_low_norms,f)
f.close()
    
