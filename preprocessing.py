import re
import numpy as np
import pandas as pd
import pickle
import json
import collections
import math

#########################################################
#                     LOAD DOCS
#########################################################
transcripts = pd.read_csv('transcripts.csv')
talk_information = pd.read_csv('ted_main.csv')
comments = pickle.load(open("comments_plain.pkl", "rb"))


#########################################################
#              PREPROCESSING FUNCTIONS
#########################################################

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

def availableTalks(talks_info,trans):
    ret = {}
    for url in list(trans['url']):
        #print(url)
        if url in list(talks_info['url']):
            ret[list(talks_info['url']).index(url)] = list(trans['url']).index(url)
    return ret

def availableComms(talks_info,comms):
    ret = {}
    for title in comms.keys():
        if title in list(talks_info['title']):
            ret[list(talks_info['title']).index(title)] = comms[title]
    return ret

def tokenize_comms(tokenize_method,input_transcript):
    """Returns a list of words contained in an entire transcript.
    Params: {tokenize_method: Function (a -> b),
             input_transcript: Tuple}
    Returns: List
    """
    final_lst = []
    for i in input_transcript.keys():
        final_lst = final_lst + list(set(tokenize_method(input_transcript[i])))
    return final_lst

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

def compute_comm(tokenize_method,input_transcript,t_idf):
    q = {}
    for i in input_transcript.keys():
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

def process_tags(talks):
    d = {'health':{}, 'sports':{}, 'education':{}, 'technology':{}, 'entertainment':{}, 'politics':{}, 'culture':{},'religion':{},'religion and culture':{}}
    for tags in talks['tags']:
        q = tags.strip(']').strip('[').strip(',').strip('\'').split('\', \'')
        for t in q:
            if t in d.keys():
                for n in q:
                    if d[t].get(n) == None:
                        d[t][n] = 1
                    else:
                        d[t][n] = d[t][n] + 1
    d['religion and culture'] = d['culture']
    for k in d['religion'].keys():
        if k in d['religion and culture'].keys():
            d['religion and culture'][k] = d['religion and culture'][k] + d['religion'][k]
        else:
            d['religion and culture'][k] = d['religion'][k]

    r = {'health':['disease','disability','medical','drug','treatment','health','care','depression','mental','sick','die','human'], 'sports':['ball','football','endurance','shot','teamwork','coach','fast','athletic','body','fitness','nutrition','fit','technique','record','physical','perserverance'], 'education':['teacher','professor','lesson','creativity','education','children','teach','public','school','university','curriculum','innovation','math','mind'], 'technology':['research','innovation','phone','tablet','automobile','car','online','web','computer','robot','engineer','prosthetic','internet','google','facebook','laptop'], 'entertainment':['music','scene','mystery','movie','fiction','netflix','classic','instrument','beat','rhythm','band','listen','watch','perform','creativity','art'], 'politics':['advocacy','global','environment','government','sustainability','party','public','speaking','leader','equality','democracy','law','world','people','candidate','money','debate','united','states','volunteer','campaign','news','media'], 'religion and culture':['religion','culture','world','diverse','god','heritage','ancestor','story','country','travel','tradition','art','nation','spirit','faith','food','song','family','roots','difference']}
    """
    for k in d['health'].keys():
        if d['health'][k] > 6:
            r['health'].append(k)
    for k in d['sports'].keys():
        if d['sports'][k] > 6:
            r['sports'].append(k)
    for k in d['education'].keys():
        if d['education'][k] > 6:
            r['education'].append(k)
    for k in d[topic].keys():
        if d[topic][k] > 6:
            r[topic].append(k)
    for k in d[topic].keys():
        if d[topic][k] > 6:
            r[topic].append(k)
    for k in d[topic].keys():
        if d[topic][k] > 6:
            r[topic].append(k)
    """    
    for k in r.keys():
        print(k)
        print(r[k])
    return r
    
#########################################################
#                    WRITE VARIABLES
#########################################################

#process_tags(talk_information)

proc_tags = process_tags(talk_information)

availTalks = availableTalks(talk_information,transcripts)

availComms = availableComms(talk_information,comments)

all_words_total_comms = tokenize_comms(tokenize, comments)
comment_word_dict = (collections.Counter(all_words_total_comms))
good_types_comms = {k:v for (k,v) in comment_word_dict.items()}

#all_words_total = tokenize_transcript(tokenize,talk_information['description'])
#print (all_words_total)
#description_word_dict = (collections.Counter(all_words_total))
#good_types_descriptions = {k:v for (k,v) in description_word_dict.items()}

#all_words_total_transcripts = tokenize_transcript(tokenize, transcripts['transcript'])
#transcript_word_dict = (collections.Counter(all_words_total_transcripts))
#good_types_transcripts = {k:v for (k,v) in transcript_word_dict.items()}

#description_low_idf = compute_idf(good_types_descriptions,len(good_types_descriptions.keys()),1,0.05)
#transcript_low_idf = compute_idf(good_types_transcripts,len(good_types_transcripts.keys()),1,0.05)
comment_idf = compute_idf(good_types_comms,len(good_types_comms.keys()),1,0.85)

#description_low_inv = compute_inv(tokenize,talk_information['description'],description_low_idf)
#transcript_low_inv = compute_inv(tokenize,transcripts['transcript'],transcript_low_idf)
comment_inv = compute_comm(tokenize,availComms,comment_idf)

#description_low_norms = compute_doc_norms(description_low_inv, description_low_idf, len(description_low_inv))
#transcript_low_norms = compute_doc_norms(transcript_low_inv, transcript_low_idf, len(transcript_low_inv))
comment_norms = compute_doc_norms(comment_inv, comment_idf, len(comment_idf))


#########################################################
#                       WRITE DOCS
#########################################################

f = open("proc_tags.pkl","wb")
pickle.dump(proc_tags,f)
f.close()


f = open("comment_inv.pkl","wb")
pickle.dump(comment_inv,f)
f.close()

f = open("comment_idf.pkl","wb")
pickle.dump(comment_idf,f)
f.close()

f = open("comment_norms.pkl","wb")
pickle.dump(comment_norms,f)
f.close()

f = open("availComms.pkl","wb")
pickle.dump(availComms,f)
f.close()


f = open("availTalks.pkl","wb")
pickle.dump(availTalks,f)
f.close()
"""

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
"""
