import re
import numpy as np
import pandas as pd
import pickle
import json
import collections
import math

transcripts = pd.read_csv('transcripts.csv')
talk_information = pd.read_csv('ted_main.csv')
comments = pickle.load(open("comments_plain.pkl", "rb"))

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

description_idf = pickle.load(open("description_idf.pkl", "rb"))
transcript_idf = pickle.load(open("transcript_idf.pkl", "rb"))

description_inv = pickle.load(open("description_inv.pkl", "rb"))
transcript_inv = pickle.load(open("transcript_inv.pkl", "rb"))

description_norms = pickle.load(open("description_norms.pkl", "rb"))
transcript_norms = pickle.load(open("transcript_norms.pkl", "rb"))

comment_idf = pickle.load(open("comment_idf.pkl", "rb"))

comment_inv = pickle.load(open("comment_inv.pkl", "rb"))

comment_norms = pickle.load(open("comment_norms.pkl", "rb"))

description_low_idf = pickle.load(open("description_low_idf.pkl", "rb"))
transcript_low_idf = pickle.load(open("transcript_low_idf.pkl", "rb"))

description_low_inv = pickle.load(open("description_low_inv.pkl", "rb"))
transcript_low_inv = pickle.load(open("transcript_low_inv.pkl", "rb"))

description_low_norms = pickle.load(open("description_low_norms.pkl", "rb"))
transcript_low_norms = pickle.load(open("transcript_low_norms.pkl", "rb"))

availComms = pickle.load(open("availComms.pkl", "rb"))
availTalks = pickle.load(open("availTalks.pkl", "rb"))

proc_tags = pickle.load(open("proc_tags.pkl", "rb"))

def index_search(query, index, idf, doc_norms, tokenize_method):
    _id = 0
    ret = []
    _id_ref = {}
    temp = list(doc_norms.keys())
    while _id < len(temp):
        _id_ref[temp[_id]] = _id
        ret.append((0,temp[_id]))
        _id += 1
    rem = {'i','want','to','learn','about','and','how','the','a'}
    temp2 = tokenize_method(query.lower())
    q = []
    for word in temp2:
        if word not in rem:
            q.append(word)

    q_comp = {}
    for w in q:
        if q_comp.get(w) == None:
            q_comp[w] = 1
        else:
            q_comp[w] += 1
    q_norm = 0
    for k in q_comp.keys():
        if idf.get(k) != None:
            q_norm += (q_comp[k] * idf[k])**2
    q_norm = math.sqrt(q_norm)

    for w in q:
        if idf.get(w) != None and index.get(w) != None:
            for ent in index[w]:
                ret[_id_ref[ent[0]]] = (ret[_id_ref[ent[0]]][0] + q_comp.get(w) * idf.get(w) * ent[1] * idf.get(w), ret[_id_ref[ent[0]]][1])
    _id = 0
    while _id < len(temp):
        if q_norm * doc_norms[temp[_id]] != 0:
            ret[_id] = (ret[_id][0] / (q_norm * doc_norms[temp[_id]]), ret[_id][1])
        _id += 1

    ret = sorted(ret,reverse=True)
    return ret

def get_prompt1_video_link(query):
    #print("Search: "+ query)
    r = index_search(query, description_inv, description_idf, description_norms,tokenize)
    ret = []
    videos = []
    for score, msg_id in r[:10]:
        #ret.append([score, talk_information['title'][msg_id], talk_information['description'][msg_id]])
        dataset_talk = talk_information['url'][msg_id]
        talk_segment = dataset_talk[26:]
        temp = "https://embed.ted.com/talks/" + talk_segment
        videos.append(temp)
        #temp = (talk_information['url'][msg_id]) + "?utm_campaign=tedspread&utm_medium=referral&utm_source=tedcomshare"
    video_link = temp
    #print(video_link)
    return videos

def get_prompt2_video_link(query):
    #print("Search: "+ query)
    r = index_search(query, transcript_inv, transcript_idf, transcript_norms,tokenize)
    ret = []
    videos = []
    for score, msg_id in r[:1]: #previously: [:10]
        #ret.append([score, talk_information['title'][msg_id], talk_information['description'][msg_id]])
        dataset_talk = talk_information['url'][msg_id]
        talk_segment = dataset_talk[26:]
        temp = "https://embed.ted.com/talks/" + talk_segment
        videos.append(temp)
        #temp = (talk_information['url'][msg_id]) + "?utm_campaign=tedspread&utm_medium=referral&utm_source=tedcomshare"
    video_link = temp
    #print(video_link)
    return videos

def combined_search(query):
    expan = index_search(query, description_low_inv, description_low_idf, description_low_norms,tokenize)
    for res in expan[:5]:
        query += " "
        #print(talk_information['tags'][res[1]])
        q = talk_information['tags'][res[1]].strip(']').strip('[').strip(',').strip('\'').split('\', \'')
        for tags in q:
            query += tags + " "
    #print(query)
    t = index_search(query, transcript_inv, transcript_idf, transcript_norms,tokenize)
    d = index_search(query, description_inv, description_idf, description_norms,tokenize)
    i_t = {i:s for (s,i) in t}
    i_d = {i:s for (s,i) in d}
    ret = []
    for i in range(0,len(i_d.keys())):
        if i_d.get(i) != None and availTalks.get(i) != None and i_t.get(availTalks[i]) != None:
            ret.append((0.1 * i_t.get(availTalks[i]) + 0.9 * i_d.get(i), i))
        elif i_d.get(i) != None:
            ret.append((0.9 * i_d.get(i), i))
    ret = sorted(ret,reverse=True)
    r = []
    for score, msg_id in ret[:10]:
        dataset_talk = talk_information['url'][msg_id]
        talk_segment = dataset_talk[26:]
        temp = "https://embed.ted.com/talks/" + talk_segment
        r.append([temp, talk_information['title'][msg_id], talk_information['description'][msg_id]]) #fixme
    #print(r)
    #print(i_t)
    return r

def comment_search(query,topic):
    query += ' '
    for q in proc_tags[topic]:
        query += q + ' '
    r = index_search(query, comment_inv, comment_idf, comment_norms,tokenize)
    ret = []
    #print(r)
    for score, msg_id in r[:10]:
        if availComms.get(msg_id) != None:
            dataset_talk = talk_information['url'][msg_id]
            talk_segment = dataset_talk[26:]
            temp = "https://embed.ted.com/talks/" + talk_segment
            ret.append([temp, talk_information['title'][msg_id], talk_information['description'][msg_id]]) #fixme
    #print(ret)
    return ret
