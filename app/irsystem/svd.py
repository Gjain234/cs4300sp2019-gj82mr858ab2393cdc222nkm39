from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse.linalg import svds
from sklearn.preprocessing import normalize
import json
import numpy as np
import ast

#opens json file and creates a list of tuples with name of video and description called documents\n",
#data represents a list of dictionarys for each video with all of the info from ted_main
documents=[]
with open("ted_main.json", encoding="utf8") as f:
    data=json.load(f)
    for x in data:
        documents.append((x["name"], x["description"]))


#given the data file and the title of the video, outputs the index that video is located at
def findindex(data,url):
    print(len(data))
    for x in range(len(data)):
        if data[x]['name']== url:
            return x


#creates tf-idf matrix, can alter max_df and min_df
vectorizer = TfidfVectorizer(stop_words = 'english', max_df = .8,
                            min_df = 40)
my_matrix = vectorizer.fit_transform([x[1] for x in documents]).transpose()


#runs svd, can alter k but values mostly live in the space under 30
words_compressed, _, docs_compressed = svds(my_matrix, k=30) #number of dimensions the data lives in
docs_compressed = docs_compressed.transpose()
#print(docs_compressed.shape)


#creates clusters of 15 vidoes based on the index of the input video
docs_compressed = normalize(docs_compressed, axis = 1)
#print(docs_compressed.shape)
def closest_projects(project_index_in, docs_compressed, k = 15):
    y = docs_compressed[project_index_in,:].transpose()
    y = np.asmatrix(y)
    y_new = docs_compressed[0]
    sims = np.dot(docs_compressed, y_new)
    asort = np.argsort(-sims)[:k+1]
    return [(documents[i][0],sims[i]/sims[asort[0]]) for i in asort[1:]]

def moodvid(lst,mood,data):
    mx=0
    vid=0;
    for x in range(len(lst)):
        name=lst[x][0]
        index=findindex(data,name)
        ratings=ast.literal_eval(data[index]['ratings'])
        for el in range(len(ratings)):
            if ratings[el]['name']==mood:
                count= ratings[el]['count']
                if count > mx:
                    mx=count
                    vid=index
    return vid

#function that takes in the closest_projects function and the index of the desirable video
#outputs the url of the second video in the cluster
def extract_cluster_ratings(data, index, mood):
    lst=closest_projects(index, docs_compressed)
    #name=lst[1][0]
    #i=findindex(data,name)
    i = moodvid(lst, mood, data)
    return data[i]['url']
