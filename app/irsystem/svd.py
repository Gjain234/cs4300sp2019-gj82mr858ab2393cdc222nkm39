from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse.linalg import svds
from sklearn.preprocessing import normalize
import json
import numpy as np

#opens json file and creates a list of tuples with name of video and description called documents\n",
#data represents a list of dictionarys for each video with all of the info from ted_main
documents=[]
with open("ted_main.json", encoding="utf8") as f:
    data=json.load(f)
    for x in data:
        documents.append((x["name"], x["description"]))


#given the data file and the title of the video, outputs the index that video is located at
def findindex(data,url):
    for x in range(len(data)):
        if data[x]['url']== url:
            return x


#creates tf-idf matrix, can alter max_df and min_df
vectorizer = TfidfVectorizer(stop_words = 'english', max_df = .8,
                            min_df = 40)
my_matrix = vectorizer.fit_transform([x[1] for x in documents]).transpose()


#runs svd, can alter k but values mostly live in the space under 30
words_compressed, _, docs_compressed = svds(my_matrix, k=30)
docs_compressed = docs_compressed.transpose()


#creates clusters of 15 vidoes based on the index of the input video
docs_compressed = normalize(docs_compressed, axis = 1)
def closest_projects(project_index_in, k = 15):
    sims = docs_compressed.dot(docs_compressed[project_index_in,:])
    asort = np.argsort(-sims)[:k+1]
    return [(documents[i][0],sims[i]/sims[asort[0]]) for i in asort[1:]]


#function that takes in the closest_projects function and the index of the desirable video
#outputs the url of the second video in the cluster
def extract_cluster_ratings(index):
    lst=closest_projects(index)
    name=lst[1][0]
    index=findindex(data,name)
    return data[index]["url"]


