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


#given the data file and the url of the video, outputs the index that video is located at
def findindex_url(data,url):
    print(len(data))
    print("HERE IS THE URL: ")
    print(url)
    #u = url[0][0:(len(url[0]))]
    #u = u.replace("embed", "www")
    url[0] = url[0].replace("embed","www")
    #print(u)
    for x in range(len(data)):
        #print(data[x]['url'])
        if data[x]['url']== url[0]: #this was a list of urls! nothing will correspond to the list. took the first one.
            print("SUCCESS")
            return x

def findindex_name(data,name):
    for x in range(len(data)):
        #print(data[x]['url'])
        if data[x]['name']== name: #this was a list of urls! nothing will correspond to the list. took the first one.
            print("SUCCESS2")
            return x
#creates tf-idf matrix, can alter max_df and min_df
vectorizer = TfidfVectorizer(stop_words = 'english', max_df = .75,
                            min_df = 50)
my_matrix = vectorizer.fit_transform([x[1] for x in documents]).transpose()


#runs svd, can alter k but values mostly live in the space under 30
words_compressed, _, docs_compressed = svds(my_matrix, k=40) #number of dimensions the data lives in
docs_compressed = docs_compressed.transpose()
#print(docs_compressed.shape)


#creates clusters of 15 vidoes based on the index of the input video
docs_compressed = normalize(docs_compressed, axis = 1)
#print(docs_compressed.shape)
def closest_projects(project_index_in, docs_compressed, k = 15):
    y = docs_compressed[project_index_in,:].transpose()
    y = np.asmatrix(y)
    print("docs_compressed.shape")
    print(docs_compressed.shape)
    print("docs_compressed[project_index_in,:].shape: ")
    print(docs_compressed[0][0].shape)
    print(docs_compressed[0][0:40])
    print("project_index_in value: ")
    print(project_index_in)
    slice = np.array(list(docs_compressed[project_index_in][0:40]))
    print("slice length")
    print(slice.shape)
    print("docs_compressed[project_index_in].shape: ")
    print(docs_compressed[project_index_in].shape)
    print("experimenting with reshape: ")
    #print(docs_compressed[project_index_in,:].reshape(30,)) WONT WORK 76500 -> 30
    #docs_compressed[project_index_in].reshape()
    #y_new = docs_compressed[0]
    sims = np.dot(docs_compressed, slice) #y_new
    print("sims.shape: ")
    print(sims.shape)
    asort = np.argsort(-sims)[:k+1]
    return [(documents[i][0],sims[i]/sims[asort[0]]) for i in asort[1:]]

def moodvid(lst,mood,data): #returning 1 video
    vid_mood_lst = []
    for x in range(len(lst)):
        name=lst[x][0]
        index=findindex_name(data,name)
        ratings=ast.literal_eval(data[index]['ratings'])
        for el in range(len(ratings)):
            if ratings[el]['name']==mood:
                count= ratings[el]['count']
                vid_mood_lst.append((index,count))
                #print("ACCUMULATING")

                #print(vid_mood_lst)
                # if count > mx:
                #     mx=count
                #     vid=index
        sorted_vid_mood_lst = sorted(vid_mood_lst, key=lambda tup: tup[1])
        print("FINAL VID MOOD LIST: ")
        print(sorted_vid_mood_lst)
    return sorted_vid_mood_lst

#function that takes in the closest_projects function and the index of the desirable video
#outputs the url of the second video in the cluster
def extract_cluster_ratings(data, index, mood):
    lst=closest_projects(index, docs_compressed)
    #name=lst[1][0]
    #i=findindex(data,name)
    i = moodvid(lst, mood, data)
    return data[i]['url']

#returns top 10 sorted results using SVD not including top video
def top_svd(data, index, mood):
    print("INDEX READING FROM TOP_SVD: ")
    print(index)
    lst=closest_projects(index, docs_compressed)
    #i = moodvid(lst, mood, data)
    ranked_vids = moodvid(lst,mood,data)
    print("ranked_vids size")
    print(len(ranked_vids))
    top_vids = []
    for i in range(10):
        ind = ranked_vids[i][0]
        temp_url = data[ind]['url']
        embed_url = "https://embed.ted.com/talks/" + temp_url[26:]
        top_vids.append([embed_url, data[ind]['title'], data[ind]['description']]) #fix me
    return top_vids
