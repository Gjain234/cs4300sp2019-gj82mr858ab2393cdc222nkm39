from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse.linalg import svds
from sklearn.preprocessing import normalize
import json

documents=[]
with open("ted_main.json", encoding="utf8") as f:
    data=json.load(f)
    for x in data:
        documents.append((x["name"], x["description"]))


def findindex(data,url):
    for x in range(len(data)):
        if data[x]['url']== url:
            return x


def getcluster(index):
    return closest_projects(index)


vectorizer = TfidfVectorizer(stop_words = 'english', max_df = .7,
                            min_df = 75)
my_matrix = vectorizer.fit_transform([x[1] for x in documents]).transpose()


u, s, v_trans = svds(my_matrix, k=50)


words_compressed, _, docs_compressed = svds(my_matrix, k=30)
docs_compressed = docs_compressed.transpose()


word_to_index = vectorizer.vocabulary_
index_to_word = {i:t for t,i in word_to_index.items()}


word_to_index = vectorizer.vocabulary_


words_compressed = normalize(words_compressed, axis = 1)


def closest_words(word_in, k = 10):
    if word_in not in word_to_index: return "Not in vocab."
    sims = words_compressed.dot(words_compressed[word_to_index[word_in],:])
    asort = np.argsort(-sims)[:k+1]
    return [(index_to_word[i],sims[i]/sims[asort[0]]) for i in asort[1:]]


docs_compressed = normalize(docs_compressed, axis = 1)
def closest_projects(project_index_in, k = 15):
    sims = docs_compressed.dot(docs_compressed[project_index_in,:])
    asort = np.argsort(-sims)[:k+1]
    return [(documents[i][0],sims[i]/sims[asort[0]]) for i in asort[1:]]




def extract_cluster_ratings(cluster_list, index):
    lst=cluster_list(index)
    name=lst[1][0]
    index=findindex(data,name)
    return data[index]["url"]

# extract_cluster_ratings(closest_projects,0)


