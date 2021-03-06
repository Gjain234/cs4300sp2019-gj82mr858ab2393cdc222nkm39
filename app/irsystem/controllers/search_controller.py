from . import *
from app.irsystem.models.helpers import *
from app.irsystem.models.helpers import NumpyEncoder as NumpyEncoder
from app.irsystem.similarity import *
from app.irsystem.svd import *
import json
import ast

project_name = "Get StartTED: TED Talk Recommendation System"
# net_id = "Andrea Benson ab2393, Caroline Chang cdc222, Nandita Mohan nkm39, Gauri Jain gj82, Michael Rivera mr858"
net_id = "Andrea Benson, Caroline Chang, Nandita Mohan, Gauri Jain, Michael Rivera"
cat_q = ["Other"]
mood_q = ["Informative"]

documents = []
data2=json.load(open("ted_main.json", encoding="utf8"))
for x in data2:
        documents.append((x["name"], x["description"]))

def process_single_prompt(url): #functionality could be in a js file as well
	url_parts = url.split('=')
	prompt = url_parts[1]
	words = prompt.split('+')
	final_str = ""
	for w in words:
		final_str = final_str + " " + w
	return final_str

@irsystem.route('/', methods=['GET'])
def search():
	query = request.args.get('search')
	cat = request.args.get('category')

	if cat!="Select Category" and cat!=None:
		cat_q.append(cat)
	elif cat!=None:
		cat_q.append("Other")
	mood = request.args.get('mood')

	if(mood!="Mood Preference" and mood!=None):
		mood_q.append(mood)
	elif mood!=None:
		mood_q.append("Informative")
	if not query:
		data = []
		output_message = ''
		return render_template('search.html', name=project_name, netid=net_id, output_message=output_message, data=data)

	else:
		output_message = query
		topic_vids = combined_search(query)
		video_url = get_prompt1_video_link(query)

		# '''S  V   D '''
		idx = findindex_url(data2, video_url)
		vectorizer = TfidfVectorizer(stop_words = 'english', max_df = .8,
		                            min_df = 40)
		my_matrix = vectorizer.fit_transform([x[1] for x in documents]).transpose()
		words_compressed, _, docs_compressed = svds(my_matrix, k=30)
		docs_compressed = docs_compressed.transpose()
		docs_compressed = normalize(docs_compressed, axis = 1)


		query_category = cat_q.pop()
		query_mood = mood_q.pop()
		mood_vids = top_svd(data2, idx, query_mood, docs_compressed)
		lifestyle_vids = comment_search(query,query_category.lower())
		data = [mood_vids, topic_vids, lifestyle_vids]
		return render_template('results.html', output_message=output_message, data=data, video_url = video_url, n=0, mood=query_mood, category=query_category)
