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
cat_q = list()
mood_q = list()

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
	if(cat!="Select Category" and cat!=None):
		cat_q.append(cat)
	mood = request.args.get('mood')
	if(mood!="Mood Preference" and mood!=None):
		mood_q.append(mood)
	#rel = mood_q.pop()


	# print("get ready:")
	# print(cat_q)
	# print(mood_q)
	if not query:
		data = []
		output_message = ''
		return render_template('search.html', name=project_name, netid=net_id, output_message=output_message, data=data)

	else:
		# https://www.ted.com/talks/kakenya_ntaiya_empower_a_girl_transform_a_community?utm_campaign=tedspread&utm_medium=referral&utm_source=tedcomshare
		output_message = query
		prompt1 = process_single_prompt(request.url)
		#video_url= "https://embed.ted.com/talks/colin_powell_kids_need_structure"
		topic_vids = combined_search(prompt1)
                # data = combined_search(prompt1)
		video_url = get_prompt2_video_link(query)

		# '''S  V   D '''
		with open("ted_main.json", encoding="utf8") as f:
			documents = []
			data2=json.load(f)
			for x in data2:
				documents.append((x["name"], x["description"]))
		idx = findindex(data2, video_url)
		vectorizer = TfidfVectorizer(stop_words = 'english', max_df = .8,
		                            min_df = 40)
		my_matrix = vectorizer.fit_transform([x[1] for x in documents]).transpose()
		words_compressed, _, docs_compressed = svds(my_matrix, k=30)
		# words_compressed = normalize(words_compressed, axis=1) #fixmeh
		docs_compressed = docs_compressed.transpose()
		docs_compressed = normalize(docs_compressed, axis = 1)

		cluster = closest_projects(idx, docs_compressed)
		mood = mood_q.pop()
		catg = cat_q.pop()
		#ec = extract_cluster_ratings(data2, idx, mood)
		mood_vids = top_svd(data2, idx, mood)
		lifestyle_vids = comment_search(query,catg.lower())
		data = [mood_vids, topic_vids, lifestyle_vids]
		return render_template('results.html', output_message=output_message, data=data, video_url = video_url, n=0, mood=mood, category=catg)
