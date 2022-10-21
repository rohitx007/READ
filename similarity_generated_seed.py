from sentence_transformers import SentenceTransformer
import scipy
import nltk
import random
# import multiprocessing
# from multiprocessing import Pool
from statistics import mean
import re
# chencherry = nltk.translate.bleu_score.SmoothingFunction()

# SAMPLES = 15
# REFSIZE = 5000
# def run_f(ele):
# 	reference, fn, weight = ele
# 	BLEUscore_f = nltk.translate.bleu_score.sentence_bleu(reference, fn, weight)  
# 	return BLEUscore_f

def save_mapping(text_output_file, seed_path, final_output_path, model_name = 'bert-base-nli-mean-tokens'):
	
	model = SentenceTransformer(model_name, device='cuda')

	with open(text_output_file,'r', encoding='utf-8') as file:
		my_file = file.read()

	sentences = my_file.split("\n")
	sentences = list(set(sentences))
	
	with open(seed_path,'r', encoding='utf-8') as file:
		my_file = file.read() 

	seed_data = my_file.split("\n")

	sentence_embeddings = model.encode(sentences)


	queries = list(set(seed_data))
	query_embeddings = model.encode(queries)

	number_top_matches = 500 #@param {type: "number"}
	# gen_real_map = []
	# print("Semantic Search Results")
	with open(final_output_path,'a', encoding='utf-8') as file:
		for query, query_embedding in zip(queries, query_embeddings):
			if query!="":
				distances = scipy.spatial.distance.cdist([query_embedding], sentence_embeddings, "cosine")[0]

				results = zip(range(len(distances)), distances)
				results = sorted(results, key=lambda x: x[1])
				# print (results)
				# if results[5][1]<0.1:
				
				# print("\n\n======================\n\n")
			
				# file.write("\n"+"Real       -->  "+query+"\n")
				# print("Query:", query)
				# print("\nTop 5 most similar sentences in corpus:")
					
				for idx, distance in results[0:number_top_matches]:
					if distance<0.15:
						generated_sentence_non_repeatitive = re.sub(r'\b(\w+) \1\b', r'\1', sentences[idx].strip(), flags=re.IGNORECASE)
						generated_sentence_non_repeatitive = generated_sentence_non_repeatitive.strip()
						file.write(query+"\t"+generated_sentence_non_repeatitive+"\t"+str(1-distance)+"\n")
			# file.write("\n"+"_"*80+"\n")


	# with open(final_output_path+"_generated_to_real",'w', encoding='utf-8') as file:
	# 	total_distance = []
	# 	query_list = []              # reference
		
		
	# 	for sentence, sentence_embedding in zip(sentences, sentence_embeddings):
	# 		distances = scipy.spatial.distance.cdist([sentence_embedding], query_embeddings, "cosine")[0]

	# 		results = zip(range(len(distances)), distances)
	# 		# idx = distances.index(distances.min())
	# 		results = sorted(results, key=lambda x: x[1])
		
	# 		file.write(sentence.strip())
	# 		file.write(" | ")
				
	# 		for idx, distance in results[0:1]:
	# 			closest_query = queries[idx].strip()
	# 			file.write(closest_query+"\n")
	# 			query_list.append(closest_query)
	# 			total_distance.append(distance)



	# with open(os.path.join(output_dir,"average_distance_metric"),'a', encoding='utf-8') as file:
	# 	file.write(final_output_path+"  --------------  "+str(mean(total_distance))+"\n")

# 	with open(final_output_path+"_sentences", 'w') as file:
# 		for i in sentences:
# 			file.write(i+"\n")

# 	with open(final_output_path+"_closest_query", 'w') as file:
# 		for i in query_list:
# 			file.write(i+"\n")

	


# def evaluate_bleu(text_output_file, seed_path, final_output_path):

# with open(final_output_path+"_sentences",'r', encoding='utf-8') as file:
# 	my_file = file.read()
# 	sentences = my_file.split("\n")

# with open(final_output_path+"_closest_query",'r', encoding='utf-8') as file:
# 	my_file = file.read()
# 	query_list = my_file.split("\n")

	# hypothesis = random.sample(sentences, len(sentences))




	# ref2 = hypothesis[:REFSIZE]

	# for ngram in range(2, 6):
	# 	weight = tuple((1. / ngram for _ in range(ngram)))
	# 	pool = Pool(multiprocessing.cpu_count())
	# 	bleu_irl = pool.map(run_f, [(query_list, sentences[i], weight) for i in range(SAMPLES)])
	# 	bleu_irl2 = pool.map(run_f, [(ref2, query_list[i], weight) for i in range(SAMPLES)])
	# 	pool.close()
	# 	pool.join()

	# 	with open(final_output_path+'_bleu_metric','a') as file:
	# 	# file.write('irl_text'+"\n")

	# 		file.write(str(len(weight)) + '-gram BLEU(b) score : ' + str(1.0 * sum(bleu_irl2) / len(bleu_irl2)) + '\n')
	# 		file.write(str(len(weight)) + '-gram BLEU(f) score : ' + str(1.0 * sum(bleu_irl) / len(bleu_irl)) + '\n')
