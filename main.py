import os
from io import open
from cleaner import clean
from gensim.models import doc2vec


def load_docs(filepath, clean_text=True):
	ret = []
	for f_name in os.listdir(filepath):
		if clean_text:
			ret.append(clean(open(filepath+"/"+f_name,'rb').read().decode('UTF-8')))
			continue
		ret.append(open(filepath+"/"+f_name,'rb').read().decode('UTF-8'))
	return ret

def format_loaded_input_docs(docs):
	ret = []
	for id_, doc in enumerate(docs):
		ret.append(doc2vec.TaggedDocument(words=doc.split(), tags=[id_]))
	return ret

if __name__ == '__main__':
	docs = load_docs("samples/")
	formatted_docs = format_loaded_input_docs(docs)
	model = doc2vec.Doc2Vec(formatted_docs, size=5, window=10, workers=2)
	for e in model.docvecs:
		print e
