from nltk.corpus import stopwords
import nltk, string
from nltk.stem.wordnet import WordNetLemmatizer

lmtzr = WordNetLemmatizer()
exclude = set(string.punctuation)

def remove_punctuations(text, customlist):
	if not customlist:
		customlist = exclude
	for punc in customlist:
		text = text.replace(punc, " ")
	return text.lower()

def handle_encoding(text):
	text = text.encode("utf-8").decode("ascii","ignore")
	return text

def remove_stopwords(text):
	tokenized_words = text.split()
	filtered_words = []
	for word in tokenized_words:
		if not word.lower() in stopwords.words('english'):
			filtered_words.append(word)
	text = " ".join(filtered_words)
	return text

def remove_alphanumerics(text):
	txt = []
	for each in text.split():
		if not any(x in each.lower() for x in "0123456789"):
			txt.append(each)
	txtsent = " ".join(txt)
	return txtsent 

def lemmatize_text(sent):
	lem = []
	for each in sent.split():
		lemma = lmtzr.lemmatize(each,'v')
		if lemma == each:
			lemma = lmtzr.lemmatize(each,'n')
		lem.append(lemma)
	lemsent = " ".join(lem)
	return lemsent

def remove_small_words_and_digits(text):
	new_text = []
	for each in text.split():
		if len(each)>2 and not each.isdigit():
			new_text.append(each)
	return " ".join(new_text)

def clean(text):
	text = handle_encoding(text)
	text = remove_punctuations(text, customlist = [])
	text = remove_stopwords(text)
	text = lemmatize_text(text)
	text = remove_small_words_and_digits(text)
	text = remove_alphanumerics(text)
	return str(text)
