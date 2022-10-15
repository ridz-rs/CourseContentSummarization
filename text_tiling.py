# Import libraries
from webbrowser import get
import nltk.tokenize.texttiling as texttiling
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


def RemoveStopWords(word_lst):
    stop_words = set(stopwords.words('english'))
    stop_words_filtered_sentence = [word for word in word_lst if word not in stop_words]
    return stop_words_filtered_sentence


def GetWordList(text):
    new_text = text.replace('\n', '\n<PARASEP>')
    new_text = new_text.replace(' ','<PARASEP>')
    word_lst = new_text.split('<PARASEP>')
    word_lst = [word for word in word_lst if word != '']
    return word_lst


def Lemmatize(word_lst):
    lemmatizer = WordNetLemmatizer()
    lemmatized_words = [lemmatizer.lemmatize(word) for word in word_lst]
    return lemmatized_words

def JoinWordLst(word_lst):
    text = ' '.join(word_lst)
    return text.replace('\n ', '\n')

def TextTiling(text):
    word_lst = GetWordList(text)
    stop_words_filtered_word_list = RemoveStopWords(word_lst)
    lemmatized_words = Lemmatize(stop_words_filtered_word_list)
    cleaned_text = JoinWordLst(lemmatized_words)
    tt = texttiling.TextTilingTokenizer()
    segments = tt.tokenize(cleaned_text)
    print(segments)
    return segments


# def texttiling(text):
# 	# Tokenize text
# 	# tokenizer = RegexpTokenizer(r'\w+')
# 	# tokens = tokenizer.tokenize(text)



#     word_lst = get_word_lst(text)
# 	# print(stop_words_filtered_sentence)

# 	# Not doing this to preserve the paragraph structure
# 	# # Removed extra ['']
# 	# filtered_stop_words = [sent for sent in stop_words_filtered_sentence if sent != ['']]
# 	# # print(filtered_stop_words)

# 	# Lemmatize words
# 	lemmatizer = WordNetLemmatizer()
# 	lemmatized_words = [lemmatizer.lemmatize(word) for word in stop_words_filtered_sentence]
# 	# print(lemmatized_words)

# 	# Join words to sentences
# 	joined_sents = ' '.join(lemmatized_words)
# 	print(joined_sents)

# 	# Create TextTiling object
# 	tt = texttiling.TextTilingTokenizer()

# 	# Get segments
# 	segments = tt.tokenize(joined_sents)

# 	# Print segments
# 	# print(segments)

# 	# Print segments with text
# 	# for segment in segments:
# 	# 	print(' '.join(tokens[segment[0]:segment[1]]))

# 	return segments