# Import libraries
import nltk.tokenize.texttiling as texttiling
from nltk.corpus import stopwords
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
    return segments

