import nltk.tokenize.texttiling as texttiling
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re

def RemoveStopWords(word_lst):
    stop_words = set(stopwords.words('english'))
    stop_words_filtered_sentence = [word for word in word_lst if word.lower() not in stop_words]
    return stop_words_filtered_sentence

def GetWordList(text):
    new_text = text.replace('\n', '\n<PARASEP>')
    new_text = new_text.replace(' ','<PARASEP>')
    word_lst = new_text.split('<PARASEP>')
    word_lst = [word for word in word_lst if word != '']
    return word_lst

def GetSentList(text):
    text = text.lower()
    text = text.replace('.', '<SEP>')
    text = text.replace('!', '<SEP>')
    text = text.replace('?', '<SEP>')
    text = text.replace('<SEP>”', '”<SEP>')
    sent_list = text.split('<SEP>')
    return sent_list

def Lemmatize(word_lst):
    lemmatizer = WordNetLemmatizer()
    lemmatized_words = []
    for word in word_lst:
        words = re.findall(r"\w+", word)
        if words:
            new_word = re.sub(r"\w+", lemmatizer.lemmatize(words[0]), word)
            lemmatized_words.append(new_word)
        else:
            lemmatized_words.append(word)
    return lemmatized_words

def JoinWordLst(word_lst):
    text = ' '.join(word_lst)
    return text.replace('\n ', '\n')

