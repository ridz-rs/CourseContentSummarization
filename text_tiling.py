# Import libraries
import nltk.tokenize.texttiling as texttiling

from preprocessing import GetWordList, GetSentList, Lemmatize, RemoveStopWords, JoinWordLst

def TextTiling(text):
    sentences = GetSentList(text)
    text_final = []
    for sent_in_text in sentences:
        word_lst = GetWordList(sent_in_text)
        stop_words_filtered_word_list = RemoveStopWords(word_lst)
        lemmatized_words = Lemmatize(stop_words_filtered_word_list)
        cleaned_sent = JoinWordLst(lemmatized_words)
        text_final.append(cleaned_sent)
    cleaned_text = '. '.join(text_final)
    tt = texttiling.TextTilingTokenizer()
    segments = tt.tokenize(cleaned_text)
    return segments