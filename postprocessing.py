from preprocessing import GetWordList, GetSentList, Lemmatize, RemoveStopWords, JoinWordLst

def post_processing(summary):
    sentences_in_summary = summary.split('.')
    sentences_in_text = GetSentList(text)
    for sent_in_text in sentences_in_text:
        word_lst = GetWordList(sent_in_text)
        stop_words_filtered_word_list = RemoveStopWords(word_lst)
        lemmatized_words = Lemmatize(stop_words_filtered_word_list)
        cleaned_text = JoinWordLst(lemmatized_words)

        for sent_in_summary in sentences_in_summary:
            if cleaned_text.lower().strip() == sent_in_summary.lower().strip():
                sentences_in_summary[sentences_in_summary.index(sent_in_summary)] = sent_in_text
                break
    for sentence in sentences_in_summary:
        print(sentence + '\n')
    new_summary = '. '.join(sentences_in_summary)
    return new_summary