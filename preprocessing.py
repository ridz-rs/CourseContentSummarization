from spellchecker import SpellChecker

def check_word_spelling(word):
    spell = SpellChecker()    

    if word.lower() in spell:
        return True
    else:
        return False



filename = "C:/Users/Riddhesh/Desktop/out_text.txt"

with open(filename, 'r') as test:
    test_data = test.read().split('\n')
    j = 0
    for line in test_data:
        line = line.replace('\n', '')
        i = 0
        line_list = line.split(' ')
        for word in line_list:
            if check_word_spelling(word) == False:
                line_list[i] = "<MATH TOKEN>"
            i += 1
        test_data[j] = ' '.join(line_list)
        j += 1
    # print(test_data)

with open(r"C:/Users/Riddhesh/Desktop/final_text.txt", 'w') as test:
    for line in test_data:
        test.write(line + "\n")



