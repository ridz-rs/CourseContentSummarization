# Requires Python 3.6 or higher due to f-strings

# Import libraries
import platform
from tempfile import TemporaryDirectory
from pathlib import Path

import pytesseract
from pdf2image import convert_from_path
from PIL import Image
from spellchecker import SpellChecker
from nltk.tokenize import RegexpTokenizer
from nltk import word_tokenize



def IsMathSymbol(c):
	if 33 < ord(c) < 47 or 58 < ord(c) < 62 or ord(c) == 94:
		return True
	else:
		return False

def IsDictWord(word):
	spell = SpellChecker()
	return (word in [',', ';',':', '.', '\'','\"', '?', '!', '`']) or\
	    (word in spell)

def FindDictWord(text):
	word_start_index = 0
	word_end_index = min(text.find(' '), text.find('\n'))
	spell = SpellChecker()
	while word_end_index != -1:
		curr_word = text[word_start_index:word_end_index]
		if curr_word in spell:
			return word_start_index
		word_start_index = word_end_index + 1
		word_end_index = min(text[word_end_index:].find(' '), text[word_end_index:].find('\n'))
	return word_start_index	

def avg_word_length(word_list):
	return sum(len(word) for word in word_list) / len(word_list)

def GetMathTokensStructured(text):
	text = text.strip()
	sent_list = text.split('\n')
	for j, sent in enumerate(sent_list):
		word_list = word_tokenize(sent)
		if len(word_list) <= 5 or avg_word_length(word_list) <= 3:
			sent_list[j] = []
			continue
		last_dict_word_found = 0
		edit_word_list = []
		for i, word in enumerate(word_list):
			if IsDictWord(word):
				last_dict_word_found = i
				edit_word_list.append(word)
			if len(word)==1 and IsMathSymbol(word):
				if len(edit_word_list)>0 and edit_word_list[-1] != '<MATH>':
					edit_word_list.append('<MATH>')
		sent_list[j] = edit_word_list[:]

	new_lst = []
	for w_list in sent_list:
		new_lst.append(" ".join(w_list))

	return "\n".join(new_lst)

def texttiling(text):
	# Import libraries
	import nltk.tokenize.texttiling as texttiling
	import nltk
	from nltk.tokenize import RegexpTokenizer
	from nltk import word_tokenize
	from nltk.corpus import stopwords
	from nltk.tokenize import word_tokenize
	from nltk.stem import WordNetLemmatizer

	# Tokenize text
	# tokenizer = RegexpTokenizer(r'\w+')
	# tokens = tokenizer.tokenize(text)

	stop_words = set(stopwords.words('english'))


	# Remove stop words
	# text_split = text.split('\n')

	words_sent = text.split(' ')
	for i in range(len(words_sent)):
		word = words_sent[i]
		if '\n' in word:
			index = words_sent.index(word)
			two_words = word.split('\n')
			count = word.count('\n')
			words_sent.insert(index+1, two_words[0])
			words_sent.insert(index+2, '\n'*count)
			words_sent.insert(index+2, two_words[1])
			words_sent.remove(word)
			i += 3

	print(words_sent)
	stop_words_filtered_sentence = [word for word in words_sent if word not in stop_words]
	# print(stop_words_filtered_sentence)

	# Not doing this to preserve the paragraph structure
	# # Removed extra ['']
	# filtered_stop_words = [sent for sent in stop_words_filtered_sentence if sent != ['']]
	# # print(filtered_stop_words)

	# Lemmatize words
	lemmatizer = WordNetLemmatizer()
	lemmatized_words = [lemmatizer.lemmatize(word) for word in stop_words_filtered_sentence]
	# print(lemmatized_words)

	# Join words to sentences
	joined_sents = ' '.join(lemmatized_words)
	print(joined_sents)

	# Create TextTiling object
	tt = texttiling.TextTilingTokenizer()

	# Get segments
	segments = tt.tokenize(joined_sents)

	# Print segments
	# print(segments)

	# Print segments with text
	# for segment in segments:
	# 	print(' '.join(tokens[segment[0]:segment[1]]))

	return segments


def main():
	''' Main execution point of the program'''
	print('In main')

	if platform.system() == "Windows":
		# We may need to do some additional downloading and setup...
		# Windows needs a PyTesseract Download
		# https://github.com/UB-Mannheim/tesseract/wiki/Downloading-Tesseract-OCR-Engine

		pytesseract.pytesseract.tesseract_cmd = (
			r"C:/Program Files/Tesseract-OCR/tesseract.exe"
		)

		# Windows also needs poppler_exe
		path_to_poppler_exe = Path(r"D:/UTM/Fall2022/CSC493/Release-22.04.0-0/poppler-22.04.0/Library/bin")
		
		# Put our output files in a sane place...
		out_directory = Path(r"~/Desktop").expanduser()
	else:
		out_directory = Path("~").expanduser()	

	# Path of the Input pdf
	PDF_file = Path(r"D:\UTM\Fall2022\CSC493\CourseContentSummarization\PS3CSC373.pdf")

	# Store all the pages of the PDF in a variable
	image_file_list = []

	text_file = "D:/UTM/Fall2022/CSC493/CourseContentSummarization/PS3CSC373.txt"
	with TemporaryDirectory() as tempdir:
		# Create a temporary directory to hold our temporary images.

		"""
		Part #1 : Converting PDF to images
		"""

		if platform.system() == "Windows":
			pdf_pages = convert_from_path(
				PDF_file, 500, poppler_path=path_to_poppler_exe
			)
		else:
			pdf_pages = convert_from_path(PDF_file, 500)
		# Read in the PDF file at 500 DPI

		# Iterate through all the pages stored above
		for page_enumeration, page in enumerate(pdf_pages, start=1):
			# enumerate() "counts" the pages for us.

			# Create a file name to store the image
			filename = f"{tempdir}\page_{page_enumeration:03}.jpg"

			page.save(filename, "JPEG")
			image_file_list.append(filename)

		"""
		Part #2 - Recognizing text from the images using OCR
		"""

		with open(text_file, "w") as output_file:
			# Open the file in append mode so that
			# All contents of all images are added to the same file

			# Iterate from 1 to total number of pages
			for image_file in image_file_list:

				# Recognize the text as string in image using pytesserct
				text = str(((pytesseract.image_to_string(Image.open(image_file)))))

				text = text.replace("-\n", "")

				math_token_text = GetMathTokensStructured(text)
				# Finally, write the processed text to the file.
				output_file.write(math_token_text)

		
				
	
if __name__ == "__main__":
	# We only want to run this if it's directly executed!
	main()
