from pytesseract_extraction import GetMathTokensStructured
with open("C:/Users/Riddhesh/Desktop/out_text.txt", "r") as f:
    text = f.read()

out_text = GetMathTokensStructured(text)
f1 = "C:/Users/Riddhesh/Desktop/without_removing_small_sents.txt"
f2 = "C:/Users/Riddhesh/Desktop/remove_small_sents.txt"
with open(f2, "w") as f:
    f.write(out_text)