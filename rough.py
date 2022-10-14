from pytesseract_extraction import texttiling

with open('PS3CSC373.txt', 'r') as f:
    text = f.read() 
    segments = texttiling(text)

with open('PS3CSC373_tt.txt', 'w') as f: 
    # for segment in segments:
    f.write(segments)