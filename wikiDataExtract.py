# Import packages
import wikipedia

text_file = open('source', "w")

# Pages to be extracted from Wikipedia
vocab = ['Diabetes', 'Cardiovascular disease', 'Cancer', 'Alzheimer\'s disease', 'Genetically modified organism', 'Tuberculosis']

for i in vocab: 
    # Extract the plain text content of the page, excluding images, tables, and other data. 
    text = wikipedia.page(i).content
    # Replace '==' with ''
    text = text.replace('==', '')
    # Replace '\n' with ''
    text = text.replace('\n', '')
    text_file.write(text)       

text_file.close()

