import os
from gensim.models import Word2Vec
from gensim.models import FastText

# EXP_HOME = "F:/MyWorks/Thesis Works/Crowdsource_Knowledge_Base/DeepGenQR/experiment"
EXP_HOME = "C:/My MSc/ThesisWorks/BigData_Code_Search/DeepGenQR/experiment"
model_file = EXP_HOME + '/pymodel/eclipse-fasttext-model'
model = FastText.load_fasttext_format(model_file)
word_file = EXP_HOME + '/w2vec-data/words.txt'
vec_file = EXP_HOME + '/w2vec-data/eclipse-vector.txt'
vec_lines = list()
words = open(word_file, 'r')
for word in words:
    try:
        if model.__contains__(word.strip()):
            vector = model[word.strip()]
            line = word.strip() + " " + ' '.join(str(x) for x in vector)
            print(line)
            vec_lines.append(line)
        else:
            print("could not find " + word)
    except IOError:
        print("Failed to get the vector of " + word)
        pass

output_file = open(vec_file, 'w')
for content in vec_lines:
    output_file.write("%s\n" % content)
output_file.close()
