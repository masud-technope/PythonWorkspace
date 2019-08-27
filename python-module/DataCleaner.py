
from gensim.models.word2vec import LineSentence
from gensim.parsing.preprocessing import  preprocess_string
from gensim.parsing.preprocessing import strip_punctuation
from gensim.parsing.preprocessing import strip_multiple_whitespaces
from gensim.parsing.preprocessing import strip_non_alphanum
from gensim.parsing.preprocessing import remove_stopwords

# EXP_HOME = "F:/MyWorks/Thesis Works/Crowdsource_Knowledge_Base/DeepGenQR/experiment"
EXP_HOME = "C:/My MSc/ThesisWorks/BigData_Code_Search/DeepGenQR/experiment"
csv_file = EXP_HOME+"/stackoverflow/eclipse/eclipse-qa.csv"

CUSTOM_FILTERS = [lambda x: x.lower(), strip_multiple_whitespaces, strip_punctuation, remove_stopwords,
                  strip_non_alphanum]
sentences = LineSentence(open(csv_file, 'r'), max_sentence_length=100000, limit=None)
pre_processed = list()
for sentence in sentences:
    # print(' '.join(sentence))
    temp = ' '.join(sentence)
    pp_sentence = preprocess_string(temp, CUSTOM_FILTERS)
    # print(pp_sentence)
    pre_processed.append(' '.join(pp_sentence))

# saving the pre-processed to the file
myFile = open(pp_raw_code, 'w')
for line in pre_processed:
    myFile.write("%s\n" % line)

print("Corpus preprocessed successfully!")






