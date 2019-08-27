from gensim.models import FastText
from gensim.models.word2vec import LineSentence
from gensim.parsing.preprocessing import preprocess_string
from gensim.parsing.preprocessing import strip_punctuation
from gensim.parsing.preprocessing import strip_multiple_whitespaces
from gensim.parsing.preprocessing import strip_non_alphanum
from gensim.parsing.preprocessing import remove_stopwords

# EXP_HOME = "F:/MyWorks/Thesis Works/Crowdsource_Knowledge_Base/DeepGenQR/experiment"
# EXP_HOME = "C:/My MSc/ThesisWorks/BigData_Code_Search/DeepGenQR/experiment"

# titleFile = EXP_HOME + '/GitHub/use.rawcode/github-deepcs.txt'
titleFile = 'C:/My MSc/ThesisWorks/Machine_Learning_using_Big_Data/MyRCBot/' \
            'experiment/chatbot-data/Google/google-corpus.txt'
# modelFile = EXP_HOME + '/pymodel/github-deepcs'
modelFile = 'C:/My MSc/ThesisWorks/Machine_Learning_using_Big_Data/MyRCBot/experiment/chatbot-data' \
            '/Google/google-talks'

# word_vectors = KeyedVectors.load_word2vec_format("/tmp/vectors.txt", binary=False)  # C text format
# create custom filter
CUSTOM_FILTERS = [lambda x: x.lower(), strip_multiple_whitespaces, strip_punctuation, remove_stopwords,
                  strip_non_alphanum]
sentences = LineSentence(open(titleFile, 'r'), max_sentence_length=1000, limit=None)

# NLP without stemming
pre_processed = list()
for sentence in sentences:
    #     # print(' '.join(sentence))
    temp = ' '.join(sentence)
    pp_sentence = preprocess_string(temp, CUSTOM_FILTERS)
    #     print(pp_sentence)
    pre_processed.append(pp_sentence)

model = FastText(sentences, size=300, window=5, min_count=5, workers=8)
# save the model
model.save(modelFile)

print(model)

print("Saved the FastText Model !")
# loading the model
# model = Word2Vec.load(modelFile)

# ft_model = FastText(sentences, workers=8)


# word2vec
# print(model.wv['java'])

# fast_text
# print(ft_model.wv['eclipse'])
