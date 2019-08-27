from gensim.models import Word2Vec
from gensim.models import FastText
from gensim.models.word2vec import LineSentence
from gensim.parsing.preprocessing import preprocess_string
from gensim.parsing.preprocessing import strip_punctuation
from gensim.parsing.preprocessing import strip_multiple_whitespaces
from gensim.parsing.preprocessing import strip_non_alphanum
from gensim.parsing.preprocessing import remove_stopwords
from gensim.parsing.preprocessing import stem_text
import pandas as pd

# EXP_HOME =" F:/MyWorks/Thesis Works/Crowdsource_Knowledge_Base/DeepGenQR/experiment"
# EXP_HOME = "C:/My MSc/ThesisWorks/BigData_Code_Search/DeepGenQR/experiment"

data_file = 'C:/My MSc/ThesisWorks/Machine_Learning_using_Big_Data/MyRCBot/experiment/chatbot-data/Twitter/corpus.txt'
model_file = 'C:/My MSc/ThesisWorks/Machine_Learning_using_Big_Data/MyRCBot/chatbot/experiment/chatbot-data/Twitter'

df = pd.read_csv(data_file, encoding='latin-1')

# create custom filter
CUSTOM_FILTERS = [lambda x: x.lower(), strip_multiple_whitespaces, strip_punctuation, remove_stopwords]

corpus_lines = list()

for index, row in df.iterrows():
    sentence = row[5]
    pp_sentence = preprocess_string(sentence, CUSTOM_FILTERS)
    # print(pp_sentence)
    corpus_lines.append(' '.join(pp_sentence))
    # vocab.update(pp_sentence)
    # count = count + 1
    # break
    # if count == 100:
    #   break
# print(corpus_lines)

model = Word2Vec(corpus_lines, size=300, window=5, min_count=2, workers=8, sg=1)

# print(model)

# save the model
# model.save(model_file)

print("Saved the Model successfully!")
# loading the model
# model = Word2Vec.load(modelFile)

# ft_model = FastText(sentences, workers=8)

# word2vec
# print(model.wv['java'])

# fast_text
# print(ft_model.wv['eclipse'])
