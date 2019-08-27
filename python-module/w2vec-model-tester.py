
from gensim.models import Word2Vec
from gensim.models import FastText
from gensim.models import KeyedVectors

# EXP_HOME = "C:/My MSc/ThesisWorks/BigData_Code_Search/DeepGenQR/experiment"
# EXP_HOME = "F:/MyWorks/Thesis Works/Crowdsource_Knowledge_Base/DeepGenQR/experiment"
# model_file = 'C:/My MSc/ThesisWorks/Machine_Learning_using_Big_Data/MyRCBot/chatbot/WikiVec/wiki-news-300d-1M.vec'
model_file = 'C:/My MSc/ThesisWorks/Machine_Learning_using_Big_Data/MyRCBot/chatbot/TwitterVec/twitterFT'

# en_model = KeyedVectors.load_word2vec_format(model_file)

model = FastText.load(model_file)
print(model)
print(len(model.wv.vocab))
print(model.wv['car'])

# Pick a word
find_similar_to = 'car'

# Finding out similar words [default= top 10]
for similar_word in model.similar_by_word(find_similar_to):
    print("Word: {0}, Similarity: {1:.2f}".format(
        similar_word[0], similar_word[1]
    ))




