

from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import time

start_time = time.time()

corpusFile = 'C:/MyWorks/Thesis Works/Crowdsource_Knowledge_Base/CodeTokenRec/experiment/evaluation/175/iman-tse/' \
             'IJaDataset-corpus-stemmed.txt'
sentences = LineSentence(open(corpusFile, 'r'), max_sentence_length=10000, limit=None)
model = Word2Vec(sentences, size=100, window=5, min_count=5, workers=8)


time_elapsed = time.time()-start_time
time.strftime("%H:%M:%S", time.gmtime(time_elapsed))

print(model.wv['time'])
