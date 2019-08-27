from gensim.models import Word2Vec
from sklearn.decomposition import PCA
import numpy as np
import numpy.core as npc
import keras
from keras_preprocessing.text import one_hot
from keras_preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Embedding
from keras.layers import Activation
from keras.layers import LSTM
from keras.layers import Dropout
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import LabelEncoder
from keras import utils
from gensim.parsing.preprocessing import preprocess_string
from gensim.parsing.preprocessing import strip_punctuation
from gensim.parsing.preprocessing import strip_multiple_whitespaces
from gensim.parsing.preprocessing import strip_non_alphanum
from gensim.parsing.preprocessing import remove_stopwords
from gensim.parsing.preprocessing import stem_text
import pandas as pd
import tensorflow as tf
from sklearn import model_selection
from sklearn.preprocessing import MultiLabelBinarizer
from keras.optimizers import SGD

# EXP_HOME = "F:/MyWorks/Thesis Works/Crowdsource_Knowledge_Base/DeepGenQR/experiment"
EXP_HOME = "C:/My MSc/ThesisWorks/BigData_Code_Search/DeepGenQR/experiment"

tag = 'eclipse'

q1 = EXP_HOME + "/stackoverflow/" + tag + "/q1.csv"
q2 = EXP_HOME + "/stackoverflow/eclipse/q2.csv"
q3 = EXP_HOME + "/stackoverflow/eclipse/q3.csv"

a1 = EXP_HOME + "/stackoverflow/" + tag + "/a1.csv"
a2 = EXP_HOME + "/stackoverflow/" + tag + "/a2.csv"
a3 = EXP_HOME + "/stackoverflow/eclipse/a3.csv"
a4 = EXP_HOME + "/stackoverflow/eclipse/a4.csv"

p1 = pd.read_csv(q1)
p2 = pd.read_csv(q2)
p3 = pd.read_csv(q3)
p4 = pd.read_csv(a1)
p5 = pd.read_csv(a2)
p6 = pd.read_csv(a3)
p7 = pd.read_csv(a4)

p4["Title"] = ""
p5["Title"] = ""
p6["Title"] = ""
p7["Title"] = ""

frames = [p1, p2, p3, p4, p5, p6, p7]

# frames = [p1, p4, p5]
eclipse_df = pd.concat(frames)

eclipse_csv_file = EXP_HOME + "/stackoverflow/" + tag + "/eclipse-qa.csv"
eclipse_df.to_csv(eclipse_csv_file, header=True, index=False)

saved_df = pd.read_csv(eclipse_csv_file)
print(saved_df)
