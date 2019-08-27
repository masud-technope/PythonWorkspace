from sklearn.feature_extraction.text import TfidfVectorizer
from keras.preprocessing.text import Tokenizer

# # list of text documents
# text = ["The quick brown fox jumped over the lazy dog.",
#         "The dog.",
#         "The fox"]
# # create the transform
# vectorizer = TfidfVectorizer()
# # tokenize and build vocab
# vectorizer.fit(text)
# # summarize
# print("Vocabulary")
# print(vectorizer.vocabulary_)
# print("IDF")
# print(vectorizer.idf_)
# # encode document
# vector = vectorizer.transform([text[0]])
# # summarize encoded vector
# print("Vector shape")
# print(vector.shape)
# print("Vector of doc0")
# print(vector.toarray())

#  doing things with keras
# define 5 documents
docs = ['Well done!',
        'Good work',
        'Great effort',
        'nice work',
        'Excellent!']
# create the tokenizer
t = Tokenizer()
print(t)
# fit the tokenizer on the documents
t.fit_on_texts(docs)
# summarize what was learned
print(t.word_counts)

for k, v in t.word_counts.items():
    print(k, v)

print(t.document_count)

# print(t.word_index)
print(t.word_docs)
# integer encode documents
encoded_docs = t.texts_to_matrix(docs, mode='count')
print(encoded_docs)

# using word-embedding in ML
