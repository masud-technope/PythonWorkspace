from gensim.models.word2vec import LineSentence
from gensim.parsing.preprocessing import preprocess_string
from gensim.parsing.preprocessing import preprocess_documents
from gensim.parsing.preprocessing import strip_punctuation
from gensim.parsing.preprocessing import strip_multiple_whitespaces
from gensim.parsing.preprocessing import strip_non_alphanum
from gensim.parsing.preprocessing import remove_stopwords
import os

questionFolder = "F:/MyWorks/Thesis Works/Crowdsource_Knowledge_Base/CodeSearchBDA/experiment/dataset/question-ext/"
titleNormFolder = "F:/MyWorks/Thesis Works/Crowdsource_Knowledge_Base/CodeSearchBDA/experiment/dataset/" \
                  "question-title-norm-ext/"


def list_textfiles(directory):
    "Return a list of filenames ending in '.txt' in DIRECTORY."
    textfiles = []
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            textfiles.append(filename)
    return textfiles


def read_first_line(filename):
    with open(questionFolder + "/" + filename) as f:
        first_line = f.readline()
        return first_line


def write_file(content, outputfile):
    myFile = open(outputfile, 'w')
    myFile.write("%s\n" % content)


CUSTOM_FILTERS = [lambda x: x.lower(), strip_multiple_whitespaces, strip_punctuation, remove_stopwords,
                  strip_non_alphanum]


def read_corpus(directory):
    # "Read and tokenize all files in DIRECTORY."
    corpus = []
    for filename in list_textfiles(directory):
        # corpus.append(tokenize(read_first_line(filename)))
        first_line = read_first_line(filename)
        cleaned = preprocess_string(first_line, CUSTOM_FILTERS)
        content2Save = ' '.join(cleaned)
        write_file(content2Save, titleNormFolder + "/" + filename)


read_corpus(questionFolder)
