import os
from gensim.parsing.preprocessing import preprocess_string
from gensim.parsing.preprocessing import preprocess_documents
from gensim.parsing.preprocessing import strip_punctuation
from gensim.parsing.preprocessing import strip_multiple_whitespaces
from gensim.parsing.preprocessing import strip_non_alphanum
from gensim.parsing.preprocessing import remove_stopwords
import io

titleFolder = "F:/MyWorks/Thesis Works/Crowdsource_Knowledge_Base/CodeSearchBDA" \
              "/experiment/dataset/question-title-norm-ext"

qcodeFolder = "F:/MyWorks/Thesis Works/Crowdsource_Knowledge_Base/CodeSearchBDA/experiment" \
              "/dataset/question-norm-code-ext"


def list_textfiles(directory):
    "Return a list of filenames ending in '.txt' in DIRECTORY."
    textfiles = []
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            textfiles.append(filename)
    return textfiles


def read_first_line(directory, filename):
    with open(directory + "/" + filename) as f:
        first_line = f.readline()
        return first_line


title_qcode_corpus = "F:/MyWorks/Thesis Works/Crowdsource_Knowledge_Base/DeepGenQR/" \
                     "experiment/stackoverflow/title-qcode-corpus.txt"

CUSTOM_FILTERS = [lambda x: x.lower(), strip_multiple_whitespaces, strip_punctuation, remove_stopwords,
                  strip_non_alphanum]


def make_corpus(titleFolder, qcodeFolder):
    corpusLines = list()
    count = 0;
    for filename in list_textfiles(titleFolder):
        try:
            title = read_first_line(titleFolder, filename)
            qcode = read_first_line(qcodeFolder, filename)
            combined = title.strip() + " " + qcode.lower().strip()
            corpusLines.append(combined)
            # count = count + 1
            # if count == 100:
            #   break;
        except IOError:
            print("could not find")
    return corpusLines


# do nothing

# making the corpus

corpusFile = open(title_qcode_corpus, 'w')
lines = make_corpus(titleFolder, qcodeFolder)
for line in lines:
    corpusFile.write("%s\n" % line)

print("Title question code corpus created successfully!")
