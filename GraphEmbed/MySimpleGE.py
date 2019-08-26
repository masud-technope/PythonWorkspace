from node2vec import Node2Vec
import re
from string import punctuation
import networkx as nx
import os
import glob


# text = ("When Sebastian Thrun started working on self-driving cars at "
#         "Google in 2007, few people outside of the company took him "
#         "seriously. \"I can tell you very senior CEOs of major American "
#         "car companies would shake my hand and turn away because I wasn’t "
#         "worth talking to,\" said Thrun, in an interview with Recode earlier "
#         "this week.")


def make_corpus(corpusDir):
    corpusLines = list()
    for file in os.listdir("./corpus"):
        print(file)
        with open(corpusDir + "/" + file, "r") as f:
            # lines = f.readlines()
            content=f.read()
            corpusLines.append(content)
    # print(corpusLines)
    return " ".join(corpusLines)


text = make_corpus("./corpus")

# keyword selection from the texts
# words = text.translate(None, str.punctuation)
# words = " ".join(c for c in text if c not in punctuation)
refinedText = " ".join(word.strip(punctuation) for word in text.split())
words = refinedText.lower().split(" ")

# String previousToken = new String();
# 				String nextToken = new String();
# 				String currentToken = tokens[index];
# 				if (index > 0)
# 					previousToken = tokens[index - 1];
#
# 				if (index < tokens.length - 1)
# 					nextToken = tokens[index + 1];
#
# 				// now add the graph nodes
# 				if (!graph.containsVertex(currentToken)) {
# 					graph.addVertex(currentToken);
# 				}
# 				if (!graph.containsVertex(previousToken)
# 						&& !previousToken.isEmpty()) {
# 					graph.addVertex(previousToken);
# 				}
# 				if (!graph.containsVertex(nextToken) && !nextToken.isEmpty()) {
# 					graph.addVertex(nextToken);
# 				}
#
# 				// adding edges to the graph
# 				if (!previousToken.isEmpty())
# 					if (!graph.containsEdge(currentToken, previousToken)) {
# 						graph.addEdge(currentToken, previousToken);
# 					}
#
# 				if (!nextToken.isEmpty())
# 					if (!graph.containsEdge(currentToken, nextToken)) {
# 						graph.addEdge(currentToken, nextToken);
# 					}


# now develop a text graph

graph = nx.Graph()

for index in range(0, len(words)):
    currentWord = words[index]
    prevWord = ""
    nextWord = ""

    if index > 0:
        prevWord = words[index - 1]

    if index < len(words) - 1:
        nextWord = words[index + 1]

    graph.add_node(currentWord)

    if not prevWord != "" and currentWord != "":
        graph.add_edge(prevWord, currentWord)

    if nextWord != "" and currentWord != "":
        graph.add_edge(nextWord, currentWord)

    if index == 0:
        graph.add_edge(nextWord, currentWord)
    if index == len(words) - 1:
        graph.add_edge(prevWord, currentWord)

# now feed the graph to the model
node2vec = Node2Vec(graph, dimensions=64, walk_length=30, num_walks=200, workers=4)

# Any keywords acceptable by gensim.Word2Vec can be passed, `diemnsions` and `workers` are automatically passed (from the Node2Vec constructor)

# FILES
EMBEDDING_FILENAME = './my-first-embeddings.emb'
EMBEDDING_MODEL_FILENAME = './my-node2vec.model'

# Look for most similar nodes
model = node2vec.fit(window=10, min_count=1, batch_words=4)

print(model.wv.most_similar("published"))  # Output node names are always strings

# Save embeddings for later use
# model.wv.save_word2vec_format(EMBEDDING_FILENAME)

# Save model for later use
# model.save(EMBEDDING_MODEL_FILENAME)
