import os

corpusLines=list()
for file in os.listdir("./corpus"):
    print(file)
    with open("./corpus/"+file,"r") as f:
        lines=f.readlines()
        corpusLines.append(lines)
print(corpusLines)




