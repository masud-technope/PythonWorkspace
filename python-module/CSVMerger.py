import pandas as pd

# EXP_HOME = "F:/MyWorks/Thesis Works/Crowdsource_Knowledge_Base/DeepGenQR/experiment"
EXP_HOME = "C:/My MSc/ThesisWorks/BigData_Code_Search/DeepGenQR/experiment"
corpus_folder = EXP_HOME + "/stackoverflow/java-question-title"
master_title_file = EXP_HOME + "/stackoverflow/java-question-title/master-java-title.txt"

title_entries = list()
for index in range(1, 30):
    corpus_file = corpus_folder + "/" + str(index) + ".csv"
    data_frame = pd.read_csv(corpus_file)
    entries = list(data_frame['Title'])
    print(len(entries))
    # title_entries.append(entries)
    title_entries.extend(entries)

print(len(title_entries))
outf_file = open(master_title_file, 'w', encoding="utf-8")
for line in title_entries:
    outf_file.write("%s\n" % line)

print("Saved the corpus !")
