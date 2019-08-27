import numpy as np
from sklearn.cluster import KMeans
import pandas as pd
import gensim
import warnings
from joblib import Parallel, delayed
import multiprocessing

if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    # EXP_HOME = "C:/My MSc/ThesisWorks/BigData_Code_Search/DeepGenQR/experiment"
    EXP_HOME = "F:/MyWorks/Thesis Works/Crowdsource_Knowledge_Base/DeepGenQR/experiment"
    # corpus_file = EXP_HOME + "/text-clustering/text8"
    # file = gensim.models.word2vec.Text8Corpus(corpus_file)
    # model = gensim.models.Word2Vec(file, size=100)
    # model_file = EXP_HOME + "/text-clustering/text-8_gensim"
    # model.save(model_file)
    # print("Model saved successfully!")

    model_file = EXP_HOME + "/text-clustering/text-8_gensim"
    model = gensim.models.Word2Vec.load(model_file)
    # print(model.wv['youtube'])
    katy_perry_csv = EXP_HOME + "/text-clustering/Youtube02-KatyPerry.csv"
    df = pd.read_csv(katy_perry_csv, encoding="latin-1")
    df = df.drop(['COMMENT_ID', 'AUTHOR', 'DATE'], axis=1)
    original_df = pd.DataFrame(df)
    df = df.drop(['CLASS'], axis=1)
    # print(df.head(10))

    final_data = []
    for i, row in df.iterrows():
        comment_vectorized = []
        comment = row['CONTENT']
        comment_all_words = comment.split(sep=" ")
        for comment_w in comment_all_words:
            try:
                comment_vectorized.append(list(model[comment_w]))
            except Exception as e:
                pass
        try:
            comment_vectorized = np.asarray(comment_vectorized)
            comment_vectorized_mean = list(np.mean(comment_vectorized, axis=0))
            # print(comment_vectorized_mean)
        except Exception as e:
            comment_vectorized_mean = list(np.zeros(100))
            pass
        try:
            len(comment_vectorized_mean)
        except:
            comment_vectorized_mean = list(np.zeros(100))

        temp_row = np.asarray(comment_vectorized_mean)
        final_data.append(temp_row)

    X = np.asarray(final_data)
    print(X)
    print('Conversion to array complete')

    # quit()

    print('Clustering Comments')
    # perform clustering
    clf = KMeans(n_clusters=2, n_jobs=4)
    clf.fit(X)
    print('Clustering complete')

    comment_label = clf.labels_
    comment_cluster_df = pd.DataFrame(original_df)
    comment_cluster_df['comment_label'] = np.nan
    comment_cluster_df['comment_label'] = comment_label
    print(comment_cluster_df.head(100))

    print('Saving to csv')
    comment_cluster_df.to_csv(EXP_HOME + '/text-clustering/comment_output-v2.csv', index=False)
