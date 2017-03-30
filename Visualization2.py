import sys
import numpy as np
import scipy.stats as ss
import pandas as pd

import random
from sklearn import cluster as Kcluster, metrics as SK_Metrics
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap,MDS
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import Normalizer




def find_pca(data_frame):
    pca = PCA(n_components=2)
    return pd.DataFrame(pca.fit_transform(data_frame))


def adaptive_sampling(data_frame, cluster_count,fraction):
    k_means = Kcluster.KMeans(n_clusters=cluster_count)
    k_means.fit(data_frame)

    data_frame['label'] = k_means.labels_
    adaptiveSampleRows = []
    for i in range(cluster_count):
         adaptiveSampleRows.append(data_frame.ix[random.sample(data_frame[data_frame['label'] == i].index, (int)(len(data_frame[data_frame['label'] == i])*fraction))])

    adaptiveSample = pd.concat(adaptiveSampleRows)
    del adaptiveSample['label']

    return adaptiveSample


def random_sampling(data_frame, fraction):
    rows = random.sample(data_frame.index, (int)(len(data_frame)*fraction))
    return benefitDF.ix[rows]


def find_iso(dataframe):
    component_count = 2
    iso = Isomap(n_components=component_count)
    return pd.DataFrame(iso.fit_transform(dataframe))


def find_MDS(dataframe, type):
    dis_mat = SK_Metrics.pairwise_distances(dataframe, metric = type)
    mds = MDS(n_components=2, dissimilarity='precomputed')
    return pd.DataFrame(mds.fit_transform(dis_mat))

data_directory = "/Users/Aman/Desktop/Projects/Vis2/project/data/"
def createFile(random_sample, adaptive_sample, file_name):
    random_sample.columns = ["r1","r2"]
    adaptive_sample.columns = ["a1","a2"]
    sample = random_sample.join([adaptive_sample])

    file_name = data_directory + file_name
    sample.to_csv(file_name, sep=',')



def calculate_values(random_sample, adaptive_sample,function,file_name):
    createFile(function(random_sample), function(adaptive_sample),file_name +".csv")


def performLSA(samplefile):
    text_file = open(data_directory + "Sample.txt")
    list = text_file.read().split(".")
    list = list[0:len(list) - 1]
    stemmer = SnowballStemmer("english")

    newList = []
    for sentence in list:
        stemmedSentece = ""
        for word in sentence.split(" "):
            stemmedSentece += " " + stemmer.stem(word)
        newList.append(stemmedSentece)

    vectorizer = TfidfVectorizer(min_df=1)
    X = vectorizer.fit_transform(newList)
    idf = vectorizer.idf_
    dict(zip(vectorizer.get_feature_names(), idf))

    vectorizer = CountVectorizer(min_df=1, stop_words='english')
    dtm = vectorizer.fit_transform(newList)
    pd.DataFrame(dtm.toarray(), index=newList, columns=vectorizer.get_feature_names
    ()).head(10)

    vectorizer.get_feature_names()

    lsa = TruncatedSVD(2, algorithm='arpack')
    dtm_lsa = lsa.fit_transform(dtm)
    dtm_lsa = Normalizer(copy=False).fit_transform(dtm_lsa)
    pd.DataFrame(lsa.components_, index=["component_1", "component_2"], columns=
    vectorizer.get_feature_names())
    lsa_samplefile = pd.DataFrame(dtm_lsa, index=newList, columns=["component_1", "component_2"])
    lsa_samplefile.columns = ["r1", "r2"]
    lsa_samplefile["type"] = 1
    lsa_samplefile.to_csv(data_directory + "sampleFileLSA.csv", sep=',')

def main():
    benefitDF = pd.read_csv("/Users/Aman/Desktop/Projects/Vis2/project/Benefits.csv")
    random_sample = random_sampling(benefitDF, 0.2)
    adaptive_sample = adaptive_sampling(benefitDF,10, 0.2)

    calculate_values(random_sample,adaptive_sample,find_pca,"pca")
    calculate_values(random_sample,adaptive_sample,find_iso,"iso")

    list_mds = ["euclidean","cosine","correlation"]
    for type_mds in list_mds:
        createFile(find_MDS(random_sample,type_mds),find_MDS(adaptive_sample,type_mds),type_mds + ".csv")


main()