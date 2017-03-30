import json
import numpy as np
from sklearn.cluster import KMeans
from sklearn import datasets, preprocessing, manifold
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import pairwise_distances
import math
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist, pdist
class analyze(object):
    def __init__(self, folder='', file="all.csv", delimiter=","):
        self.rootFolder = folder
        self.filename = file
        self.delimiter = delimiter

    def readFile(self):
        dataframe = pd.read_csv(self.rootFolder+self.filename, usecols=['State', 'District', 'Persons', 'Males', 'Females', 'Growth..1991...2001.', 'Number.of.households', 'Household.size..per.household.', 'Persons..literate', 'Males..Literate', 'Females..Literate', 'Matric.Higher.Secondary.Diploma', 'Graduate.and.Above', 'Primary', 'Middle'])
        dataframe.fillna(0, inplace=True)
        return dataframe
        

    def kElbow(self, data):
        data = data.drop(['State', 'District'], axis=1)
        data_normalized = preprocessing.normalize(data, norm='l2')
        Ks = range(1, 13)
        km = [KMeans(n_clusters=i).fit(data_normalized) for i in Ks]
        #score = [km[i].fit(data_normalized).score(data_normalized) for i in range(len(km))]
        #extract centroid
        centroids = [X.cluster_centers_ for X in km]
        #calculate euclidean dist
        k_euclid = [cdist(data_normalized, cent, 'euclidean') for cent in centroids]
        dist = [np.min(D,axis=1) for D in k_euclid]
    

        # Total with-in cluster sum of square
        wcss = [sum(d**2) for d in dist]
        #total sum of squares
        tss = sum(pdist(data_normalized)**2)/data_normalized.shape[0]
        bss = tss-wcss
        #Percentage of variance explained
        p = bss/tss*100
        elbowData = []
        count = 1
        for i in p:
            rowdata = {}
            rowdata['y'] = i
            rowdata['x'] = count
            count += 1
            elbowData.append(rowdata)
        return elbowData

    def performKMeansOnline(self, data, start, end):
        np.random.seed(5)
        data = data.drop(['State', 'District'], axis=1)
        data_normalized = preprocessing.normalize(data, norm='l2')
        clusterdata = {}
        graphElbowCluster = []
        clusterlabel = []
        clusterdata['data'] = graphElbowCluster
        clusterdata['labels'] = clusterlabel
        
        for i in range(start, end):
            estimator = KMeans(n_clusters=i)
            estimator.fit(data_normalized)
            labels = estimator.labels_
            for k in range(i):
                clusterlabel.append({"label" : np.argwhere(labels==k)})
            graphElbowCluster.append({"x":i,"y":estimator.inertia_})
        return clusterdata

    def randomSampling(self, data, clusterdata, limit=300):
        newdata = data.sample(n=limit)
        data = {}
        labels = []
        length = len(clusterdata['labels'])
        data['df'] = newdata
        data['cluster'] = labels
        for row in newdata.iterrows():
            for i in range(length):
                if row[0] in clusterdata['labels'][i]['label']:
                    labels.append(i)
                    break
        return data

    def adaptiveSampling(self, data, clusterdata, limit=300):
        length = len(clusterdata['labels'])
        counts = []
        for i in range(length):
            counts.append(clusterdata['labels'][i]['label'].size)
        norm = [int(round((float(i)/sum(counts))*100)) for i in counts]
        index = []
        labels = []
        for i in range(len(norm)):
            np.random.shuffle(clusterdata['labels'][i]['label'])
            index = np.append(index, clusterdata['labels'][i]['label'][0:norm[i]])
            labels = labels + [i for x in range(norm[i])]
        print labels
        adaptiveDF = data.ix[index]
        data = {}
        data['df'] = adaptiveDF
        data['cluster'] = labels
        return data



    def MDS(self, data, type, labels):
        mdata = data.drop(['State', 'District'], axis=1)
        if (type == 'EUCLID'):
            distances =  pairwise_distances(X = mdata, metric = "euclidean")
        else:
            distances =  pairwise_distances(X = mdata, metric = "correlation")
        mds = manifold.MDS(n_components=2, dissimilarity='precomputed')
        newdata = mds.fit_transform(distances)
        cluster = 0
        mdsdata = []
        for row in newdata:
            rowdata = {}
            rowdata['pointname'] = data.iloc[cluster][0]+','+str(data.iloc[cluster][3])
            rowdata['xvalue'] = row[0]
            rowdata['yvalue'] = row[1]
            rowdata['cluster'] = labels[cluster]
            mdsdata.append(rowdata)
            cluster +=1
        return mdsdata

    def scree(self, data):
        screedata = []
        data_scaled = pd.DataFrame(preprocessing.scale(data),columns = data.columns)
        pc = PCA(n_components=3)
        pc.fit_transform(data_scaled)
        print pd.DataFrame(pc.components_,columns=data_scaled.columns,index = ['PC-1','PC-2','PC-3'])
        #x =['Persons', 'Males', 'Females', 'Growth..1991...2001.', 'Number.of.households', 'Household.size..per.household.', 'Persons..literate', 'Males..Literate', 'Females..Literate', 'Matric.Higher.Secondary.Diploma', 'Graduate.and.Above', 'Primary', 'Middle']
        pca = PCA(n_components=13)
        data_normalized = preprocessing.normalize(data, norm='l2')
        pca.fit(data_normalized)

        count = 1
        for i in pca.explained_variance_:
            rowdata = {}
            rowdata['x'] = count
            rowdata['y'] = (i)

            count += 1
            screedata.append(rowdata)
        return screedata

    def PCA(self, data, labels):
        mdata = data.drop(['State', 'District'], axis=1)
        screedata = self.scree(mdata);
        pca = PCA(n_components=13)
        pca.fit(mdata.values)
        first_pc = pca.components_[0]
        second_pc = pca.components_[1]
        third_pc = pca.components_[2]
        print first_pc
        transformed_data = pca.transform(mdata)
        cluster = 0
        pcadata = []
        for row in transformed_data:
            rowdata = {}
            rowdata['pointname'] = data.iloc[cluster][0]+','+str(data.iloc[cluster][3])
            rowdata['xvalue'] = row[0]
            rowdata['yvalue'] = row[1]
            rowdata['cluster'] = labels[cluster]
            pcadata.append(rowdata)
            cluster +=1
        return pcadata, screedata
    
    def isomap(self, data, labels):
        mdata = data.drop(['State', 'District'], axis=1)
        mdata = manifold.Isomap(n_neighbors=5, n_components=2).fit_transform(mdata)
        cluster = 0
        isomapdata=[]
        for row in mdata:
            rowdata = {}
            rowdata['pointname'] = data.iloc[cluster][0]+','+str(data.iloc[cluster][3])
            rowdata['xvalue'] = row[0]
            rowdata['yvalue'] = row[1]
            rowdata['cluster'] = labels[cluster]
            isomapdata.append(rowdata)
            cluster +=1
        return isomapdata


