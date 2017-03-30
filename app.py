from flask import Flask, request, jsonify
from flask import render_template

from datamanipulation import analyze
import json

app = Flask(__name__)
folder = ''
file = "all.csv"
delimiter = ","
RANDOMSAMPLING = 0
ADAPTIVESAMPLING = 1
PCA = 0
MDS_EUCLIDEAN = 1
MDS_COSINE = 2
MDS_CORRELATION = 3
ISOMAP = 4

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/generateClusterElbow")
def generateK():
    d = analyze(folder, file, delimiter)
    data = d.readFile()
    elbowData = d.kElbow(data)
    print elbowData
    return jsonify(data = elbowData)

@app.route("/display")
def Displaycharts():
    clusters = int(request.args.get('clusters','3',type=str))
    sampling = int(request.args.get('sampling','0',type=str))
    viz = int(request.args.get('viz','0',type=str))
    #print clusters, sampling, viz
    d = analyze(folder, file, delimiter)
    data = d.readFile()
    clusterdata = d.performKMeansOnline(data, clusters, clusters+1)
    if sampling == RANDOMSAMPLING:
        sampleData = d.randomSampling(data, clusterdata)
    elif sampling == ADAPTIVESAMPLING:
        sampleData = d.adaptiveSampling(data, clusterdata)
    screedata=[]
    if viz == PCA:
        data, screedata = d.PCA(sampleData['df'],sampleData['cluster'])
    elif viz == ISOMAP:
        data = d.isomap(sampleData['df'],sampleData['cluster'])
    else:
        if viz == MDS_EUCLIDEAN:
            data = d.MDS(sampleData['df'], "EUCLID", sampleData['cluster'])
        elif viz == MDS_COSINE:
            data = d.MDS(sampleData['df'], "COSINE", sampleData['cluster'])
        else:
            data = d.MDS(sampleData['df'], "CORRELATION", sampleData['cluster'])
    return jsonify(data = data, scree = screedata)

if __name__ == "__main__":
   app.run(host='0.0.0.0',port=5000,debug=True)