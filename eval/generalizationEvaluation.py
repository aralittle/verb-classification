# -*- coding: utf-8 -*-
'''
Created on 29/06/2016

@author: lara

'''

from collections import OrderedDict
import operator
import pandas
import os
import numpy
import argparse
from scipy.spatial.distance import pdist, cosine, dice, euclidean
import pickle
import time
import Feat_Based


metrics1 = {'cosine': cosine, 'dice': dice}


def GetMedoids(dataMatrixTrain, clustering):
    '''
    Obtains the centroids of a clustering
    :param dataMatrixTrain: training data, pandas object
    :param clustering: {numCluster:[senses],...}
    :return: a dic that contains the cluster ID's and their corresponding centroid arrays
    Centroid: where the center of the cluster lies
    '''

    dic = {}
    for cluster in clustering:
        verbs = clustering[cluster]
        m = []
        for v in verbs:
            vector = dataMatrixTrain.loc[v]
            m.append(vector)
        M = numpy.array(m)
        #print 'medoid imput',M
        medoid = numpy.mean(M, 0)
        dic[cluster] = medoid
    return dic


def classifyExamplesMedoid(dataMatrixTest, dist, dic_medoids):
    '''
    Classify test verbs in cluster groups according to their distance to the medoid of the group
    :param dataMatrixTest: testdata, pandas object
    :param dist: distance used to create the clustering
    :param dic_medoids: dict object containing the medoids of a clustering
    :return: dict object {sense:[int(cluster), int(cluster)], ...}
    '''

    resultAssign = {}

    for verb in dataMatrixTest.iterrows():

        distances = []
        clust = []

        lema = verb[0]
        data = verb[1]
        resultAssign[lema] = []

        for c in dic_medoids:
            dist1 = dist(dic_medoids[c], data)
            distances.append(dist1)
            clust.append(c)

        indices = [i for i, j in enumerate(distances) if j == min(distances)]
        minC = [clust[i] for i in indices] #closer cluster(s)
        for C in minC:
            resultAssign[lema].append(C)

    return resultAssign


def classifyExamplesSingle(dataMatrixTest, dataMatrixTraining, clustering, dist):
    '''
    Classify test verbs in cluster groups according to their distance to the closest
     sense of each group

    :param dataMatrixTest: pandas object, test data
    :param dataMatrixTraining: pandas object, training data
    :param clustering: {numCluster:[senses],...}
    :param dist: distance included in the clustering
    :return: dict object {sense:[int(cluster), int(cluster)], ...}
    '''

    resultAssign = {}
    for verb in dataMatrixTest.iterrows():

        distances = []
        lemmas = []

        lema = verb[0]
        data = verb[1]
        resultAssign[lema] = []

        for verb2 in dataMatrixTraining.iterrows():
            lema2 = verb2[0]
            data2 = verb2[1]
            dist1 = dist(data, data2)  # dist entre dos verbos
            distances.append(dist1)
            lemmas.append(lema2)

        indices = [i for i, j in enumerate(distances) if j == min(distances)]
        minLemas = [lemmas[i] for i in indices]

        # recover cluster ID of the closest senses to the test sense
        for L in minLemas:  # lmost similar senses
            for cluster, verbs in clustering.iteritems():
                if L in verbs:
                    if cluster not in resultAssign[lema]:
                        resultAssign[lema].append(cluster)

    return resultAssign


def classifyExamplesComplete(dataMatrixTest, dataMatrixTraining, clustering, dist):
    '''
    Classify test verbs in cluster groups according to their distance to the furthest
     sense of each group. The test sense gets the clusterID of the less far sense
    :param dataMatrixTest: pandas object, test data
    :param dataMatrixTraining: pandas object, training data
    :param clustering: {numCluster:[senses],...}
    :param dist: distance used in the clustering
    :return: dict object {sense:[int(cluster), int(cluster)], ...}
    '''

    resultAssign = {}
    for verb in dataMatrixTest.iterrows():
        temp = {}
        lema = verb[0]
        data = verb[1]
        resultAssign[lema] = []

        for cluster, verbs in clustering.iteritems():
            biggest = None #furtherst element in cluster

            for verb2 in dataMatrixTraining.iterrows():
                lema2 = verb2[0]
                if lema2 in verbs:
                    data2 = verb2[1]
                    dist1 = dist(data, data2)

                    if biggest == None:
                        biggest = dist1

                    else:
                        if dist1 >= biggest:
                            biggest = dist1

            temp[cluster] = biggest  # biggest distance with a element in a cluster

        best = [key for key, val in temp.iteritems() if val == min(temp.values())]
        for e in best:
            resultAssign[lema].append(e)

    return resultAssign


def classifyExamplesAverage(dataMatrixTest, dataMatrixTraining, clustering, dist):
    '''
    Classify test verbs in cluster groups according to their average distance to all the
    senses of each group.
    :param dataMatrixTest: pandas object
    :param dataMatrixTraining: pandas object
    :param clustering: {numCluster:[senses],...}
    :param dist: distance used in the clustering
    :return: dict object {sense:[int(cluster), int(cluster)], ...}
    '''

    resultAssign = {}
    for verb in dataMatrixTest.iterrows():
        temp = {}

        lema = verb[0]
        data = verb[1]
        resultAssign[lema] = []

        for cluster, verbs in clustering.iteritems():
            dist_media = None

            for verb2 in dataMatrixTraining.iterrows():
                lema2 = verb2[0]

                if lema2 in verbs:

                    data2 = verb2[1]
                    dist1 = dist(data, data2)
                    if dist_media:
                        dist_media += dist1
                    else:
                        dist_media = dist1

            if dist_media:  # average distance to all of the verbs in the cluster
                df = dist_media / float(len(verbs))
                temp[cluster] = df

        closest = [key for key, val in temp.iteritems() if val == min(temp.values())]
        for e in closest:
            resultAssign[lema].append(e)

    return resultAssign


##evaluar con las features mas relevantes de la clase o con las de f1


def normalizar(vector_orig, vector_a_escalar):
    '''
    Normalise vector according to the scale of another vector
    :param vector_orig: vector that servers as a model
    :param vector_a_escalar: vector that has to be scaled, frequencies of components
    :return: normalised vector
    '''

    normalizado = []
    rango_x = max(vector_orig) - min(vector_orig)
    rango_y = max(vector_a_escalar) - min(vector_a_escalar)

    for c in range(len(vector_a_escalar)):
        try:
            valor_normal = (vector_a_escalar[c] - min(vector_a_escalar)) * float(rango_x) / rango_y + min(vector_orig)
            normalizado.append(valor_normal)
        except Exception as inst:  #if it is flat it does not work (all feats weight the same)
            return None
    return normalizado


def distance_from_point(g, vector_x, vector_normalizado):
    '''
    Measures the distance to each component to the origin [0,0]
    http://ieeexplore.ieee.org/document/4378393/
    :param g: labels for components, ordered from more to less weight
    :param vector_x: vector that contains the indices of the components[0,1,2,3,n]
    :param vector_normalizado: vector that contains the frequency of each component normalised according to vector x
    :return: list containing distances, Ordered dict {componentName:distance}
    # vector_x = [1, 2, 3, 4] la primera barrita en el 0 o en el 1?
    '''

    dicy = OrderedDict()
    counter = []
    for i in range(len(vector_x)):
        label = g[i]
        x = vector_x[i]
        y = vector_normalizado[i]

        distance = euclidean([0, 0], [x, y])
        dicy[label] = distance
        counter.append(distance)
    return counter, dicy


def cutInFreqFeats(counter, ordict):
    '''
    Obtains the most frequent components
    :param counter: list of distances of each component to the origin
    :param ordict: dict object that contains ordered features according to their frequency
    :return: ordered dict object that contains only the most frequent and important features (according to distance to [0,0])
    '''

    cut_positions = [i for i, j in enumerate(counter) if j == min(counter)] #index of the minimum distance
    cut_position = cut_positions[-1]
    feats = ordict.keys()
    fea_freq = ordict.values()
    #feature corresppnding to cut position is not included

    imp_feat = feats[:cut_position]
    imp_values = fea_freq[:cut_position]

    o = OrderedDict()  # dict containing most freq feat and their freqs
    for i in range(len(imp_feat)):
        key = imp_feat[i]
        o[key] = imp_values[i]
    return o


def getFreqFeats(fileTraining, clustering):
    '''
    Obtain the most important features for each cluster (turmo)
    :param fileTraining: csv file with training data
    :param clustering: {numCluster:[senses],...}
    :return: ordered dict {clusterID:{orderedDict(feat: frequency), ...}, ...}
    '''

    result = OrderedDict()
    dicValues = Feat_Based2refactored.csvToDic(fileTraining)
    weightDic = Feat_Based2refactored.WeightCF(clustering, dicValues)  # dict {cluster:[feats:weight]}
    for c in weightDic:
        feats = weightDic[c]
        #order features-components according to decreasing order of frequency [('tag2', 10), ('tag3', 7), ('tag1', 1)]
        sorted_v_h = sorted(feats.items(), key=operator.itemgetter(1), reverse=True)
        ordict = OrderedDict(sorted_v_h)  # store it in dict

        feats = ordict.keys()  # features
        normalizado = normalizar(range(len(ordict)), ordict.values())  # el range is number of feats
        if normalizado:
            #distance to origin for all features
            counter, dicy = distance_from_point(feats, range(len(ordict)), normalizado) #most important feats
            import_feat = cutInFreqFeats(counter, ordict)  # {feat:weight}
            result[c] = import_feat

    return result


####### Evaluate with most relevant features: check compatibility between semantic schemas of verbs and classes

### UNSEEN VERBS
def evalCompatNV(op0, op1, op2, op3, tr, method, file_roles_test, dataMatrixTest, dataMatrixTraining,
                 clustering, dist):
    '''
    It evaluates the ability of the most representative features for a cluster to label sentences associated
    to unseen verbs that have been assigned to these clusters.
    :param op0: maximal features (train roles) according to weight, {cluster:[feat,feat], ...}
    :param op1: maximal features (train roles) according to recall, {cluster:[feat,feat], ...}
    :param op2: maximal features (train roles) according to f-measure, {cluster:[feat,feat], ...}
    :param op3: all features (train roles) whose frequency is !=0
    :param method: method used to create the clustering
    :param file_roles_training: csv file that contains train verbs with semantic role information
    :param file_roles_test: csv file that contains test verbs with semantic role information
    :param dataMatrixTest: pandas object, test verbs with features
    :param dataMatrixTraining: pandas object, training verbs with features
    :param clustering: # {numCluster:[verbs], ...}
    :param dist: distance metric used
    :return: [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]
    Each sublist is a criterion (peso, reacall, f-measure, turmo, all features of cluster),
    first index is how many cluster features there are in the verb,
    second index is how many verb features are in the cluster
    '''


    # 1. Classify unseen verbs in clusters created with training data

    if method == 'single':
        assigned = classifyExamplesSingle(dataMatrixTest, dataMatrixTraining, clustering, dist)
    if method == 'average':
        assigned = classifyExamplesAverage(dataMatrixTest, dataMatrixTraining, clustering, dist)

    if method == 'complete':
        assigned = classifyExamplesComplete(dataMatrixTest, dataMatrixTraining, clustering, dist)


    dverbos1 = Feat_Based2refactored.csvToDic(file_roles_test) # file of test verbs (to be classified) with roles
    totalRecall = [0, 0, 0, 0, 0] #for all test verbs, all assigned clusters
    totalPrecision = [0, 0, 0, 0, 0]

    for verb, listac in assigned.iteritems():  ##listac is a list of clusters in which verb can fit
    # assigned: dict object {sense:[int(cluster), int(cluster)], ...}

        # 2. obtain roles for the test verb

        feats4verb = {k: v for k, v in dverbos1[verb].iteritems() if float(v) != 0} # delete roles that have 0 frequency
        feats = feats4verb.keys()
        #print 'test', verb, feats

        # 3. initialize evaluation for each verb #1
        clustersRecall = [0, 0, 0, 0, 0]
        clustersPrecision = [0, 0, 0, 0, 0]

        # 4. initialize evaluation for each cluster assigned to verb
        for c in listac: #each candidate cluster

            # 5. Get sets of important roles for each cluster obtained with different criteria
            feats0 = []  # peso
            feats1 = []  # recall
            feats2 = []  # fmeasure
            feats3 = []  # turmo
            feats4 = {}  # all features of cluster

            if c in op0:  # op0 is a dict, {cluster:[feat,feat], ...}
                feats0 = op0[c]
            if c in op1:  # op1 is a dict, {cluster:[feat,feat], ...}
                feats1 = op1[c]
            if c in op2:  # op2 is a dict, {cluster:[feat,feat], ...}
                feats2 = op2[c]
            if c in op3:
                feats4 = op3[c]
            if c in tr:
                feats3 = [key for key, val in tr[c].iteritems()]  # dict object
                # feats3 = [{orderedDict(feat: frequency), ...}]

            possib = [feats0, feats1, feats2, feats3, feats4]

            # 6. compare cluster and verb role features
            for index, listaF in enumerate(possib):
                testF = set(feats) #gold
                trainF = set(listaF) #hip

                TP = len(testF)-len(testF.difference(trainF))
                FN = len(testF.difference(trainF))
                FP = len(trainF.difference(testF))

                #print 'TP', TP, 'FN', FN, 'FP', FP
                if TP+FN != 0:
                    recall = float(TP)/(TP+FN)
                else:
                    recall = 0
                if TP+FP != 0:
                    precision = float(TP)/(TP+FP)
                else:
                    precision = 0
                clustersRecall[index] += recall
                clustersPrecision[index] += precision


        for index, val in enumerate(clustersRecall):
            totalRecall[index] += clustersRecall[index] / float(len(listac))
            totalPrecision[index] += clustersPrecision[index] / float(len(listac))

    # obtain average for all verbs, all clusters for each criterion
    finalRecall = [r/float(len(assigned)) for r in totalRecall]
    finalPrecision = [r/float(len(assigned)) for r in totalPrecision]

    return finalRecall, finalPrecision


def getClosest4baseline(csvTraining, csvTest, dist):
    '''
    For each test verb, finds the most similar in training
    :param csvTraining: csv containing training data, linguistic feats
    :param csvTest: csv containing test data, unseen verbs, linguistic feats
    :param dist: distance used
    :return: dic that contains closest verbs for each test sense {verb:[closeSense, closesense], ...}
    '''
    train = pandas.read_csv(csvTraining, index_col=0, header=0)
    test = pandas.read_csv(csvTest, index_col=0, header=0)

    d = {}
    for e in test.iterrows():
        verb = e[0]
        vals = e[1]

        closestv = []
        distancia = []
        for e2 in train.iterrows():
            verb2 = e2[0]
            vals2 = e2[1]

            closestv.append(verb2)
            distancia.append(dist(vals, vals2))

        indices = [i for i, j in enumerate(distancia) if j == min(distancia)]
        minSenses = [closestv[i] for i in indices]

        d[verb] = minSenses

    return d


def baseline(dic_closest_verbs, csvTraining, csvTest):
    '''
    Measures the accuracy or recall given by the closest verb to test verb according to linguistic data
    Two things are measured: how many of the roles in the sim verb are in the test verb and
    how many of the roles of the test verb are in the similar verbs
    :param dic_closest_verbs: dict object that contains all unseen test senses
    and a list of their closest training senses
    :param csvTraining: csv with training data, role information
    :param csvTest: csv with test verbs and role information
    :return: the average of ratios of coincident features for all test verbs,
    float (feats in similar verb that are in test verb), float (#feats in test verb that are in similar verb)
    '''

    #print 'baseline NV----'
    dverbosTrain = Feat_Based2refactored.csvToDic(csvTraining)
    dverbosTest = Feat_Based2refactored.csvToDic(csvTest)

    # for each test verb, list of similar verbs
    TotalRecall = 0
    TotalPrecision = 0

    for v1, vsim in dic_closest_verbs.iteritems():
        featTest = {k: v for k, v in dverbosTest[v1].iteritems() if float(v) != 0} #dict object that contains the roles and their frequency for a given test verb

        SimsensesRecall = 0
        SimsensesPrecision = 0

        for sentido in vsim: # for each similar ver to a given test verb, obtain its roles

            featTrain = {k: v for k, v in dverbosTrain[sentido].iteritems() if float(v) != 0}

            testF = set(featTest.keys())
            trainF = set(featTrain.keys())

            TP = len(testF) - len(testF.difference(trainF))
            FN = len(testF.difference(trainF))
            FP = len(trainF.difference(testF))
            if TP + FN != 0:
                recall = float(TP) / (TP + FN)
            else:
                recall = 0
            if TP + FP != 0:
                precision = float(TP) / (TP + FP)
            else:
                precision = 0

            SimsensesRecall+= recall
            SimsensesPrecision += precision

        TotalRecall += SimsensesRecall/ float(len(vsim))
        TotalPrecision += SimsensesPrecision / float(len(vsim))

    RecallFinal = TotalRecall / float(len(dic_closest_verbs))
    PrecisionFinal = TotalPrecision / float(len(dic_closest_verbs))
    return RecallFinal, PrecisionFinal






def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output',  default='../evalresults/task/', help='Output folder for the evaluation files with results')
    parser.add_argument('-i', '--input',  default='../clusterings/sampe2batch/', help='Folder that contains the clusterings')
    parser.add_argument('-c', '--csvs', default='../GeneratedData/',
                        help='Folder that contains the csv files with features for the clustering algorithm')
    parser.add_argument('-r', '--withRoles', default='../GeneratedData/withRoles/', help='Folder that contains csvs with roles for evaluation')
    parser.add_argument('-s', '--selection', default=None, nargs='*', help='List of clusterings that should be evaluated')
    args = parser.parse_args()

    resultsDir = args.output
    csvRoles = args.withRoles

    infoCsv = 'fileResultsEvaluado,' + 'method_clustering,' + 'TipoRolesAsignado,' + 'NV-MFeats-Weight-Recall,' + 'NV-MFeats-Weight-Precision,' \
              + 'NV-MFeats-Weight-F1,' + 'NV-MFeats-Recall-Recall,' + 'NV-MFeats-Recall-Precision,' + 'NV-MFeats-Recall-F1,' + 'NV-MFeats-F-Recall,' + 'NV-MFeats-F-Precision,' + 'NV-MFeats-F-F1,' \
              + 'NV-Convergent-Recall,' + 'NV-Convergent-Precision,' + 'NV-Convergent-F1,' \
              + 'NV-AllFeats-Recall,' + 'NV-AllFeats-Precision,' + 'NV-AllFeats-F1,' \
              + 'NV-baseline-Recall,' + 'NV-baseline-Precision,' + 'NV-baseline-F1\n'

    for file in os.listdir(args.input):  # clustering outputs, clusterings made with file
        if args.selection is None or file in args.selection[0]:
            on_task = open(os.path.join(resultsDir, file), 'a')
            on_task.write(infoCsv)

            print('evaluating cluserings with these feats: ', file)
            f = open(os.path.join(args.input + file), 'r')
            clustering_dic = pickle.load(f)

            for data, clustering in clustering_dic.iteritems():  # info, clustering
                ldata = data.split('_')
                numC = ldata[1]
                method = ldata[2]
                metr = ldata[3]
                metric = metrics1[metr]
                clusterDict = clustering[0]

                if int(numC) in [i for i in range(2, 201,3)]: #in case the evaluation was interrupted
                    print(file, time.strftime("%X"))

                    csvTraining = args.csvs+ 'training/'+ file
                    csvNV = args.csvs+ 'testNV/' + file
                    dataMatrixTrainFeats = pandas.read_csv(csvTraining, index_col=0, header=0)
                    dataMatrixTestNVFeats = pandas.read_csv(csvNV, index_col=0, header=0)

                    for fi2 in os.listdir(csvRoles+'training/'): #file being evaluated

                        on_task.write(file + ',' + data + ',' + fi2 + ',')
                        csvRolesTrain = csvRoles+'training/' + fi2
                        dicValues = Feat_Based2refactored.csvToDic(csvRolesTrain)
                        AllFeats = Feat_Based2refactored.getAllFeats(csvRolesTrain)
                        dic_closest_verbs = getClosest4baseline(csvTraining, csvNV, metric)

                        op0, op1 = Feat_Based2refactored.setmaximalFeats(clusterDict, dicValues, AllFeats)
                        op2 = Feat_Based2refactored.setmaximalFeatsNew(clusterDict, dicValues, AllFeats)  # criterio: f-measure
                        op3 = Feat_Based2refactored.clusterFeats(clusterDict, dicValues)
                        tr = getFreqFeats(csvRolesTrain, clusterDict)  ##convex
                        lista_ev_NVRecall, lista_ev_NVPrecision = evalCompatNV(op0, op1, op2, op3, tr, method,
                                                                                          csvRoles + 'testNV/' + fi2,
                                                                                          dataMatrixTestNVFeats,
                                                                                          dataMatrixTrainFeats, clusterDict,
                                                                                          metric)

                        baselineNVR, baselineNVP = baseline(dic_closest_verbs, csvRolesTrain, csvRoles + 'testNV/' + fi2)

                        for i, val in enumerate(lista_ev_NVRecall):
                            F1 = Feat_Based2refactored.Fmeasurelocal(lista_ev_NVPrecision[i], val)
                            on_task.write(str(round(val, 4)) + ',' + str(round(lista_ev_NVPrecision[i], 4)) + ',' + str(
                                round(F1, 4)) + ',')

                        F1 = Feat_Based2refactored.Fmeasurelocal(baselineNVP, baselineNVR)
                        on_task.write(str(round(baselineNVR, 4)) + ',' + str(round(baselineNVP, 4)) + ',' + str(round(F1, 4)) + ',')
                        on_task.write('\n')
            on_task.close()



if __name__ == '__main__':
    main()
