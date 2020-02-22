#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'lara'
import warnings
warnings.filterwarnings("ignore")

import imp
import os
import pickle
import numpy
from sklearn import metrics
from coclust.evaluation.external import accuracy
import argparse
import f1_clustering



def evalComparativaVmeasure(data, clusterDict, labels, possibilities):

	predicted_labels = []
	for verbG in labels: #true
		for claseAuto, verbs in clusterDict.iteritems():
			if verbG in verbs:
				predicted_labels.append(claseAuto)
				break

	scores = numpy.array([0.0,0.0,0.0])
	for p in possibilities:
		true_labels = []
		setPos = set(p)
		mapped = {pos:i for i, pos in enumerate(setPos)}
		for pos in p:
			true_labels.append(mapped[pos])

		hom_com_v = metrics.homogeneity_completeness_v_measure(true_labels, predicted_labels)
		scores += numpy.array(hom_com_v)

	return {data : scores/len(possibilities)}

def evalComparativaAccuracy(data, clusterDict, labels, possibilities):

	predicted_labels = []
	for verbG in labels: #gold
		for claseAuto, verbs in clusterDict.iteritems():
			if verbG in verbs:
				predicted_labels.append(claseAuto)
				break

	scores = numpy.array([0.0])
	for p in possibilities:
		true_labels = []
		setPos = set(p)
		mapped = {pos:i for i, pos in enumerate(setPos)}
		for pos in p:
			true_labels.append(mapped[pos])

		ac = accuracy(true_labels, predicted_labels)
		contingency_matrix = metrics.cluster.contingency_matrix(true_labels, predicted_labels)
		purity = numpy.sum(numpy.amax(contingency_matrix, axis=0)) / float(numpy.sum(contingency_matrix)) 

		fm = (2*ac*purity)/float(ac+purity)
		scores += [fm]

	return {data : scores/len(possibilities)}



def evalComparativaAveraged_F1(data, clusterDict,labels, possibilities):
	'''
	Evaluates clustering against gold classes
	:param data: data used to create clustering (str)
	:param clusterDict: automatic clustering
	:param labels: verbs used in both gold and auto
	:param possibilities: list of lists. Sublist: classes for the verbs
	:return:
	'''

	dic = {}
	scoreSum = 0
	Clusterdic = {}

	for verb in labels:
		for claseAuto, verbs in clusterDict.iteritems():
			if verb in verbs:
				if not verb in Clusterdic:
					Clusterdic[verb]= [claseAuto]
				else:
					Clusterdic[verb].append(claseAuto)

	counter = 0
	for posib in possibilities: #lista de tags
		gold = {}
		for index, verb in enumerate(labels):
			gold[verb]= [posib[index]]

		tempset = set()
		for v in Clusterdic:
			tempset.add(Clusterdic[v][0])

		score = moduleF1.f1(Clusterdic, gold)
		scoreSum += score
		counter +=1

	valD = scoreSum/float(counter)
	keyD = data+'_'+str(counter)
	dic[keyD] = [valD]
	return dic  # {fileRoles_numGrupos_sh: [rand, mu,hom_com_v]} #



def GoldComparSenSemClass(outputs, results, fileV):
	'''
	Compares an automatic classification against Sensem classes using several scores
	:param outputs: folder containing clustering files (pickle objects)
	:param results: folder where the file with results is written
	:param file: file with information about the sensem classes
	:return: nothing
	'''


	abstractionLevels = ['sensem', 'medium', 'abstract']

	docClases = open(fileV, 'r')

	for claseRow in range(len(abstractionLevels)):
		print(abstractionLevels[claseRow])
		comp = open(os.path.join(results, 'comparacionSensemClassesF1{0}.csv'.format(abstractionLevels[claseRow])), 'w')
		comp.write('file_evaluado,tipo_clustering,Nclasses\n')

		classificationTime = {}  # verb, class
		for line in docClases:
			lista = line.split(',')
			classificationTime[lista[0]] = lista[claseRow+1].strip('\n')
		docClases.seek(0)

		possibilitiesGold, labelsGold = [classificationTime.values()], classificationTime.keys()

		for file in os.listdir(outputs):  # clusterings made with file
			print('evaluating clusters over file: ', file)
			f = open(outputs + file, 'r')
			clustering_dic = pickle.load(f)

			for data, clustering in clustering_dic.iteritems():  # info, clustering
				ldata = data.split('_')
				numC = ldata[1]
				if int(numC) in [i for i in range(2, 201,3)]:
					clusterDict = clustering[0]

					f1score = evalComparativaAveraged_F1(data, clusterDict, labelsGold, possibilitiesGold)
					accuracyScore = evalComparativaAccuracy(data, clusterDict, labelsGold, possibilitiesGold)
					vmeasure = evalComparativaVmeasure(data, clusterDict, labelsGold, possibilitiesGold)

					for k in f1score:

						comp.write(','.join([file, data, k])+',')
						for g in f1score[k]:
							comp.write(str(round(g, 3)) + ',')
						for acc in accuracyScore[k]:
							comp.write(str(round(acc, 3))+ ',')
						for v in vmeasure[k]:
							comp.write(str(round(v, 3)) + ',')

						comp.write('\n')
			f.close()
		comp.close()
	docClases.close()


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('-o', '--output',  default='../evalresults/comparison/def/',
						help='Output folder for the evaluation files with results')
	parser.add_argument('-i', '--input',  default='../clusterings/2ndBatch/',
						help='Folder that contains the clusterings')
	parser.add_argument('-c', '--classes', default = '../Auxdata/SensemClases.csv')

	args = parser.parse_args()
	GoldComparSenSemClass(args.input, args.output, args.classes)


if __name__ == '__main__':
	main()

